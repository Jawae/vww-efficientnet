# The following will override tf.contrib.quantize with our implementation (see tf_quantize_patcher/quantize/python).
# Has to be imported before tensorflow (or anything that uses tensorflow).
import tf_quantize_patcher
import tensorflow as tf
import tensorflow.contrib.quantize as tf_quantize
# For version consistency, use Keras from tensorflow (rather than having 2 moving targets).
# IntelliJ / PyCharm also fail to cope with nested TF's imports
# noinspection PyUnresolvedReferences
import tensorflow.keras.backend as K

import argparse
import os
import ast

from tensorpack import PredictConfig, ModelSaver, SyncMultiGPUTrainerParameterServer, \
    DataParallelInferenceRunner, ScalarStats, ScheduledHyperParamSetter, StatMonitorParamSetter, OfflinePredictor
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import logger, gpu, argtools, fs
from tensorpack.dataflow import BatchData
from tensorpack.contrib.keras import KerasModel

from tensorflow import compat as tf_compat  # TOCO doesn't support FusedBatchNormV3 yet
from tqdm import tqdm

from tflite_tools import create_tflite_model, cluster_tflite_model_weights, evaluate_tflite_model

import models
import datasets


@argtools.memoized
def get_model_func(mode, model_name, quant_type, img_shape, num_classes, quant_delay=0):
    assert mode in ["train", "eval"]

    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g).as_default(), tf_compat.forward_compatibility_horizon(2019, 6, 5):
            # Just trying to get names, don't pollute the default graph...
            K.set_learning_phase(mode == "train")
            model = models.MODELS[model_name](num_classes, img_shape)
            input_spec = [tf.TensorSpec(t.shape, t.dtype, t.name[:-2]) for t in model.inputs]
            output_spec = [tf.TensorSpec(t.shape, t.dtype, t.name[:-2]) for t in model.outputs]

    def model_func(image):
        K.set_learning_phase(mode == "train")
        with tf_compat.forward_compatibility_horizon(2019, 6, 5):
            m = tf.keras.models.clone_model(model, input_tensors=image)
            if quant_type != "none":
                with tf.variable_scope('quants', reuse=tf.AUTO_REUSE):
                    if mode == "train":
                        tf_quantize.experimental_create_training_graph(tf.get_default_graph(), quant_delay=quant_delay, quant_type=quant_type)
                    else:
                        tf_quantize.experimental_create_eval_graph(tf.get_default_graph(), quant_type=quant_type)
        return m

    return model_func, input_spec, output_spec


def train(checkpoint_dir, model_name, dataset, num_epochs, quant_type, batch_size_per_gpu, lr=None, post_quantize_only=False):
    train_data, test_data, (img_shape, label_shape) = datasets.DATASETS[dataset]()

    num_gpus = max(gpu.get_num_gpu(), 1)
    effective_batch_size = batch_size_per_gpu * num_gpus
    train_data = BatchData(train_data, batch_size_per_gpu)
    test_data = BatchData(test_data, batch_size_per_gpu, remainder=True)
    steps_per_epoch = len(train_data) // num_gpus

    if lr:
        if isinstance(lr, str):
            lr = ast.literal_eval(lr)
        if isinstance(lr, float):
            lr_schedule = [(0, lr)]
        else:
            lr_schedule = lr
    else:
        lr_schedule = [(0, 0.005), (8, 0.1), (25, 0.005), (30, 0)]

    if num_epochs is None:
        num_epochs = lr_schedule[-1][0]
    if post_quantize_only:
        start_quantising_at_epoch = 0
    else:
        start_quantising_at_epoch = lr_schedule[-2][0] if len(lr_schedule) > 1 else max(0, num_epochs - 5)

    logger.info(f"Training with LR schedule: {str(lr_schedule)}")
    logger.info(f"Quantising at epoch {start_quantising_at_epoch}")

    # train_data = FakeData([(batch_size_per_gpu,) + img_shape, (batch_size_per_gpu, ) + label_shape])

    model_func, input_spec, output_spec = get_model_func("train", model_name, quant_type, img_shape,
                                                         num_classes=label_shape[0],
                                                         quant_delay=steps_per_epoch * start_quantising_at_epoch)
    target_spec = [tf.TensorSpec(t.shape, t.dtype, name=t.name.split("/")[-1] + "_target") for t in output_spec]
    model = KerasModel(get_model=model_func,
                       input_signature=input_spec,
                       target_signature=target_spec,
                       input=train_data,
                       trainer=SyncMultiGPUTrainerParameterServer(num_gpus, ps_device='gpu'))

    lr = tf.get_variable('learning_rate', initializer=lr_schedule[0][1], trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)
    model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9),
                  loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])

    model.fit(steps_per_epoch=steps_per_epoch,
              max_epoch=num_epochs,
              callbacks=[
                  ModelSaver(max_to_keep=1, checkpoint_dir=checkpoint_dir),
                  DataParallelInferenceRunner(test_data, ScalarStats(model._stats_to_inference), num_gpus),
                  ScheduledHyperParamSetter('learning_rate', lr_schedule, interp="linear"),
                  StatMonitorParamSetter('learning_rate', 'validation_categorical_accuracy',
                                         lambda x: x / 2, threshold=0.001, last_k=10, reverse=True)
              ],
              session_init=SaverRestore(checkpoint_dir + "/checkpoint") if post_quantize_only else None)


def export_eval_protobuf_model(checkpoint_dir, model_name, dataset, quant_type, output_file, batch_size):
    _, test_data, (img_shape, label_shape) = datasets.DATASETS[dataset]()

    model_func, input_spec, output_spec = get_model_func("eval", model_name, quant_type, img_shape, label_shape[0])
    input_names = [i.name for i in input_spec]
    output_names = [o.name for o in output_spec]
    predictor_config = PredictConfig(
        session_init=SaverRestore(checkpoint_dir + "/checkpoint"),
        tower_func=model_func,
        input_signature=input_spec,
        input_names=input_names,
        output_names=output_names,
        create_graph=False)

    print("Exporting optimised protobuf graph...")
    K.set_learning_phase(False)
    ModelExporter(predictor_config).export_compact(output_file, optimize=False)

    K.clear_session()
    pred = OfflinePredictor(predictor_config)

    test_data = BatchData(test_data, batch_size, remainder=True)
    test_data.reset_state()

    num_correct = 0
    num_processed = 0
    for img, label in tqdm(test_data):
        num_correct += sum(pred(img)[0].argmax(axis=1) == label.argmax(axis=1))
        num_processed += img.shape[0]

    print("Exported model has accuracy {:.4f}".format(num_correct / num_processed))

    return input_names, output_names, {i.name: i.shape for i in input_spec}


def main():
    parser = argparse.ArgumentParser(description='CortexML')

    parser.add_argument("--arch", type=str, default="small", help="Architecture to use")
    parser.add_argument("--dataset", type=str, default="MNIST", help="The dataset you want to use",
                        choices=datasets.DATASETS.keys())
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--quant", type=str, default="affine", choices=["affine", "symmetric", "none"])
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument("--weight-clusters", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU to use")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./train_log")
    parser.add_argument("--lr", type=str, default=None, help="Learning rate or learning rate schedule")
    parser.add_argument("--post-quantize", action='store_true', help="Will load a pretrained model and quantize for the specified amount of epochs")
    args = parser.parse_args()

    logger.set_logger_dir(args.log_dir, action='n')
    fs.mkdir_p(args.checkpoint_dir)

    # 1. Train the model (if it doesn't exist)
    if args.no_cache or not os.path.exists(args.checkpoint_dir + "/checkpoint") or args.post_quantize:
        print("Model not found, training...")
        train(args.checkpoint_dir, model_name=args.arch, dataset=args.dataset, num_epochs=args.epochs,
              quant_type=args.quant, batch_size_per_gpu=args.batch, lr=args.lr, post_quantize_only=args.post_quantize)
        print("Model training complete.")
        K.clear_session()
    else:
        print("Using a model in {}. Delete the corresponding folder to start afresh.".format(args.checkpoint_dir))

    if args.quant == "none":
        return

    # 2. Evaluate the model after training
    # TODO

    # 3. Convert quantized model into a simplified protobuf graphdef
    protobuf_file = args.checkpoint_dir + "/compact_graph.pb"
    print("Exporting quantized model as a protobuf file:", protobuf_file)
    predictor_tensors = export_eval_protobuf_model(args.checkpoint_dir, model_name=args.arch,
                                                   dataset=args.dataset, quant_type=args.quant,
                                                   output_file=protobuf_file, batch_size=args.batch)
    inputs, outputs, input_shapes = predictor_tensors

    # 4. Convert the model into tflite format
    tflite_model_file = args.checkpoint_dir + "/quantized_model.tflite"
    print("Converting quantized model into a tflite file:", tflite_model_file)
    create_tflite_model(protobuf_file, inputs, outputs, input_shapes, tflite_model_file)
    if args.weight_clusters > 0:
        cluster_tflite_model_weights(tflite_model_file, args.weight_clusters)
        tflite_model_file = args.checkpoint_dir + "/clustered_quantized_model.tflite"

    evaluate_tflite_model(tflite_model_file, dataset=args.dataset)


if __name__ == "__main__":
    main()
