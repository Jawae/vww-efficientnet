import numpy as np

from tflite import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType
import tensorflow.lite as tf_lite
from tqdm import tqdm

from sklearn import cluster

import datasets


def cluster_weights(weights, n_clusters):
    kmeans = cluster.KMeans(n_clusters=n_clusters).fit(weights.reshape((-1, 1)))
    return kmeans.labels_.reshape(weights.shape), np.around(kmeans.cluster_centers_).astype(np.int32)


# Flatbuffers provide a per-byte view on data, so we need to cast the underlying buffer to the correct datatype
def get_buffer_as_numpy(tensor, buffer):
    if tensor.Type() == TensorType.UINT8:
        arr = buffer.DataAsNumpy()
    elif tensor.Type() == TensorType.INT16:
        arr = np.frombuffer(buffer.DataAsNumpy(), dtype=np.dtype(np.int16).newbyteorder("<"))
    elif tensor.Type() == TensorType.INT32:
        arr = np.frombuffer(buffer.DataAsNumpy(), dtype=np.dtype(np.int32).newbyteorder("<"))
    elif tensor.Type() == TensorType.INT64:
        arr = np.frombuffer(buffer.DataAsNumpy(), dtype=np.dtype(np.int64).newbyteorder("<"))
    else:
        raise NotImplementedError()
    return arr.reshape(tensor.ShapeAsNumpy())


def discover_tflite_weights(model_bytes):
    model = Model.Model.GetRootAsModel(model_bytes, 0)
    subgraph = model.Subgraphs(0)

    weights = []
    for o in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(o)
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        inputs = op.InputsAsNumpy()

        parametrised_opcodes = [BuiltinOperator.CONV_2D, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEPTHWISE_CONV_2D]
        if opcode not in parametrised_opcodes:
            continue

        weight_tensor = subgraph.Tensors(inputs[1])
        buffer_idx = weight_tensor.Buffer()
        buffer = model.Buffers(buffer_idx)
        # Return a buffer index and contents as an ndarray
        weights.append((buffer_idx, get_buffer_as_numpy(weight_tensor, buffer)))

    return weights


def overwrite_flatbuffers_buffer(serialized_model, buffer_idx, new_contents):
    model = Model.Model.GetRootAsModel(serialized_model, 0)
    orig_buffer = model.Buffers(buffer_idx)
    # NB. Update this to directly manipulate `serialized_model` if this view becomes unwriteable
    orig_buffer.DataAsNumpy()[:] = new_contents.astype(np.uint8).flatten()


def cluster_tflite_model_weights(tflite_model_file, weight_clusters=0):
    with open(tflite_model_file, 'rb') as f:
        model_bytes = bytearray(f.read())
        print("Clustering weights into {} clusters...".format(weight_clusters))
        weights = discover_tflite_weights(model_bytes)
        for b_index, weight in weights:
            assignments, centroids = cluster_weights(weight, weight_clusters)
            overwrite_flatbuffers_buffer(model_bytes, b_index,
                                         np.squeeze(centroids[assignments], axis=-1))
        tflite_model_file = "clustered_" + tflite_model_file
        with open(tflite_model_file, 'wb') as f:
            f.write(model_bytes)


def evaluate_tflite_model(tflite_model_file, dataset):
    print("Evaluating quantized model...")
    _, test_data, _ = datasets.DATASETS[dataset]()
    test_data.reset_state()

    interpreter = tf_lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    input_index = input_info["index"]
    scale, offset = input_info["quantization"]
    output_index = interpreter.get_output_details()[0]["index"]

    correct = 0
    for img, label in tqdm(test_data):
        interpreter.set_tensor(input_index, np.expand_dims(img.astype(np.uint8), axis=0))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        if predictions.argmax() == label.argmax():
            correct += 1
    print("{} classified correctly out of {} ({:.2f}%)".format(correct, len(test_data), correct / len(test_data) * 100))


def create_tflite_model(protobuf_file, inputs, outputs, input_shapes, output_file):
    """
    Quantises weight matrices and biases, and exports the model in the TFLite format.
    """
    converter = tf_lite.TFLiteConverter.from_frozen_graph(protobuf_file, input_arrays=inputs,
                                                          output_arrays=outputs, input_shapes=input_shapes)
    from tensorflow.lite.python import lite_constants
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    # converter.optimizations = [tf_lite.Optimize.DEFAULT]
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0, 1)}  # mean, std_dev
    with open(output_file, "wb") as f:
        f.write(converter.convert())
    print("Done!")
