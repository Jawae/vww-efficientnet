# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential, Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    AveragePooling2D, Add, Input, InputLayer, DepthwiseConv2D, Multiply
# noinspection PyUnresolvedReferences
from applications import mobilenet_v2, efficientnet
# from keras_applications import mobilenet_v2, set_keras_submodules
# import sys
# set_keras_submodules(**{mod: sys.modules["tensorflow.python.keras." + mod]
#                         for mod in ["backend", "layers", "models", "utils"]})


def get_model_big(num_classes=10, img_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=img_shape, activation='relu', name="Conv1"))
    # model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', name="Conv2"))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name="Conv3"))
    # model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', name="Conv4"))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name="FC1"))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="FC2"))
    # model.add(Activation('softmax'))

    return model


def get_model_small(num_classes=10, img_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=img_shape, name="Conv1"))
    model.add(Conv2D(16, (3, 3), activation='relu', name="Conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, use_bias=False, name="FC1"))
    model.add(BatchNormalization())  # Layer preceding batchnorm must have no activation or bias
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="FC2"))

    return model


def get_mobilenet_v2(num_classes=1000, img_shape=(128, 128, 3)):
    # TODO: build a new top of num_classes != 1000
    return mobilenet_v2.MobileNetV2(input_shape=img_shape, alpha=0.35,
                                    weights="imagenet" if num_classes == 1000 else None, classes=num_classes)


def get_resnet18(num_classes=10, img_shape=(32, 32, 3)):
    def residual_block(i, out_ch, k_size, stride, idx):
        # print("Adding Residual Layer #", idx)
        if out_ch != i.shape[-1] or stride != 1:  # in/out channels don't match
            # print("applying shortcut")
            x_in = Conv2D(out_ch, kernel_size=(1, 1), strides=(stride, stride), padding='same', use_bias=False, name=("shortcut" + str(idx)))(i)
            x_in = BatchNormalization(fused=False)(x_in)
        else:
            x_in = i

        x = Conv2D(out_ch, kernel_size=(k_size, k_size), strides=(stride, stride), padding='same', use_bias=False, name=("Residual" + str(idx) + "in"))(i)
        x = BatchNormalization(fused=False)(x)
        x = Activation('relu')(x)

        x = Conv2D(out_ch, kernel_size=(k_size, k_size), strides=(1, 1), padding='same', use_bias=False, name=("Residual" + str(idx) + "out"))(x)
        x = BatchNormalization(fused=False)(x)

        return Activation('relu')(Add()([x_in, x]))

    def network(i):
        x = Conv2D(64, kernel_size=(3, 3), use_bias=False, padding='same', name="Conv1")(i)
        x = BatchNormalization(fused=False)(x)
        x = Activation('relu')(x)

        x = residual_block(x, 64, 3, stride=1, idx=2)
        x = residual_block(x, 64, 3, stride=1, idx=3)

        x = residual_block(x, 128, 3, stride=2, idx=4)
        x = residual_block(x, 128, 3, stride=2, idx=5)

        x = residual_block(x, 256, 3, stride=2, idx=6)
        x = residual_block(x, 256, 3, stride=2, idx=7)

        x = residual_block(x, 512, 3, stride=2, idx=8)
        x = residual_block(x, 512, 3, stride=2, idx=9)

        # x = AveragePooling2D()(x)

        x = Flatten()(x)

        x = Dense(num_classes, activation='softmax', name="FC2")(x)

        return x

    image_tensor = Input(shape=img_shape)
    network_output = network(image_tensor)

    return Model(inputs=[image_tensor], outputs=[network_output])


def get_tinyResNet(num_classes=10, img_shape=(32, 32, 3)):
    def residual_block(i, out_ch, k_size, stride, idx):
        # print("Adding Residual Layer #", idx)
        if out_ch != i.shape[-1] or stride != 1:  # in/out channels don't match
            # print("applying shortcut")
            x_in = Conv2D(out_ch, kernel_size=(1, 1), strides=(stride, stride), padding='same', use_bias=False, name=("shortcut" + str(idx)))(i)
            x_in = BatchNormalization(fused=False)(x_in)
        else:
            x_in = i

        x = Conv2D(out_ch, kernel_size=(k_size, k_size), strides=(stride, stride), padding='same', use_bias=False, name=("Residual" + str(idx) + "in"))(i)
        x = BatchNormalization(fused=False)(x)
        x = Activation('relu')(x)

        x = Conv2D(out_ch, kernel_size=(k_size, k_size), strides=(1, 1), padding='same', use_bias=False, name=("Residual" + str(idx) + "out"))(x)
        x = BatchNormalization(fused=False)(x)

        return Activation('relu')(Add()([x_in, x]))

    def network(i):
        x = Conv2D(64, kernel_size=(3, 3), use_bias=False, padding='same', name="Conv1")(i)
        x = BatchNormalization(fused=False)(x)
        x = Activation('relu')(x)

        x = residual_block(x, 64, 3, stride=2, idx=1)
        x = residual_block(x, 128, 3, stride=2, idx=2)
        x = residual_block(x, 256, 3, stride=2, idx=3)
        x = residual_block(x, 512, 3, stride=2, idx=4)

        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name="FC")(x)

        return x

    image_tensor = Input(shape=img_shape)
    network_output = network(image_tensor)

    return Model(inputs=[image_tensor], outputs=[network_output])


def get_efficientnet_b0(num_classes=10, img_shape=(224, 224, 3)):
    return efficientnet.EfficientNetB0(input_shape=img_shape,
                                       weights="imagenet" if num_classes == 1000 else None,
                                       classes=num_classes)


def get_efficientnet_bz(num_classes=10, img_shape=(224, 224, 3)):
    from applications.efficientnet.efficientnet.params import efficientnet as net_params
    block_params, global_params = net_params(width_coefficient=0.35, depth_coefficient=0.2,
                                             dropout_rate=0.2, drop_connect_rate=0.2)
    global_params = global_params._replace(**{'num_classes': num_classes, 'l2_reg': 0.00001})
    return efficientnet.EfficientNet(img_shape, block_params, global_params)


MODELS = {
    "small": get_model_small,
    "big": get_model_big,
    "resnet18": get_resnet18,
    "tinyresnet": get_tinyResNet,
    "mobilenet": get_mobilenet_v2,
    "efficientnet_b0": get_efficientnet_b0,
    "efficientnet_bz": get_efficientnet_bz,
}


def compute_model_stats(model):
    """Computes Peak Memory Usage, Model Size and Inference Cost (in MACs) of a Keras model."""
    bytes_per_parameter = 1
    peak_memory_usage = 0
    model_size = 0
    inference_cost = 0

    def tensor_size(tensor):
        size = 1
        for dim in tensor.shape:
            if dim.value is not None:
                size *= int(dim)
        return size

    for m in model.layers:
        if isinstance(m, InputLayer):
            continue
        model_size += bytes_per_parameter * sum(tensor_size(w) for w in m.weights)
        if isinstance(m, Conv2D) or isinstance(m, DepthwiseConv2D):
            inference_cost += tensor_size(m.output) // int(m.output.shape[-1]) * tensor_size(m.weights[0])
            if m.bias is not None:
                inference_cost += tensor_size(m.output)
        elif isinstance(m, (Add, Multiply)):
            inference_cost += sum(tensor_size(w) for w in m.input[1:])
        elif isinstance(m, Dense):
            inference_cost += tensor_size(m.input) * m.units
            if m.bias is not None:
                inference_cost += tensor_size(m.output)

        # Wrong --- doesn't support parallel branches correctly
        inputs = [m.input] if not isinstance(m.input, list) else m.input
        outputs = [m.output] if not isinstance(m.output, list) else m.output
        mem_usage = sum(tensor_size(w) for w in inputs + outputs)
        print(m.name, mem_usage)
        peak_memory_usage = max(peak_memory_usage, mem_usage)

    return peak_memory_usage, model_size, inference_cost


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    model = get_efficientnet_bz(2, (128, 128, 3))
    model.summary()
    stats = compute_model_stats(model)
    plot_model(model, "model.png")
    print(stats, stats <= (250_000, 250_000, 60_000_000))
