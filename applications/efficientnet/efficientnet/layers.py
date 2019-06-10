import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.utils import get_custom_objects


class Swish(KL.Layer):

    def call(self, inputs):
        return tf.nn.swish(inputs)


class DropConnect(KL.Layer):

    def __init__(self, drop_connect_rate=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate
            inv_keep_prob = 1.0 / keep_prob

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype) + keep_prob
            return inputs * (tf.floor(random_tensor) * inv_keep_prob)

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config['drop_connect_rate'] = self.drop_connect_rate
        return config


get_custom_objects().update({
    'DropConnect': DropConnect,
    'Swish': Swish,
})
