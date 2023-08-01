import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.activations import sigmoid

KERNEL_INITIALIZER = 'he_normal'
KERNEL_REGULARUZER = tf.keras.regularizers.l2(1e-3)

class PreNorm(tf.keras.layers.Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs[0], inputs[1], inputs[1], None
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, [batch_size, -1, self.d_model])
        outputs = self.dense(concat_attention)
        return outputs

class MLP(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()

        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return tf.keras.layers.Activation(gelu)

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(units=hidden_dim),
            GELU(),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(units=dim),
            tf.keras.layers.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class WeightedAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.nn.softmax(self.w, axis=-1)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(WeightedAdd, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

class IntensitiyTransform(tf.keras.layers.Layer):

    def __init__(self, intensities, channels, **kwargs):
        super(IntensitiyTransform, self).__init__(**kwargs)
        self.channels = channels
        self.scale = intensities - 1

    def call(self, inputs):
        x = tf.map_fn(self.intensity_transform, inputs, dtype='float32')
        return x

    def get_config(self):
        config = super(IntensitiyTransform, self).get_config()
        config.update({'channels': self.channels, 'scale': self.scale})
        return config

    def intensity_transform(self, inputs):
        images, transforms = inputs
        images = 0.5 * images + 0.5
        images = tf.cast(tf.math.round(self.scale * images), dtype='int32')
        images = tf.split(images, num_or_size_splits=self.channels, axis=-1)
        transforms = tf.split(transforms, num_or_size_splits=self.channels, axis=-1)
        x = tf.concat([tf.gather_nd(image, transform) for image, transform in zip(transforms, images)], axis=-1)

        return x

class LuminanceAttention(tf.keras.Model):

    def __init__(self, seq_len, dimension, **kwargs):
        super(LuminanceAttention, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(d_model=256, num_heads=8)
        self.self_attention = MultiHeadAttention(d_model=256,num_heads=8)

        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dropout3 = tf.keras.layers.Dropout(0.1)

        self.mlp = MLP(256, 1024)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def tile_queries(self, input_tensor, training=False):
        batch_size = tf.shape(input_tensor)[0]
        cluster_centers = tf.tile(self.ins_query, [batch_size, 1, 1])
        return cluster_centers

    def call(self, inputs, training=True, **kwargs):
        global_feature = inputs[0]
        luminance_feature = inputs[1]

        input_query = global_feature

        output2 = self.attention([input_query, luminance_feature])
        output = input_query + self.dropout1(output2)
        output = self.layernorm1(output)

        output2 = self.self_attention([output, output])
        output = output + self.dropout3(output2)
        output = self.layernorm3(output)

        output2 = self.mlp(output)
        output = output + self.dropout2(output2)
        output = self.layernorm2(output)

        return output

def scaled_dot_product_attention(query, key, value, mask):

    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights

def conv_block(inputs, filters, kernel_size, strides, dropout_rate=0.1, name=''):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides,
                               padding='same', use_bias=False,
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name + '_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_norm')(x)
    x = tf.keras.layers.Activation('swish', name=name + '_act')(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, (None, 1, 1, 1), name=name + '_dropout')(x)
    return x

def build_vgg16():
    vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                              weights='imagenet',
                                              input_tensor=None,
                                              input_shape=(None, None, 3),
                                              pooling=None)
    im = vgg16.input
    x1 = vgg16.get_layer('block1_conv2').output
    x2 = vgg16.get_layer('block2_conv2').output
    x3 = vgg16.get_layer('block3_conv2').output
    return tf.keras.Model(im, [x1, x2, x3])

def inverted_residual_block(inputs, filters_in=32, filters_out=16,
                            kernel_size=3, strides=1, expand_ratio=6,
                            se_ratio=0.25, dropout_rate=0.1, name=''):
    filters = filters_in * expand_ratio
    x = tf.keras.layers.Conv2D(filters, 1, use_bias=False,
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name + '_expand_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_expand_bn')(x)
    x = tf.keras.layers.Activation('swish', name=name + '_expand_swish')(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides,
                                        padding='same', use_bias=False,
                                        depthwise_initializer=KERNEL_INITIALIZER,
                                        depthwise_regularizer=KERNEL_REGULARUZER,
                                        name=name + '_dwconv')(x)
    x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
    x = tf.keras.layers.Activation('swish', name=name + '_swish')(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(name=name + '_se_squeeze')(x)
        se = tf.keras.layers.Reshape((1, 1, filters), name=name + '_se_reshape')(se)
        se = tf.keras.layers.Conv2D(filters_se, 1, use_bias=False,
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    kernel_regularizer=KERNEL_REGULARUZER,
                                    name=name + '_se_reduce_conv')(se)
        se = tf.keras.layers.Activation('swish', name=name + '_se_expand_swish')(se)
        se = tf.keras.layers.Conv2D(filters, 1, use_bias=False,
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    kernel_regularizer=KERNEL_REGULARUZER,
                                    name=name + '_se_expand_conv')(se)
        se = tf.keras.layers.Activation('sigmoid', name=name + '_se_expand_sigmoid')(se)
        x = tf.keras.layers.Multiply(name=name + '_se_excite')([x, se])

    x = tf.keras.layers.Conv2D(filters_out, 1, use_bias=False,
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name=name + '_project_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=name + '_project_bn')(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, (None, 1, 1, 1), name=name + '_dropout')(x)

    return x

def upsample_block(inputs, upsample_size, name=''):
    x_prev, x_skip = inputs
    print(x_prev.shape, x_skip.shape)
    x_prev = tf.keras.layers.UpSampling2D(upsample_size, interpolation='bilinear', name=name + '_up')(x_prev)
    _, x_h, x_w, _ = tf.shape(x_skip)
    z = tf.keras.layers.Concatenate(name=name + '_concat')([x_prev[:, :x_h, :x_w, :], x_skip])
    return z

def lightnet_v6_new():
    inputs = keras.layers.Input(shape=(192, 192, 16,), dtype='float32')

    x = conv_block(inputs, 16, 5, 1, dropout_rate=0.3, name='lightnet1')
    x = conv_block(x, 16, 7, 1, dropout_rate=0.3, name='lightnet2')
    x = conv_block(x, 16, 7, 1, dropout_rate=0.3, name='lightnet3')
    x = conv_block(x, 16, 5, 1, dropout_rate=0.3, name='lightnet4')

    x = conv_block(x, 32, 5, 2, name='attention_conv_1')
    x = conv_block(x, 64, 7, 2, name='attention_conv_2')
    x = conv_block(x, 256, 5, 2, name='attention_conv_3')

    luminance_feature_flatten = tf.reshape(x, [-1, 24 * 24, 256])

    fc = keras.layers.GlobalAveragePooling2D()(x)

    fc = tf.keras.layers.Reshape((1, 1, 256))(fc)
    fc = tf.keras.layers.Conv2D(256, 1, 1, use_bias=False,
                                kernel_initializer=KERNEL_INITIALIZER,
                                kernel_regularizer=KERNEL_REGULARUZER)(fc)
    fc = tf.squeeze(fc, [1, 2])
    model = keras.Model(inputs=inputs, outputs=(fc, x, luminance_feature_flatten))

    return model

def build_luminance_transform_function(lightnet, attn_net):
    x = tf.keras.Input((None, None, 3), dtype='float32')
    y = tf.keras.layers.Lambda(tf.image.resize, arguments={'size': [384, 384]}, name='input_resize')(x)

    y = conv_block(y, 16, 5, 2, dropout_rate=0.1, name='stage1')

    l1, x1, luminance_feature = lightnet(y)

    y = inverted_residual_block(y, 32, 40, 5, 2, dropout_rate=0.3, name='stage2')
    y = inverted_residual_block(y, 40, 40, 5, 2, dropout_rate=0.3, name='stage3')
    y = inverted_residual_block(y, 40, 80, 5, 2, dropout_rate=0.3, name='stage4')
    y = inverted_residual_block(y, 80, 112, 5, 2, dropout_rate=0.3, name='stage5')
    y = conv_block(y, 128, 3, 2, dropout_rate=0.15, name='stage6')

    y = conv_block(y, 256, 1, 1, dropout_rate=0.1, name='stage7')
    y = tf.reshape(y, [-1, 6 * 6, 256])

    y = attn_net([y, luminance_feature])
    y = tf.keras.layers.Reshape((256, 36), name='rchannel_reshape')(y)
    y = IntensitiyTransform(256, 3, name='intensity_transform')([x, y])

    return tf.keras.Model(x, (y, l1), name='global_enhancement_network')

def post_processing_module():
    x = tf.keras.Input((None, None, 36), dtype='float32')

    z1 = conv_block(x, 16, 5, 1, dropout_rate=0.1, name='stage1')

    z2 = inverted_residual_block(z1, 16, 24, 5, 1, dropout_rate=0.3, name='stage2')
    z3 = inverted_residual_block(z2, 24, 40, 5, 1, dropout_rate=0.3, name='stage3')
    z4 = inverted_residual_block(z3, 40, 80, 5, 1, dropout_rate=0.3, name='stage4')
    z5 = inverted_residual_block(z4, 80, 24, 5, 1, dropout_rate=0.0, name='stage5')
    z6 = inverted_residual_block(z5, 24, 16, 5, 1, dropout_rate=0.0, name='stage6')

    y = tf.keras.layers.Conv2D(36, 5, padding='same',
                               kernel_initializer=KERNEL_INITIALIZER,
                               kernel_regularizer=KERNEL_REGULARUZER,
                               name='stage7_conv')(z6)
    y = WeightedAdd(name='add')([x, y], name='aggregation')
    y = tf.keras.layers.Conv2D(3, 1,padding='same')(y)
    y = tf.keras.layers.Activation('tanh', name='stage7_tanh')(y)
    return tf.keras.Model(x, outputs=y, name='local_enhancement_network')
    
