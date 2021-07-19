import os, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import tensorflow as tf

from tensorflow.python.keras import layers
from tensorflow.python.keras import Model


class MLP(layers.Layer):
    """MultiLayer Perceptron"""
    def __init__(
        self,
        hidden_dim:         int,
        hidden_dim_mlp:     int,
        drop_rate:          float,
        **kwargs) -> None:
        super(MLP, self).__init__(**kwargs)

        self.drop_rate      = drop_rate
        self.hidden_dim     = hidden_dim
        self.hidden_dim_mlp = hidden_dim_mlp

    def build(self, input_shape):
        self.dense1 = layers.Dense(
            units      = self.hidden_dim_mlp,
            activation = tf.nn.gelu,
            name       = 'mlp_dense1'
        )
        self.drop1 = layers.Dropout(
            rate = self.drop_rate,
            name ='mlp_drop1'
        )
        self.dense2 = layers.Dense(
            units = self.hidden_dim,
            name  = 'mlp_dense2'
        )
        self.drop2 = layers.Dropout(
            rate = self.drop_rate,
            name ='mlp_drop2'
        )
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return x


class Patch(layers.Layer):
    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        **kwargs) -> None:
        super(Patch, self).__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
    
    def build(self, input_shape):
        num_patch = (input_shape[1] * input_shape[2]) // (self.patch_size**2)

        self.conv = layers.Conv2D(
            filters     = self.hidden_dim,
            kernel_size = self.patch_size,
            strides     = self.patch_size,
            name        = 'patch_conv2d'
        )
        self.reshape = layers.Reshape(
            target_shape = (num_patch, self.hidden_dim),
            name         = 'patch_reshape'
        )
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.reshape(x)

        return x


class Encoder(layers.Layer):
    """Patch & Embedding"""
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        num_patch = (self.image_size // self.patch_size)**2

        # weight
        self.patch_weight = self.add_weight(
            name        = 'patch_posemb',
            shape       = (1, 1, self.hidden_dim),
            initializer = tf.keras.initializers.zeros()
        )
        self.emb_weight = self.add_weight(
            name        = 'patch_emb',
            shape       = (1, num_patch + 1, self.hidden_dim),
            initializer = tf.keras.initializers.random_normal(stddev = 0.02)
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patch_weight = tf.broadcast_to(
            input   = self.patch_weight,
            shape   = [batch_size, 1, self.hidden_dim],
        )
        x = layers.Concatenate(
            axis = 1,
            name = 'patch_posemb_concat')([patch_weight, inputs])
        x = layers.Add(
            name = 'patch_patchemb_add')([x, self.emb_weight])
        return x


class MultiHeadSelfAttention(layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        drop_rate: float = 0.,
        **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.drop_rate = drop_rate
    
    def build(self, input_shape):      
        # Dense
        self.d_query = layers.Dense(self.hidden_dim, name='MHA_query')
        self.d_key = layers.Dense(self.hidden_dim, name='MHA_query')
        self.d_value = layers.Dense(self.hidden_dim, name='MHA_query')

        # Reshape
        self.reshape = layers.Reshape(
                target_shape=(-1, self.num_heads, self.hidden_dim//self.num_heads))
        # Perm
        self.perm = layers.Permute(dims = (2,1,3))

        # softmax
        self.softmax = layers.Activation(activation = 'softmax', name = 'qkt_softmax')

        # dropout
        self.dropout = layers.Dropout(rate = self.drop_rate)

        # for output
        self.reshape_f = layers.Reshape(target_shape = (-1, self.hidden_dim))
        self.dense = layers.Dense(units = self.hidden_dim)

    def reshape_and_perm(self, inputs):
        x = self.reshape(inputs)
        return self.perm(x)

    def call(self, inputs):
        # Dense
        query = self.reshape_and_perm(self.d_query(inputs))
        key = self.reshape_and_perm(self.d_key(inputs))
        value = self.reshape_and_perm(self.d_value(inputs))

        # qkt = QK^T
        qkt = tf.matmul(a = query, b = key, transpose_b = True)

        # qkt = softmax(qkt/sqrt(dim))
        d_k = tf.cast(self.hidden_dim, dtype=qkt.dtype)
        qkt = self.softmax(qkt / tf.sqrt(d_k))

        # dropout
        qkt = self.dropout(qkt)

        # qktV
        attn = tf.matmul(qkt, value)

        # [Batch, num_heads, patch^2+1, dim // num_heads] -> [Batch, patch^2+1, dim]
        attn = self.perm(attn)
        attn = self.reshape_f(attn)

        # final dense
        out = self.dense(attn)

        return out


class Transformer(layers.Layer):
    """Transformer block"""
    def __init__(
        self,
        num_heads:          int,
        hidden_dim:         int,
        hidden_dim_mlp:     int,
        patch_size:         int,
        mlp_drop_rate:      float,
        attn_drop_rate:     float,
        **kwargs) -> None:
        super(Transformer, self).__init__(**kwargs)

        self.num_heads      = num_heads
        self.hidden_dim     = hidden_dim
        self.hidden_dim_mlp = hidden_dim_mlp
        self.patch_size     = patch_size
        self.mlp_drop_rate  = mlp_drop_rate
        self.attn_drop_rate = attn_drop_rate

    def build(self, input_shape):
        self.layernorm1 = layers.LayerNormalization(
            epsilon = 1e-06,
            name    = 'transformer_layernorm_1'
        )
        self.mha = MultiHeadSelfAttention(
            hidden_dim = self.hidden_dim,
            num_heads  = self.num_heads,
            drop_rate  = self.attn_drop_rate,
            name       = 'transformer_multiheadattension'
        )
        self.layernorm2 = layers.LayerNormalization(
            epsilon = 1e-06,
            name    = 'transformer_layernorm_2'
        )
        self.mlp = MLP(
            hidden_dim     = self.hidden_dim,
            hidden_dim_mlp = self.hidden_dim_mlp,
            drop_rate      = self.mlp_drop_rate,
            name           = 'transformer_mlp'
        )

    def call(self, inputs):
        x = self.layernorm1(inputs)
        x = self.mha(x)
        x = layers.Add(name='transformer_MHA_add')([inputs, x])

        y = self.layernorm2(x)
        y = self.mlp(y)

        x = layers.Add(name='transformer_MLP_add')([x, y])
        return x


class MLPHead(layers.Layer):
    """MLPHead layer"""
    def call(self, inputs):
        return inputs[:,0]


def build_vit(
    image_size:             int,
    channels:               int,
    classes:                int,
    patch_size:             int,
    hidden_dim:             int,
    hidden_dim_mlp:         int,
    mlp_drop_rate:          float,
    attn_drop_rate:         float,
    num_heads:              int,
    num_transformer_layers: int,
    model_name:             str
    ) -> Model:
    """Build vision transformer."""

    assert image_size % patch_size == 0

    # input
    inputs = layers.Input(
        shape = (image_size, image_size, channels),
        name = 'Input')

    patch = Patch(
        patch_size = patch_size,
        hidden_dim = hidden_dim,
    )(inputs)

    # patch encoder
    patch_enc = Encoder(
        image_size = image_size,
        patch_size = patch_size,
        hidden_dim = hidden_dim,
        name       = 'Encoder'
    )(patch)

    # transformer
    for i in range(num_transformer_layers):
        patch_enc = Transformer(
            num_heads      = num_heads,
            hidden_dim     = hidden_dim,
            hidden_dim_mlp = hidden_dim_mlp,
            patch_size     = patch_size,
            mlp_drop_rate  = mlp_drop_rate,
            attn_drop_rate = attn_drop_rate,
            name           = 'Transformer_{:02}'.format(i+1)
        )(patch_enc)

    # final
    x = layers.LayerNormalization(
        epsilon = 1e-06,
        name    = 'Layer_Norm')(patch_enc)
    x = MLPHead(name='MLP_Head')(x)
    outputs = layers.Dense(
        units      = classes,
        activation = 'softmax',
        name       = 'Output'
        )(x)
    model = Model(inputs, outputs, name = model_name)

    return model


vitS_conf = dict(
    patch_size              = 16,
    hidden_dim              = 384,
    hidden_dim_mlp          = 1536,
    mlp_drop_rate           = 0.,
    attn_drop_rate          = 0.,
    num_heads               = 6,
    num_transformer_layers  = 12
)
vitB_conf = dict(
    patch_size              = 16,
    hidden_dim              = 768,
    hidden_dim_mlp          = 3072,
    mlp_drop_rate           = 0.,
    attn_drop_rate          = 0.,
    num_heads               = 12,
    num_transformer_layers  = 12
)
vitL_conf = dict(
    patch_size              = 16,
    hidden_dim              = 1024,
    hidden_dim_mlp          = 4096,
    mlp_drop_rate           = 0.1,
    attn_drop_rate          = 0.,
    num_heads               = 16,
    num_transformer_layers  = 24
)
vitH_conf = dict(
    patch_size              = 16,
    hidden_dim              = 1280,
    hidden_dim_mlp          = 5120,
    mlp_drop_rate           = 0.1,
    attn_drop_rate          = 0.,
    num_heads               = 16,
    num_transformer_layers  = 32
)


def ViT_S16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_S16',
        **vitS_conf)


def ViT_S32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_S32',
        **dict(vitS_conf, patch_size = 32))


def ViT_B16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_B16',
        **vitB_conf)


def ViT_B32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_B32',
        **dict(vitB_conf, patch_size = 32))


def ViT_L16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_L16',
        **vitL_conf)


def ViT_L32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_L32',
        **dict(vitL_conf, patch_size = 32))


def ViT_H16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_H16',
        **vitH_conf)


def ViT_H32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        model_name  = 'ViT_H32',
        **dict(vitH_conf, patch_size = 32))


if not __debug__:
    model = ViT_H32(256, 3, 8)
    model.summary()
