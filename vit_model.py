import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


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


class Encoder(layers.Layer):
    """Patch Encoder"""
    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        num_patch = (input_shape[1] * input_shape[2]) // (self.patch_size**2)
        # weight
        self.patch_weight = self.add_weight(
            name  = 'patch_posemb',
            shape = (1, 1, self.hidden_dim)
        )
        self.emb_weight = self.add_weight(
            name  = 'patch_emb',
            shape = (1, num_patch + 1, self.hidden_dim)
        )

        # layer
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
        batch_size = tf.shape(inputs)[0]
        patch_weight = tf.repeat(
            input   = self.patch_weight,
            repeats = batch_size,
            axis    = 0
        )

        x = self.conv(inputs)
        x = self.reshape(x)
        x = layers.Concatenate(
            axis = 1,
            name = 'patch_posemb_concat')([x, patch_weight])
        x = layers.Add(
            name = 'patch_patchemb_add')([x, self.emb_weight])
        return x


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
        self.mha = layers.MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim   = self.hidden_dim,
            dropout   = self.attn_drop_rate,
            name      = 'transformer_multihedattension'
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
        x = self.mha(x, x)
        x = layers.Add(name='transformer_MHA_add')([inputs, x])

        y = self.layernorm2(x)
        y = self.mlp(y)

        x = layers.Add(name='transformer_MLP_add')([x, y])
        return x

class Pop(layers.Layer):
    """Partial layer"""
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
    ) -> Model:
    """Build vision transformer."""

    assert image_size % patch_size == 0

    # input
    inputs = layers.Input(
        shape = (image_size, image_size, channels),
        name = 'Input')

    # patch encoder
    patch_enc = Encoder(
        patch_size = patch_size,
        hidden_dim = hidden_dim,
        name       = 'Patch_Encoder'
    )(inputs)

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
        name    = 'Final_layernorm')(patch_enc)
    x = Pop(name='Final_pop')(x)
    outputs = layers.Dense(
        units      = classes,
        activation = 'softmax',
        name       = 'Output'
        )(x)
    model = Model(inputs, outputs)

    return model

# config
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


def ViT_S16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        **vitS_conf)


def ViT_S32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        **dict(vitS_conf, **{'patch_size':32}))


def ViT_B16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        **vitB_conf)


def ViT_B32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        **dict(vitB_conf, **{'patch_size':32}))


def ViT_L16(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        **vitB_conf)


def ViT_L32(
    image_size: int,
    channels:   int,
    classes:    int) -> Model:
    return build_vit(
        image_size  = image_size,
        channels    = channels,
        classes     = classes,
        **dict(vitL_conf, **{'patch_size':32}))


if not __debug__:
    model = ViT_L32(256, 3, 20)
    model.summary()
