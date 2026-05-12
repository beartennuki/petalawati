from prefect import task
from prefect.cache_policies import NO_CACHE


@task(name="build-model", cache_policy=NO_CACHE)
def build_model(architecture: str, num_classes: int, image_size: int):
    builders = {
        "kernarc": _build_kernarc,
        "swiftpan": _build_swiftpan,
        "sepfuse": _build_sepfuse,
        "dualfuse": _build_dualfuse,
        "syncgen": _build_syncgen,
        "nanonet": _build_nanonet,
        "alexnet": _build_alexnet,
        "vggnet": _build_vggnet,
        "resnet": _build_resnet,
        "inceptionv3": _build_inception_v3,
        "efficientnet": _build_efficientnet,
        "vit": _build_vit,
        "densenet": _build_densenet,
        "mobilenet": _build_mobilenet,
        "swin": _build_swin,
        "convnext": _build_convnext,
    }
    builder = builders.get(architecture)
    if builder is None:
        raise ValueError(f"Unknown architecture: {architecture}")
    return builder(num_classes, image_size)


def _input_tensor(image_size: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    return inputs, x


def _classifier_head(x, num_classes: int, dense_units: int = 256, dropout: float = 0.3, pooling: str = "avg"):
    import tensorflow as tf

    if pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.layers.Dense(num_classes, activation="softmax")(x)


def _conv_bn_relu(x, filters: int, kernel: int, stride: int = 1):
    import tensorflow as tf

    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def _depthwise_block(x, pointwise_filters: int, stride: int = 1):
    import tensorflow as tf

    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Conv2D(pointwise_filters, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(6.0)(x)


def _se_block(x, ratio: int = 4):
    import tensorflow as tf

    channels = x.shape[-1]
    reduced = max(channels // ratio, 8)
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(reduced, activation="swish")(se)
    se = tf.keras.layers.Dense(channels, activation="sigmoid")(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.Multiply()([x, se])


def _mbconv_block(x, out_filters: int, expand_ratio: int = 4, stride: int = 1, se_ratio: int = 4):
    import tensorflow as tf

    in_filters = x.shape[-1]
    expanded = in_filters * expand_ratio
    shortcut = x

    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(expanded, 1, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = _se_block(x, ratio=se_ratio)
    x = tf.keras.layers.Conv2D(out_filters, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride == 1 and in_filters == out_filters:
        x = tf.keras.layers.Add()([shortcut, x])
    return x


def _residual_block(x, filters: int, stride: int = 1):
    import tensorflow as tf

    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation("relu")(x)


def _dense_block(x, growth_rate: int, repeats: int):
    import tensorflow as tf

    for _ in range(repeats):
        y = tf.keras.layers.BatchNormalization()(x)
        y = tf.keras.layers.Activation("relu")(y)
        y = tf.keras.layers.Conv2D(growth_rate * 4, 1, padding="same", use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation("relu")(y)
        y = tf.keras.layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)(y)
        x = tf.keras.layers.Concatenate()([x, y])
    return x


def _transition_block(x, compression: float = 0.5):
    import tensorflow as tf

    filters = max(int(x.shape[-1] * compression), 32)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    return tf.keras.layers.AveragePooling2D(2, strides=2, padding="same")(x)


def _inception_module(x, filters_1x1: int, filters_3x3: tuple[int, int], filters_5x5: tuple[int, int], pool_proj: int):
    import tensorflow as tf

    b1 = _conv_bn_relu(x, filters_1x1, 1)

    b2 = _conv_bn_relu(x, filters_3x3[0], 1)
    b2 = _conv_bn_relu(b2, filters_3x3[1], 3)

    b3 = _conv_bn_relu(x, filters_5x5[0], 1)
    b3 = _conv_bn_relu(b3, filters_5x5[1], 5)

    b4 = tf.keras.layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = _conv_bn_relu(b4, pool_proj, 1)

    return tf.keras.layers.Concatenate()([b1, b2, b3, b4])


def _convnext_block(x, filters: int, layer_scale: float = 1e-6):
    import tensorflow as tf

    shortcut = x
    x = tf.keras.layers.DepthwiseConv2D(7, padding="same")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(filters * 4)(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.Lambda(lambda t: t * layer_scale)(x)
    return tf.keras.layers.Add()([shortcut, x])


@__import__("tensorflow").keras.utils.register_keras_serializable(package="petalawati")
class SwinBlock(__import__("tensorflow").keras.layers.Layer):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0, mlp_ratio: float = 4.0, **kwargs):
        import tensorflow as tf

        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(dim // num_heads, 8))
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp1 = tf.keras.layers.Dense(int(dim * mlp_ratio), activation="gelu")
        self.mlp2 = tf.keras.layers.Dense(dim)

    def call(self, x):
        import tensorflow as tf

        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        batch = tf.shape(x)[0]
        ws = self.window_size
        shift = self.shift_size

        shifted = tf.roll(x, shift=[-shift, -shift], axis=[1, 2]) if shift else x
        num_h = h // ws
        num_w = w // ws

        windows = tf.reshape(shifted, [batch, num_h, ws, num_w, ws, c])
        windows = tf.transpose(windows, [0, 1, 3, 2, 4, 5])
        windows = tf.reshape(windows, [-1, ws * ws, c])

        attn_input = self.norm1(windows)
        windows = windows + self.attn(attn_input, attn_input)
        mlp_input = self.norm2(windows)
        windows = windows + self.mlp2(self.mlp1(mlp_input))

        windows = tf.reshape(windows, [batch, num_h, num_w, ws, ws, c])
        windows = tf.transpose(windows, [0, 1, 3, 2, 4, 5])
        shifted = tf.reshape(windows, [batch, h, w, c])

        if shift:
            shifted = tf.roll(shifted, shift=[shift, shift], axis=[1, 2])
        return shifted

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
        })
        return config


def _swin_stage(x, dim: int, num_heads: int, window_size: int):
    import tensorflow as tf

    x = SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0)(x)
    return SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2)(x)


# ---------------------------------------------------------------------------
# Existing custom architectures
# ---------------------------------------------------------------------------
def _build_kernarc(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    b1 = tf.keras.layers.Conv2D(8, 2, padding="same", activation="relu")(x)
    b2 = tf.keras.layers.Conv2D(8, 4, padding="same", activation="relu")(x)
    b3 = tf.keras.layers.Conv2D(8, 8, padding="same", activation="relu")(x)
    merged = tf.keras.layers.Concatenate()([b1, b2, b3])
    y = tf.keras.layers.BatchNormalization()(merged)
    y = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(4)(y)
    gate = tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")(y)
    resized = tf.keras.layers.Resizing(image_size // 4, image_size // 4)(x)
    attended = tf.keras.layers.Multiply()([resized, gate])
    out = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(attended)
    out = tf.keras.layers.MaxPooling2D(2)(out)
    out = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(out)
    outputs = _classifier_head(out, num_classes)
    return tf.keras.Model(inputs, outputs, name="KuantanArc")


def _build_swiftpan(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x1 = tf.keras.layers.Conv2D(1, 1, activation="linear")(x)
    cm1 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="swish")(x1)
    cb1 = tf.keras.layers.Conv2D(1, 9, padding="same", activation="swish")(x1)
    mp1 = tf.keras.layers.MaxPooling2D(2)(tf.keras.layers.Add()([tf.keras.layers.Multiply()([cm1, x1]), cb1]))
    pp2 = tf.keras.layers.MaxPooling2D(2)(x1)
    cm2 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="swish")(mp1)
    cb2 = tf.keras.layers.Conv2D(1, 9, padding="same", activation="swish")(mp1)
    mp2 = tf.keras.layers.MaxPooling2D(2)(tf.keras.layers.Add()([tf.keras.layers.Multiply()([cm2, pp2]), cb2]))
    pp3 = tf.keras.layers.MaxPooling2D(4)(x1)
    cm3 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="swish")(mp2)
    cb3 = tf.keras.layers.Conv2D(1, 9, padding="same", activation="swish")(mp2)
    mp3 = tf.keras.layers.MaxPooling2D(2)(tf.keras.layers.Add()([tf.keras.layers.Multiply()([cm3, pp3]), cb3]))
    sc = tf.keras.layers.BatchNormalization()(mp3)
    sc = tf.keras.layers.Conv2D(3, 3, padding="same", activation="swish")(sc)
    sc = tf.keras.layers.BatchNormalization()(sc)
    out = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(sc)
    out = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(out)
    outputs = _classifier_head(out, num_classes)
    return tf.keras.Model(inputs, outputs, name="SwiftPan")


def _build_sepfuse(num_classes: int, image_size: int):
    import tensorflow as tf

    def sep_bn(x, filters):
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return tf.keras.layers.MaxPooling2D()(x)

    def lat(x, filters):
        x = tf.keras.layers.SeparableConv2D(filters, 1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation("relu")(x)

    inputs, x = _input_tensor(image_size)
    c1 = sep_bn(x, 32)
    c2 = sep_bn(c1, 64)
    c3 = sep_bn(c2, 128)
    c4 = sep_bn(c3, 256)
    p4 = lat(c4, 128)
    p3 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Add()([lat(c3, 128), tf.keras.layers.UpSampling2D()(p4)]))
    p2 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Add()([lat(c2, 128), tf.keras.layers.UpSampling2D()(p3)]))
    out = tf.keras.layers.GlobalMaxPooling2D()(p2)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(128, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
    return tf.keras.Model(inputs, outputs, name="SepFuse")


def _build_dualfuse(num_classes: int, image_size: int):
    import tensorflow as tf

    def cnn_stream(inp, filters_seq):
        h = inp
        for filters, kernel in filters_seq:
            h = tf.keras.layers.Conv2D(filters, kernel, padding="same", activation="relu")(h)
            h = tf.keras.layers.MaxPooling2D(2)(h)
        return tf.keras.layers.GlobalAveragePooling2D()(h)

    inputs, x = _input_tensor(image_size)
    a = cnn_stream(x, [(8, 7), (16, 5), (32, 3)])
    x_half = tf.keras.layers.Resizing(image_size // 2, image_size // 2)(x)
    b = cnn_stream(x_half, [(8, 5), (16, 3)])
    merged = tf.keras.layers.Concatenate()([a, b])
    out = tf.keras.layers.Dense(512, activation="relu")(merged)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
    return tf.keras.Model(inputs, outputs, name="DualFuse")


def _build_syncgen(num_classes: int, image_size: int):
    import tensorflow as tf

    def enc_block(x, filters, stride):
        x = tf.keras.layers.Conv2D(filters, 5, strides=stride, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.LeakyReLU(0.2)(x)

    inputs, x = _input_tensor(image_size)
    x = enc_block(x, 16, 2)
    x = enc_block(x, 32, 2)
    x = enc_block(x, 64, 4)
    x = enc_block(x, 128, 2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="SyncGen")


def _build_nanonet(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = _depthwise_block(x, 32, 1)
    x = _depthwise_block(x, 64, 2)
    x = _depthwise_block(x, 64, 1)
    x = _depthwise_block(x, 128, 2)
    x = _depthwise_block(x, 128, 1)
    x = _depthwise_block(x, 256, 2)
    outputs = _classifier_head(x, num_classes)
    return tf.keras.Model(inputs, outputs, name="NanoNet")


# ---------------------------------------------------------------------------
# Canonical architecture family
# ---------------------------------------------------------------------------
def _build_alexnet(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = tf.keras.layers.Conv2D(64, 11, strides=4, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(192, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="AlexNetLite")


def _build_vggnet(num_classes: int, image_size: int):
    import tensorflow as tf

    def vgg_block(x, filters: int, repeats: int):
        for _ in range(repeats):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.MaxPooling2D(2)(x)

    inputs, x = _input_tensor(image_size)
    x = vgg_block(x, 32, 2)
    x = vgg_block(x, 64, 2)
    x = vgg_block(x, 128, 3)
    x = vgg_block(x, 256, 3)
    x = vgg_block(x, 256, 3)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="VGGNetLite")


def _build_resnet(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    for filters, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
        x = _residual_block(x, filters, stride)
    outputs = _classifier_head(x, num_classes, dense_units=256, dropout=0.3)
    return tf.keras.Model(inputs, outputs, name="ResNetLite")


def _build_inception_v3(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = _conv_bn_relu(x, 32, 3, stride=2)
    x = _conv_bn_relu(x, 32, 3)
    x = _conv_bn_relu(x, 64, 3)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = _conv_bn_relu(x, 80, 1)
    x = _conv_bn_relu(x, 128, 3)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = _inception_module(x, 64, (48, 64), (16, 32), 32)
    x = _inception_module(x, 96, (64, 96), (24, 48), 48)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = _inception_module(x, 128, (96, 128), (32, 64), 64)
    x = _inception_module(x, 160, (128, 160), (48, 64), 64)
    outputs = _classifier_head(x, num_classes, dense_units=320, dropout=0.4)
    return tf.keras.Model(inputs, outputs, name="InceptionV3Lite")


def _build_efficientnet(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = _conv_bn_relu(x, 32, 3, stride=2)
    x = _mbconv_block(x, 16, expand_ratio=1, stride=1)
    x = _mbconv_block(x, 24, expand_ratio=4, stride=2)
    x = _mbconv_block(x, 24, expand_ratio=4, stride=1)
    x = _mbconv_block(x, 40, expand_ratio=4, stride=2)
    x = _mbconv_block(x, 40, expand_ratio=4, stride=1)
    x = _mbconv_block(x, 80, expand_ratio=6, stride=2)
    x = _mbconv_block(x, 80, expand_ratio=6, stride=1)
    x = _mbconv_block(x, 112, expand_ratio=6, stride=1)
    x = _conv_bn_relu(x, 192, 1)
    outputs = _classifier_head(x, num_classes, dense_units=320, dropout=0.3)
    return tf.keras.Model(inputs, outputs, name="EfficientNetLite")


def _build_vit(num_classes: int, image_size: int):
    import tensorflow as tf

    patch_size = 16
    embed_dim = 192
    num_heads = 3
    depth = 6

    inputs, x = _input_tensor(image_size)
    x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding="valid")(x)
    num_patches = x.shape[1] * x.shape[2]
    x = tf.keras.layers.Reshape((num_patches, embed_dim))(x)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    x = x + pos_embed

    for _ in range(depth):
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1)(y, y)
        x = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.Dense(embed_dim * 4, activation="gelu")(y)
        y = tf.keras.layers.Dropout(0.1)(y)
        y = tf.keras.layers.Dense(embed_dim)(y)
        x = tf.keras.layers.Add()([x, y])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ViTLite")


def _build_densenet(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = _conv_bn_relu(x, 32, 7, stride=2)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = _dense_block(x, growth_rate=16, repeats=4)
    x = _transition_block(x, compression=0.5)
    x = _dense_block(x, growth_rate=20, repeats=4)
    x = _transition_block(x, compression=0.5)
    x = _dense_block(x, growth_rate=24, repeats=4)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    outputs = _classifier_head(x, num_classes, dense_units=256, dropout=0.3)
    return tf.keras.Model(inputs, outputs, name="DenseNetLite")


def _build_mobilenet(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    for filters, stride in [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
        x = _depthwise_block(x, filters, stride)
    outputs = _classifier_head(x, num_classes, dense_units=256, dropout=0.2)
    return tf.keras.Model(inputs, outputs, name="MobileNetLite")


def _build_swin(num_classes: int, image_size: int):
    import tensorflow as tf

    aligned_size = ((image_size + 31) // 32) * 32
    inputs, x = _input_tensor(image_size)
    if aligned_size != image_size:
        x = tf.keras.layers.Resizing(aligned_size, aligned_size)(x)

    x = tf.keras.layers.Conv2D(64, 4, strides=4, padding="same")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = _swin_stage(x, dim=64, num_heads=2, window_size=4)
    x = tf.keras.layers.Conv2D(128, 2, strides=2, padding="same")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = _swin_stage(x, dim=128, num_heads=4, window_size=4)
    x = tf.keras.layers.Conv2D(192, 2, strides=2, padding="same")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = _swin_stage(x, dim=192, num_heads=6, window_size=4)
    outputs = _classifier_head(x, num_classes, dense_units=256, dropout=0.3)
    return tf.keras.Model(inputs, outputs, name="SwinLite")


def _build_convnext(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs, x = _input_tensor(image_size)
    x = tf.keras.layers.Conv2D(64, 4, strides=4, padding="same")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    for _ in range(2):
        x = _convnext_block(x, 64)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Conv2D(128, 2, strides=2, padding="same")(x)
    for _ in range(2):
        x = _convnext_block(x, 128)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Conv2D(256, 2, strides=2, padding="same")(x)
    for _ in range(2):
        x = _convnext_block(x, 256)
    outputs = _classifier_head(x, num_classes, dense_units=320, dropout=0.3)
    return tf.keras.Model(inputs, outputs, name="ConvNeXtLite")
