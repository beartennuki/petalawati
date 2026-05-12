from prefect import task


@task(name="build-model")
def build_model(architecture: str, num_classes: int, image_size: int):
    import tensorflow as tf
    builders = {
        "kernarc":  _build_kernarc,
        "swiftpan": _build_swiftpan,
        "sepfuse":  _build_sepfuse,
        "dualfuse": _build_dualfuse,
        "syncgen":  _build_syncgen,
    }
    builder = builders.get(architecture)
    if builder is None:
        raise ValueError(f"Unknown architecture: {architecture}")
    return builder(num_classes, image_size)


# ---------------------------------------------------------------------------
# KernArc — KtnArc Invariant
# Multi-kernel (2×, 4×, 8×) parallel branches → concat → BN → MaxPool →
# sigmoid attention gate → Multiply with resized input → classifier head.
# Faithful to KtnArc Invariant (Section 2.5) with mf=2 so kernels are
# mf/2=1 (clamped to 2), mf=2→4, mf×2=4→8.
# ---------------------------------------------------------------------------
def _build_kernarc(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    # Three parallel branches at different receptive fields
    b1 = tf.keras.layers.Conv2D(8,  2, padding="same", activation="relu")(x)
    b2 = tf.keras.layers.Conv2D(8,  4, padding="same", activation="relu")(x)
    b3 = tf.keras.layers.Conv2D(8,  8, padding="same", activation="relu")(x)

    # Concatenate and compress
    merged = tf.keras.layers.Concatenate()([b1, b2, b3])          # (H, W, 24)
    y = tf.keras.layers.BatchNormalization()(merged)
    y = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(4)(y)                         # (H/4, W/4, 16)

    # Attention gate — sigmoid selects attended channels
    gate = tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")(y)  # (H/4, W/4, 3)

    # Resize original to match gate spatial size
    s = image_size // 4
    resized = tf.keras.layers.Resizing(s, s)(x)                    # (H/4, W/4, 3)

    # Attended feature map
    attended = tf.keras.layers.Multiply()([resized, gate])         # (H/4, W/4, 3)

    # Classifier head
    out = tf.keras.layers.Conv2D(64,  3, padding="same", activation="relu")(attended)
    out = tf.keras.layers.MaxPooling2D(2)(out)
    out = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(out)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.3)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs, outputs, name="KernArc")


# ---------------------------------------------------------------------------
# SwiftPan — Rapid Min Pancake
# 3-stage attention chain: Conv_small × input  +  Conv_large → MaxPool,
# each stage pools from the original for on-link reference (Section 2.4).
# Only ~300 preprocessing params — very lightweight.
# ---------------------------------------------------------------------------
def _build_swiftpan(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    # Project to 1 channel for attention masking (like the original grayscale input)
    x1 = tf.keras.layers.Conv2D(1, 1, activation="linear")(x)     # (H, W, 1)

    # Stage 1
    cm1 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="swish")(x1)
    cb1 = tf.keras.layers.Conv2D(1, 9, padding="same", activation="swish")(x1)
    ml1 = tf.keras.layers.Multiply()([cm1, x1])                    # pointwise attention
    ad1 = tf.keras.layers.Add()([ml1, cb1])                        # add context
    mp1 = tf.keras.layers.MaxPooling2D(2)(ad1)                     # H/2

    # Stage 2 — original ref pooled in parallel
    pp2 = tf.keras.layers.MaxPooling2D(2)(x1)                      # H/2
    cm2 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="swish")(mp1)
    cb2 = tf.keras.layers.Conv2D(1, 9, padding="same", activation="swish")(mp1)
    ml2 = tf.keras.layers.Multiply()([cm2, pp2])
    ad2 = tf.keras.layers.Add()([ml2, cb2])
    mp2 = tf.keras.layers.MaxPooling2D(2)(ad2)                     # H/4

    # Stage 3
    pp3 = tf.keras.layers.MaxPooling2D(4)(x1)                      # H/4
    cm3 = tf.keras.layers.Conv2D(1, 3, padding="same", activation="swish")(mp2)
    cb3 = tf.keras.layers.Conv2D(1, 9, padding="same", activation="swish")(mp2)
    ml3 = tf.keras.layers.Multiply()([cm3, pp3])
    ad3 = tf.keras.layers.Add()([ml3, cb3])
    mp3 = tf.keras.layers.MaxPooling2D(2)(ad3)                     # H/8

    # Normalise via BN + color projection (1→3 channels)
    sc = tf.keras.layers.BatchNormalization()(mp3)
    sc = tf.keras.layers.Conv2D(3, 3, padding="same", activation="swish")(sc)
    sc = tf.keras.layers.BatchNormalization()(sc)                   # (H/8, W/8, 3)

    # Classifier head
    out = tf.keras.layers.Conv2D(64,  3, padding="same", activation="relu")(sc)
    out = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(out)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.3)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs, outputs, name="SwiftPan")


# ---------------------------------------------------------------------------
# SepFuse — Separable-FPN (Section 1.4)
# Bottom-up: SeparableConv2D + BN + relu + MaxPool → c1…c4
# Top-down:  1×1 SepConv + BN → UpSample + Add → p1…p3
# Head: GlobalMaxPooling2D → Dense → Dropout → softmax
# ---------------------------------------------------------------------------
def _build_sepfuse(num_classes: int, image_size: int):
    import tensorflow as tf

    def sep_bn(x, filters, pool=True):
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if pool:
            x = tf.keras.layers.MaxPooling2D()(x)
        return x

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    # Bottom-up (4 levels with MaxPool → H/2, H/4, H/8, H/16)
    c1 = sep_bn(x,   32,  pool=True)   # H/2
    c2 = sep_bn(c1,  64,  pool=True)   # H/4
    c3 = sep_bn(c2,  128, pool=True)   # H/8
    c4 = sep_bn(c3,  256, pool=True)   # H/16 (deepest)

    def lat(x, filters):
        x = tf.keras.layers.SeparableConv2D(filters, 1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation("relu")(x)

    # Top-down pyramid — each UpSampling2D doubles spatial size so shapes align
    # p4: H/16 → upsample → H/8, merged with c3 (H/8) → p3
    # p3: H/8  → upsample → H/4, merged with c2 (H/4) → p2
    p4 = lat(c4, 128)                                               # H/16
    p3 = tf.keras.layers.Add()([lat(c3, 128), tf.keras.layers.UpSampling2D()(p4)])
    p3 = tf.keras.layers.BatchNormalization()(p3)                   # H/8
    p2 = tf.keras.layers.Add()([lat(c2, 128), tf.keras.layers.UpSampling2D()(p3)])
    p2 = tf.keras.layers.BatchNormalization()(p2)                   # H/4

    # Head — p2 carries multi-scale info from H/4 → H/16
    out = tf.keras.layers.GlobalMaxPooling2D()(p2)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(128, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs, outputs, name="SepFuse")


# ---------------------------------------------------------------------------
# DualFuse — Dual-Input CNN + Vector Regressor (Section 3.2)
# Adapted for image-only classification: two parallel CNN streams at
# full-resolution and half-resolution are concatenated before the head,
# preserving the dual-stream fusion spirit of the original architecture.
# ---------------------------------------------------------------------------
def _build_dualfuse(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    def cnn_stream(inp, filters_seq):
        h = inp
        for filters, k in filters_seq:
            h = tf.keras.layers.Conv2D(filters, k, padding="same", activation="relu")(h)
            h = tf.keras.layers.MaxPooling2D(2)(h)
        return tf.keras.layers.GlobalAveragePooling2D()(h)

    # Stream A — full resolution (7×7, 5×5, 3×3 kernels — from original)
    a = cnn_stream(x, [(8, 7), (16, 5), (32, 3)])

    # Stream B — half resolution (shifted perspective)
    half = image_size // 2
    x_half = tf.keras.layers.Resizing(half, half)(x)
    b = cnn_stream(x_half, [(8, 5), (16, 3)])

    # Fusion
    merged = tf.keras.layers.Concatenate()([a, b])
    out = tf.keras.layers.Dense(512, activation="relu")(merged)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs, outputs, name="DualFuse")


# ---------------------------------------------------------------------------
# SyncGen — Supervised Generative Model (Section 3.4)
# The original is a decoder (noise→image). Inverted here as an encoder:
# strided Conv2D + BN + LeakyReLU blocks progressively compress the image
# to a bottleneck, then a Dense head classifies. Architecture mirrors the
# transposed-conv sizes/strides of the decoder in reverse.
# ---------------------------------------------------------------------------
def _build_syncgen(num_classes: int, image_size: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    def enc_block(x, filters, stride=2):
        x = tf.keras.layers.Conv2D(filters, 5, strides=stride, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x

    # Progressive strided encoder (reverse of decoder depth order)
    x = enc_block(x,  16, stride=2)   # H/2
    x = enc_block(x,  32, stride=2)   # H/4
    x = enc_block(x,  64, stride=4)   # H/16
    x = enc_block(x, 128, stride=2)   # H/32

    # Bottleneck head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="SyncGen")
