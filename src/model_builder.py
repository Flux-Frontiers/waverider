"Model Builder for the canonical ML tests with Waverider"

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402

# ---------------------------------------------------------------------------
# Internal helper — shared by build_resnet and build_manifold_resnet
# ---------------------------------------------------------------------------


def _residual_block(x, filters):
    """One residual block: Conv→BN→ReLU→Conv→BN→add skip→ReLU.

    If the skip connection has a different number of channels, a 1×1 Conv
    is inserted to match shapes before the addition.

    :param x: Input tensor.
    :param filters: Number of Conv2D filters.
    :returns: Output tensor after residual connection.
    """
    skip = x
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    if skip.shape[-1] != filters:
        skip = keras.layers.Conv2D(filters, (1, 1), padding="same")(skip)
    x = keras.layers.Add()([x, skip])
    x = keras.layers.ReLU()(x)
    return x


def build_standard_model(input_dim, n_classes, lr=0.001, optimizer=None):
    """Canonical MLP baseline: input → 1024 → 512 → n_classes.

    Standard over-parameterised baseline for pixel-space inputs.
    Expected accuracy: ~45–50% on CIFAR-10 with full training data,
    no augmentation.

    :param optimizer: Optional optimizer instance. Defaults to Adam(lr).
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=(optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate=lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001, optimizer=None):
    """Manifold-informed architecture: input → d → output.

    The bottleneck width matches the discovered intrinsic dimensionality.

    :param optimizer: Optional optimizer instance. Defaults to Adam(lr).
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=(optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate=lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_wide_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Wider manifold-informed: input → d+1 → output.

    One hidden layer of width d+1 — one neuron wider than the pure manifold
    model to give marginal additional capacity without abandoning the
    manifold bottleneck constraint.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(d + 1, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → 2d → d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_mlp_wide(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected wide MLP: d → 4d → 2d → output.

    Wider first layer than the standard PCA+MLP — more room to learn the
    nonlinear projection before compressing toward the bottleneck.

    Architecture::

        PCA(d*) → Dense(4d, relu) → Dense(2d, relu) → Dense(C, softmax)

    :param n_classes: Number of output classes C.
    :param intrinsic_dim: PCA output dimensionality d*.
    :param lr: Learning rate.
    :returns: Compiled Keras model (input shape = d*).
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(4 * d, activation="relu"),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_mlp_deep(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected deep MLP: d → 2d → 2d → d → output.

    Three hidden layers at the same width budget as the standard PCA+MLP
    but spread deeper.  Tests whether depth helps in the PCA-projected space.

    Architecture::

        PCA(d*) → Dense(2d, relu) → Dense(2d, relu) → Dense(d, relu) → Dense(C, softmax)

    :param n_classes: Number of output classes C.
    :param intrinsic_dim: PCA output dimensionality d*.
    :param lr: Learning rate.
    :returns: Compiled Keras model (input shape = d*).
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_ub_deep(n_classes, d_star, lr=0.001):
    """PCA→UB-deep: PCA(d*) → w* → w* → C.

    Two hidden layers of UB-theorem width w* = d* + C - 1 after PCA
    projection.  Tests whether double nonlinear refinement in the
    UB-width space pushes past the single-layer UB-PCA ceiling.

    Architecture::

        PCA(d*) → Dense(w*, relu) → Dense(w*, relu) → Dense(C, softmax)

    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (PCA input width).
    :param lr: Learning rate.
    :returns: Compiled Keras model (input shape = d*).
    """
    w = d_star + n_classes - 1
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(d_star,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_linear_model(n_classes, intrinsic_dim, lr=0.001):
    """Pure linear classifier on PCA-projected input: d → output (no hidden layer).

    :param n_classes: Number of output classes.
    :param intrinsic_dim: PCA output dimensionality.
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_nc_model(n_classes, intrinsic_dim, hidden_width=None, lr=0.001):
    """PCA pre-projected model: d* → H → C → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    A single hidden layer of width H (defaults to n_classes when not specified)
    precedes the softmax output.

    Architecture::

        PCA(d*) → Dense(H, relu) → Dense(C, softmax)

    Example configurations:
        CIFAR-100 same-width (d*=19, C=100, H=100):
            Dense(19→100): 2,000  |  Dense(100→100): 10,100  |  Total = 12,100
        CIFAR-10 with CIFAR-100 hidden width (d*=19, C=10, H=100):
            Dense(19→100): 2,000  |  Dense(100→10):   1,010  |  Total =  3,010

    :param n_classes: Number of output classes.
    :param intrinsic_dim: PCA output dimensionality d*.
    :param hidden_width: Hidden layer width H.  Defaults to n_classes when None.
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    h = hidden_width if hidden_width is not None else n_classes
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(h, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_intrinsic_dim_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    The network only needs to learn the nonlinear classification
    in the manifold subspace, not the projection itself.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(intrinsic_dim, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_resnet(
    input_dim,
    n_classes,
    intrinsic_dim,
    lr=0.001,
    optimizer=None,
    spatial_shape=(32, 32, 3),
    dropout=0.0,
):
    """ManifoldResNet-d: ResNet with filter count equal to the intrinsic dimension d*.

    Replaces the arbitrary 32-filter choice in a standard small ResNet with the
    manifold-derived intrinsic dimension d*, motivated by the hypothesis that d*
    independent feature maps suffice to represent all discriminative signal present
    in the data.  After GlobalAveragePooling the network outputs a d*-dimensional
    feature vector — the same dimensionality as the data manifold — before the
    final linear classifier.

    Architecture::

        Input(input_dim) → Reshape(spatial_shape)
        → ResBlock(d*)  → MaxPool(2×2)
        → ResBlock(d*)  → MaxPool(2×2)
        → ResBlock(d*)  → MaxPool(2×2)
        → GlobalAveragePool          ← d*-dimensional feature vector
        → Dense(n_classes, softmax)  ← d* × C + C parameters

    Estimated parameters for CIFAR-10 (d*=18, C=10):
        ResBlock1 (3→18):   3×3×3×18 + 3×3×18×18 + 2×BN(18) + skip(1×1×3×18) ≈  3,636
        ResBlock2 (18→18):  3×3×18×18 × 2          + 2×BN(18)                 ≈  5,976
        ResBlock3 (18→18):  same                                                ≈  5,976
        Dense (18→10):      18×10 + 10                                         =    190
        Total ≈ 15,778  (vs ResNet-32 at 47,978 — 3× reduction)

    :param input_dim: Flat input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes.
    :param intrinsic_dim: Manifold intrinsic dimension d* — used as the Conv2D
        filter count at every residual block.
    :param lr: Learning rate (used when optimizer is None).
    :param optimizer: Optional pre-configured optimizer.  Defaults to Adam(lr).
    :param spatial_shape: Tuple (H, W, C) to reshape the flat input into before
        the convolutional blocks.  Must satisfy H*W*C == input_dim.
    :returns: Compiled Keras model.
    """
    d = intrinsic_dim

    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Reshape(spatial_shape)(inp)

    x = _residual_block(x, d)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, d)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, d)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    if dropout > 0.0:
        x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=(optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate=lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_resnet_2d(
    input_dim,
    n_classes,
    intrinsic_dim,
    lr=0.001,
    optimizer=None,
    spatial_shape=(32, 32, 3),
):
    """ManifoldResNet-2d: ResNet with filter count equal to 2 × intrinsic dimension.

    Motivated by the Whitney embedding theorem: a smooth d*-dimensional manifold
    can always be smoothly embedded in R^(2d*).  Using 2d* filters provides the
    minimal ambient dimension guaranteed to faithfully represent the manifold
    geometry, while remaining a principled, data-derived hyperparameter.

    For CIFAR-10 (d*=18): 2×18 = 36 filters — slightly more than the conventional
    ResNet-32 baseline (32 filters) — with a theoretical justification replacing
    an arbitrary choice.

    Architecture::

        Input(input_dim) → Reshape(spatial_shape)
        → ResBlock(2d*)  → MaxPool(2×2)
        → ResBlock(2d*)  → MaxPool(2×2)
        → ResBlock(2d*)  → MaxPool(2×2)
        → GlobalAveragePool          ← 2d*-dimensional feature vector
        → Dense(n_classes, softmax)

    :param input_dim: Flat input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes.
    :param intrinsic_dim: Manifold intrinsic dimension d* — filter count is 2×d*.
    :param lr: Learning rate (used when optimizer is None).
    :param optimizer: Optional pre-configured optimizer.  Defaults to Adam(lr).
    :param spatial_shape: Tuple (H, W, C) to reshape the flat input.
    :returns: Compiled Keras model.
    """
    return build_manifold_resnet(
        input_dim,
        n_classes,
        2 * intrinsic_dim,
        lr=lr,
        optimizer=optimizer,
        spatial_shape=spatial_shape,
    )


def build_manifold_resnet_nc(
    input_dim,
    n_classes,
    intrinsic_dim,
    lr=0.001,
    optimizer=None,
    spatial_shape=(32, 32, 3),
):
    """ManifoldResNet-d+C: ManifoldResNet-d with a n_classes-wide hidden layer before softmax.

    Extends ManifoldResNet-d by inserting a Dense(n_classes, relu) layer between
    GlobalAveragePooling and the final classifier.  This gives the network a
    representation space wide enough to hold all class boundaries before the
    decision layer, without abandoning the manifold-guided filter count.

    Architecture::

        Input(input_dim) → Reshape(spatial_shape)
        → ResBlock(d*)  → MaxPool(2×2)
        → ResBlock(d*)  → MaxPool(2×2)
        → ResBlock(d*)  → MaxPool(2×2)
        → GlobalAveragePool              ← d*-dimensional feature vector
        → Dense(n_classes, relu)         ← class-capacity hidden layer  [NEW]
        → Dense(n_classes, softmax)

    Estimated parameters for CIFAR-100 (d*=19, C=100):
        Conv blocks:             ≈ 17,200
        Dense(19→100, relu):     19×100 + 100  =  2,000
        Dense(100→100, softmax): 100×100 + 100 = 10,100
        Total ≈ 29,300  (vs ResNet-32 at ~51K — 1.75× reduction)

    :param input_dim: Flat input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes; also sets the hidden layer width.
    :param intrinsic_dim: Manifold intrinsic dimension d* — Conv2D filter count.
    :param lr: Learning rate (used when optimizer is None).
    :param optimizer: Optional pre-configured optimizer.  Defaults to Adam(lr).
    :param spatial_shape: Tuple (H, W, C) for reshaping the flat input.
    :returns: Compiled Keras model.
    """
    d = intrinsic_dim

    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Reshape(spatial_shape)(inp)

    x = _residual_block(x, d)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, d)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, d)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(n_classes, activation="relu")(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=(optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate=lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_universal_bottleneck_pca(n_classes, d_star, lr=0.001):
    """Universal Bottleneck (PCA-seeded): PCA(d*) → (d*+C-1) → C.

    Tests the Universal Bottleneck Theorem with PCA pre-projection.
    Input is already projected to d* dimensions. The single hidden layer
    of width w* = d* + C - 1 must learn the C-1 class-separation coordinates
    on top of the d* geometry already provided by PCA.

    Architecture::

        PCA(d*) → Dense(d*+C-1, relu) → Dense(C, softmax)

    Example (CIFAR-10, d*=19, C=10): w* = 28, params = 532 + 290 = 822

    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (PCA input width).
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    w = d_star + n_classes - 1
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(d_star,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_class_augmented_pca(n_classes, d_star, lr=0.001):
    """Class-Augmented PCA classifier: PCA(d*+C) → C.

    Projects input into a (d*+C)-dimensional space via PCA (applied externally
    before this model receives input), then classifies directly with a single
    linear layer.  No hidden bottleneck.

    The projection dimension d*+C is chosen so the space has room for both
    the manifold geometry (d* coordinates) and full class separation (C
    coordinates — one per class, unlike the UB theorem's C-1).  This removes
    the Shannon bottleneck entirely: classes are not squeezed; they are
    explicitly allocated a dedicated coordinate.

    Architecture::

        PCA(d*+C) → Dense(C, softmax)

    Example (CIFAR-10, d*=29, C=10): projection dim = 39, params = 39×10+10 = 400
    Example (CIFAR-100, d*=27, C=100): projection dim = 127, params = 127×100+100 = 12,800

    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic manifold dimension d*.
    :param lr: Learning rate.
    :returns: Compiled Keras model (input shape = d*+C).
    """
    proj_dim = d_star + n_classes
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(proj_dim,)),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lda_pca_augmented(n_classes, d_star, lr=0.001):
    """LDA+PCA class-augmented classifier: concat(PCA(d*), LDA(C-1)) → C.

    The input to this model is a concatenation of two projections computed
    externally before training:
      - PCA(d*) : d* dimensions capturing manifold geometry (variance-maximising)
      - LDA(C-1): C-1 dimensions capturing class separation (discriminant-maximising)

    Total input width: d* + (C-1).  A single linear softmax layer then
    classifies.  No hidden bottleneck.

    This is the supervised counterpart to build_class_augmented_pca: where
    that model gave PCA the extra C dimensions (unsupervised, so they just
    captured the next variance directions), this model replaces those C
    dimensions with LDA discriminants — guaranteed to be maximally
    class-separating by construction.

    Architecture::

        concat[PCA(d*), LDA(C-1)] → Dense(C, softmax)

    Example (CIFAR-10, d*=29, C=10): input = 38, params = 38×10+10 = 390
    Example (CIFAR-100, d*=27, C=100): input = 126, params = 126×100+100 = 12,700

    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic manifold dimension d*.
    :param lr: Learning rate.
    :returns: Compiled Keras model (input shape = d* + C - 1).
    """
    proj_dim = d_star + n_classes - 1
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(proj_dim,)),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ub_pca_mlp(input_dim, n_classes, d_star, lr=0.001):
    """Universal Bottleneck with linear (PCA-equivalent) middle layer.

    A linear Dense layer acts as a learnable PCA projection between the
    nonlinear input stage and the classifier.

    Architecture::

        Input(input_dim) → Dense(w, relu) → Dense(w, linear) → Dense(C, softmax)

    where w = d* + C (= 119 for CIFAR-100 with d*=19, C=100).

    The middle Dense(w, linear) layer has no activation — it is a pure learned
    linear projection, equivalent in role to a PCA rotation of the w-dim space.

    :param input_dim: Ambient input dimensionality.
    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (w = d* + C).
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    w = d_star + n_classes
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(w, activation=None),  # linear — learnable PCA
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_universal_bottleneck_mlp(input_dim, n_classes, d_star, lr=0.001):
    """Universal Bottleneck MLP: input → w* → w* → C.

    Two hidden layers of width w* = d* + C - 1 from raw input.
    No PCA pre-projection, no convolutional layers, no compression.
    The network must discover geometry and class separation simultaneously
    across two w*-wide layers.

    Architecture::

        Input(input_dim) → Dense(w*, relu) → Dense(w*, relu) → Dense(C, softmax)

    Example (CIFAR-100, input=3072, d*=19, C=100): w* = 118, params ≈ 375,754

    :param input_dim: Ambient input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (determines w* = d* + C - 1).
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    w = d_star + n_classes - 1
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_universal_bottleneck_raw(input_dim, n_classes, d_star, lr=0.001):
    """Universal Bottleneck (raw input): input → (d*+C-1) → C.

    Tests the Universal Bottleneck Theorem from raw ambient input without
    PCA pre-projection. The network must discover both the manifold geometry
    (d* dimensions) and class-separation coordinates (C-1 dimensions)
    simultaneously within w* = d* + C - 1 neurons.

    The dimension probe confirmed that a network given d* total neurons
    spontaneously allocates ~d* for geometry and C-1 for class separation.
    This architecture tests whether the same self-organisation occurs when
    the bottleneck is set to exactly w* = d* + C - 1 from raw input.

    Architecture::

        Input(input_dim) → Dense(d*+C-1, relu) → Dense(C, softmax)

    Example (CIFAR-10, input=3072, d*=19, C=10): w* = 28, params ≈ 86,346

    :param input_dim: Ambient input dimensionality (e.g. 3072 for CIFAR-10).
    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (determines bottleneck width w*).
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    w = d_star + n_classes - 1
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_class_augmented_mlp(input_dim, n_classes, d_star, lr=0.001):
    """ManifoldResNet-CA MLP: input → (d*+C) → C.

    Drops the Shannon bottleneck entirely.  The Universal Bottleneck width
    w* = d* + C - 1 squeezes C-class separation into one fewer dimension than
    the number of classes, which is geometrically insufficient when C >> d* —
    you cannot fully separate 100 classes in 99 coordinates while simultaneously
    encoding the manifold geometry.

    This architecture instead uses w = d* + C (no minus-1): d* neurons for
    manifold geometry and a full C neurons for class separation.  No compression,
    no information bottleneck.  Pure class augmentation on top of the manifold
    geometry.

    Observed for CIFAR-10 (d*=29, C=10) and CIFAR-100 (d*=27, C=100): both
    datasets share ~99.1% noise, confirming the ambient space is geometrically
    insufficient — the extra C dimension is the minimal correction.

    Architecture::

        Input(input_dim) → Dense(d*+C, relu) → Dense(C, softmax)

    Example (CIFAR-100, input=3072, d*=27, C=100): w = 127, params ≈ 403,327

    :param input_dim: Ambient input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (manifold geometry dimensions).
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    w = d_star + n_classes
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_c_expand_c(input_dim, n_classes, d_star, lr=0.001):
    """Sigmoid-gated class augmentation: input → (d*+C, sigmoid) → C.

    A single hidden layer of width d*+C with sigmoid activation, followed
    by a softmax classifier.  The sigmoid gate produces soft [0,1] activations
    that act as per-neuron switches — d* neurons for manifold geometry and C
    neurons for class separation — without the unbounded outputs of ReLU.

    Architecture::

        Input(input_dim) → Dense(d*+C, sigmoid) → Dense(C, softmax)

    Example (CIFAR-10, input=3072, d*=29, C=10): w = 39, params = 3072×39+39 + 39×10+10 = 120,247
    Example (CIFAR-100, input=3072, d*=27, C=100): w = 127, params = 3072×127+127 + 127×100+100 = 403,271

    :param input_dim: Ambient input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* (determines hidden width d*+C).
    :param lr: Learning rate.
    :returns: Compiled Keras model.
    """
    w = d_star + n_classes
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(w, activation="sigmoid"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_class_augmented_resnet(
    input_dim, n_classes, d_star, lr=0.001, optimizer=None, spatial_shape=(32, 32, 3)
):
    """ManifoldResNet-CA: ResNet with (d*+C) filters per residual block.

    Drops the Shannon bottleneck entirely.  Where ManifoldResNet-d uses d*
    filters (manifold geometry only) and ManifoldResNet-UB uses d*+C-1 filters
    (Universal Bottleneck), this architecture uses d*+C filters: one per
    manifold dimension plus one per class.  No minus-1, no compression.

    Motivated by the observation that CIFAR-10 (d*=29, C=10) and CIFAR-100
    (d*=27, C=100) share ~99.1% noise in pixel space: the embedding geometry
    is insufficient to fully separate C classes using only C-1 coordinates.
    Giving the convolutional feature maps a full C-dimensional class subspace
    on top of d* geometry dimensions removes that constraint.

    Architecture::

        Input(input_dim) → Reshape(spatial_shape)
        → ResBlock(d*+C)  → MaxPool(2×2)
        → ResBlock(d*+C)  → MaxPool(2×2)
        → ResBlock(d*+C)  → MaxPool(2×2)
        → GlobalAveragePool          ← (d*+C)-dimensional feature vector
        → Dense(n_classes, softmax)

    Example (CIFAR-100, d*=27, C=100): filters = 127
        ResBlock1 (3→127):   large, but geometrically motivated
        Dense (127→100):     127×100 + 100 = 12,800 parameters
        Total driven by conv blocks

    :param input_dim: Flat input dimensionality (e.g. 3072 for CIFAR-10/100).
    :param n_classes: Number of output classes C.
    :param d_star: Intrinsic dimensionality d* — filter count is d*+C.
    :param lr: Learning rate (used when optimizer is None).
    :param optimizer: Optional pre-configured optimizer.  Defaults to Adam(lr).
    :param spatial_shape: Tuple (H, W, C) to reshape the flat input.  Must
        satisfy H*W*C == input_dim.
    :returns: Compiled Keras model.
    """
    filters = d_star + n_classes

    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Reshape(spatial_shape)(inp)

    x = _residual_block(x, filters)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, filters)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, filters)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=(optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate=lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
