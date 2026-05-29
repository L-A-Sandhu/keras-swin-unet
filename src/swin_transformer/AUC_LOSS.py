import tensorflow as tf


def _smooth_labels(y_true, label_smoothing):
    if label_smoothing == 0:
        return y_true
    num_classes = tf.shape(y_true)[-1]
    return y_true * (1.0 - label_smoothing) + label_smoothing / tf.cast(
        num_classes, y_true.dtype
    )


# ---------------------------------------------------------------------------
# Keras Loss subclasses — serializable via get_config / from_config
# ---------------------------------------------------------------------------


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = _smooth_labels(y_true, self.label_smoothing)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        fl = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items()
                      if k in ("alpha", "gamma", "label_smoothing", "name")})


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0, name="dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred, [-1])
        inter = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        return 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"smooth": self.smooth})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items()
                      if k in ("smooth", "name")})


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class BCELoss(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.0, name="bce_loss"):
        super().__init__(name=name)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = _smooth_labels(y_true, self.label_smoothing)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"label_smoothing": self.label_smoothing})
        return cfg

    @classmethod
    def from_config(cls, config):
        # Strip parent-class keys (reduction, fn) that our __init__ doesn't accept
        _skip = {"reduction", "fn"}
        return cls(**{k: v for k, v in config.items() if k not in _skip})


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, name="tversky_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "beta": self.beta, "smooth": self.smooth})
        return cfg

    @classmethod
    def from_config(cls, config):
        # Strip parent-class keys (reduction, fn) that our __init__ doesn't accept
        _skip = {"reduction", "fn"}
        return cls(**{k: v for k, v in config.items() if k not in _skip})


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class BCEDiceLoss(tf.keras.losses.Loss):
    def __init__(self, bce_weight=0.5, dice_smooth=1.0, label_smoothing=0.0, name="bce_dice_loss"):
        super().__init__(name=name)
        self.bce_weight = bce_weight
        self.dice_smooth = dice_smooth
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred_c = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        yt = _smooth_labels(y_true, self.label_smoothing)
        bce = -tf.reduce_mean(yt * tf.math.log(y_pred_c) + (1.0 - yt) * tf.math.log(1.0 - y_pred_c))

        yt_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        yp_f = tf.reshape(y_pred, [-1])
        inter = tf.reduce_sum(yt_f * yp_f)
        union = tf.reduce_sum(yt_f) + tf.reduce_sum(yp_f)
        dice = 1.0 - (2.0 * inter + self.dice_smooth) / (union + self.dice_smooth)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"bce_weight": self.bce_weight, "dice_smooth": self.dice_smooth, "label_smoothing": self.label_smoothing})
        return cfg

    @classmethod
    def from_config(cls, config):
        # Strip parent-class keys (reduction, fn) that our __init__ doesn't accept
        _skip = {"reduction", "fn"}
        return cls(**{k: v for k, v in config.items() if k not in _skip})


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class FocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.5, dice_smooth=1.0, label_smoothing=0.0, name="focal_dice_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_smooth = dice_smooth
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred_c = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        yt = _smooth_labels(y_true, self.label_smoothing)
        alpha_t = yt * self.alpha + (1.0 - yt) * (1.0 - self.alpha)
        p_t = yt * y_pred_c + (1.0 - yt) * (1.0 - y_pred_c)
        fl = tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t))

        yt_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        yp_f = tf.reshape(y_pred, [-1])
        inter = tf.reduce_sum(yt_f * yp_f)
        union = tf.reduce_sum(yt_f) + tf.reduce_sum(yp_f)
        dice = 1.0 - (2.0 * inter + self.dice_smooth) / (union + self.dice_smooth)
        return self.focal_weight * fl + (1.0 - self.focal_weight) * dice

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma, "focal_weight": self.focal_weight, "dice_smooth": self.dice_smooth, "label_smoothing": self.label_smoothing})
        return cfg

    @classmethod
    def from_config(cls, config):
        # Strip parent-class keys (reduction, fn) that our __init__ doesn't accept
        _skip = {"reduction", "fn"}
        return cls(**{k: v for k, v in config.items() if k not in _skip})


@tf.keras.utils.register_keras_serializable(package="SwinUNet")
class FocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3, smooth=1.0, name="focal_tversky_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth

    def call(self, y_true, y_pred):
        yt_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        yp_f = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(yt_f * yp_f)
        fp = tf.reduce_sum((1.0 - yt_f) * yp_f)
        fn = tf.reduce_sum(yt_f * (1.0 - yp_f))
        tv = 1.0 - (tp + self.smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth)
        return tf.pow(tv, self.gamma)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma, "tversky_alpha": self.tversky_alpha, "tversky_beta": self.tversky_beta, "smooth": self.smooth})
        return cfg

    @classmethod
    def from_config(cls, config):
        # Strip parent-class keys (reduction, fn) that our __init__ doesn't accept
        _skip = {"reduction", "fn"}
        return cls(**{k: v for k, v in config.items() if k not in _skip})


# ---------------------------------------------------------------------------
# Closure-based builders (for programmatic use, not saved in model configs)
# ---------------------------------------------------------------------------

def focal_loss(alpha=0.25, gamma=2.0, label_smoothing=0.0):
    return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)

def dice_loss_fn(smooth=1.0):
    return DiceLoss(smooth=smooth)

def bce_loss_fn(label_smoothing=0.0):
    return BCELoss(label_smoothing=label_smoothing)

def tversky_loss_fn(alpha=0.7, beta=0.3, smooth=1.0):
    return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

def bce_dice_loss(bce_weight=0.5, dice_smooth=1.0, label_smoothing=0.0):
    return BCEDiceLoss(bce_weight=bce_weight, dice_smooth=dice_smooth, label_smoothing=label_smoothing)

def focal_dice_loss(alpha=0.25, gamma=2.0, focal_weight=0.5, dice_smooth=1.0, label_smoothing=0.0):
    return FocalDiceLoss(alpha=alpha, gamma=gamma, focal_weight=focal_weight, dice_smooth=dice_smooth, label_smoothing=label_smoothing)

def focal_tversky_loss(alpha=0.25, gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3, smooth=1.0):
    return FocalTverskyLoss(alpha=alpha, gamma=gamma, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta, smooth=smooth)

def auc_focal_loss(alpha=0.25, gamma=2.0):
    """Legacy wrapper — returns FocalLoss for backwards compatibility."""
    return FocalLoss(alpha=alpha, gamma=gamma)


# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------

_LOSS_REGISTRY = {
    "focal":          FocalLoss,
    "dice":           DiceLoss,
    "bce":            BCELoss,
    "bce_dice":       BCEDiceLoss,
    "focal_dice":     FocalDiceLoss,
    "focal_tversky":  FocalTverskyLoss,
    "tversky":        TverskyLoss,
}


def get_loss(loss_type="focal", **kwargs):
    """Return a Keras Loss instance by name.

    Supported loss types:
        "focal"          — FocalLoss (kwargs: alpha, gamma, label_smoothing)
        "dice"           — DiceLoss (kwargs: smooth)
        "bce"            — BCELoss (kwargs: label_smoothing)
        "bce_dice"       — BCEDiceLoss (kwargs: bce_weight, dice_smooth, label_smoothing)
        "focal_dice"     — FocalDiceLoss (kwargs: alpha, gamma, focal_weight, dice_smooth)
        "focal_tversky"  — FocalTverskyLoss (kwargs: alpha, gamma, tversky_alpha, tversky_beta)
        "tversky"        — TverskyLoss (kwargs: alpha, beta, smooth)
    """
    if loss_type not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Supported: {list(_LOSS_REGISTRY.keys())}"
        )

    cls = _LOSS_REGISTRY[loss_type]

    # Filter kwargs to only pass relevant ones to each loss class
    if loss_type == "focal":
        return cls(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )
    elif loss_type == "dice":
        return cls(smooth=kwargs.get("smooth", 1.0))
    elif loss_type == "bce":
        return cls(label_smoothing=kwargs.get("label_smoothing", 0.0))
    elif loss_type == "bce_dice":
        return cls(
            bce_weight=kwargs.get("bce_weight", 0.5),
            dice_smooth=kwargs.get("dice_smooth", 1.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )
    elif loss_type == "focal_dice":
        return cls(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            focal_weight=kwargs.get("focal_weight", 0.5),
            dice_smooth=kwargs.get("dice_smooth", 1.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
        )
    elif loss_type == "focal_tversky":
        return cls(
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
            tversky_alpha=kwargs.get("tversky_alpha", 0.7),
            tversky_beta=kwargs.get("tversky_beta", 0.3),
            smooth=kwargs.get("smooth", 1.0),
        )
    elif loss_type == "tversky":
        return cls(
            alpha=kwargs.get("alpha", 0.7),
            beta=kwargs.get("beta", 0.3),
            smooth=kwargs.get("smooth", 1.0),
        )
    return cls(**{k: v for k, v in kwargs.items() if k not in ("loss_type",)})
