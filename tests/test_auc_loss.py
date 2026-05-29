import pytest
import tensorflow as tf
import numpy as np
from swin_transformer.AUC_LOSS import (
    auc_focal_loss, get_loss, FocalLoss, DiceLoss, BCELoss,
    BCEDiceLoss, FocalDiceLoss, FocalTverskyLoss, TverskyLoss,
)


def test_focal_loss_is_scalar():
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(y_true, y_pred)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert not tf.math.is_nan(loss)
    assert loss.numpy() > 0


def test_focal_loss_extreme_predictions():
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(y_true, y_pred)
    assert loss.numpy() >= 0


def test_focal_loss_with_model_compile():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss=FocalLoss(alpha=0.25, gamma=2.0), metrics=["accuracy"])
    X = np.random.rand(4, 10).astype(np.float32)
    y = tf.keras.utils.to_categorical([0, 1, 1, 0], num_classes=2)
    try:
        history = model.fit(X, y, epochs=1, verbose=0)
    except Exception as e:
        if "DNN library initialization failed" in str(e):
            pytest.skip("CuDNN version mismatch in this environment")
        raise
    assert "loss" in history.history


def test_all_loss_types_return_scalar():
    y_true = tf.constant([[[1,0],[0,1]], [[1,0],[0,1]]], dtype=tf.float32)
    y_pred = tf.constant([[[0.9,0.1],[0.2,0.8]], [[0.8,0.2],[0.3,0.7]]], dtype=tf.float32)
    for name in ["focal", "dice", "bce", "bce_dice", "focal_dice", "focal_tversky", "tversky"]:
        loss_fn = get_loss(name)
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == (), f"'{name}' should be scalar, got {loss.shape}"
        assert loss.numpy() >= 0, f"'{name}' should be non-negative"


def test_serialization_roundtrip():
    """Loss classes should serialize and deserialize via get_config / from_config."""
    for cls in [FocalLoss, DiceLoss, BCELoss, BCEDiceLoss, FocalDiceLoss, FocalTverskyLoss, TverskyLoss]:
        instance = cls()
        config = instance.get_config()
        restored = cls.from_config(config)
        assert isinstance(restored, cls), f"{cls.__name__} round-trip failed"


def test_legacy_auc_focal_loss_is_focal_loss():
    y_true = tf.constant([[[1,0],[0,1]]], dtype=tf.float32)
    y_pred = tf.constant([[[0.9,0.1],[0.2,0.8]]], dtype=tf.float32)
    legacy = auc_focal_loss(alpha=0.25, gamma=2.0)(y_true, y_pred)
    standard = FocalLoss(alpha=0.25, gamma=2.0)(y_true, y_pred)
    assert abs(legacy.numpy() - standard.numpy()) < 1e-6
