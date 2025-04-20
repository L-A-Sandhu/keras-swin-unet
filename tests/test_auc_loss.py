import pytest
import tensorflow as tf
import numpy as np
from swin_transformer.AUC_LOSS import auc_focal_loss

def test_auc_focal_loss_output_shape_and_value():
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)

    loss_fn = auc_focal_loss(alpha=0.25, gamma=2.0)
    loss = loss_fn(y_true, y_pred)

    assert loss.shape == y_true.shape, "❌ Loss shape mismatch"
    assert not tf.math.reduce_any(tf.math.is_nan(loss)), "❌ Loss contains NaNs"
    assert tf.reduce_mean(loss).numpy() > 0, "❌ Loss should be positive"

def test_auc_focal_loss_extreme_predictions():
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    loss_fn = auc_focal_loss(alpha=0.25, gamma=2.0)
    loss = loss_fn(y_true, y_pred)

    assert tf.reduce_all(loss >= 0), "❌ Loss should be non-negative"

def test_auc_focal_loss_with_model_compile():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss=auc_focal_loss(alpha=0.25, gamma=2.0), metrics=["accuracy"])

    X = np.random.rand(4, 10).astype(np.float32)
    y = tf.keras.utils.to_categorical([0, 1, 1, 0], num_classes=2)

    history = model.fit(X, y, epochs=1, verbose=0)

    assert "loss" in history.history, "❌ Model did not track loss"
