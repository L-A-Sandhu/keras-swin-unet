#!/usr/bin/env python
import os
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from swin_transformer.model_loader import get_model
from swin_transformer.split_data import split_dataset
from swin_transformer.data_loader import DynamicDataLoader
from swin_transformer.AUC_LOSS import (
    get_loss, auc_focal_loss,
    FocalLoss, DiceLoss, BCELoss, BCEDiceLoss,
    FocalDiceLoss, FocalTverskyLoss, TverskyLoss,
)
from keras_swin_unet import transformer_layers, swin_layers


def visualize_comparison(
    k,
    X_images,
    y,
    refined_segmentation,
    batch_idx,
    num_classes,
    font_size=12,
    font_style="DejaVu Sans",
    xtick_size=10,
    ytick_size=10,
):
    """Visualize original, predicted mask, ground truth, and TP/FP/FN/TN overlay."""
    plt.rcParams.update({"font.size": font_size, "font.family": font_style})
    plt.figure(figsize=(12, 12))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(X_images[batch_idx])
    plt.title("Original Image", fontsize=font_size, fontweight="bold")
    plt.axis("off")

    # Predicted Mask
    proposed_mask = np.argmax(refined_segmentation[batch_idx], axis=-1)
    plt.subplot(2, 2, 2)
    plt.imshow(X_images[batch_idx])
    plt.imshow(proposed_mask, alpha=0.5, cmap="coolwarm")
    plt.title("Predicted Mask", fontsize=font_size, fontweight="bold")
    plt.axis("off")

    # Ground Truth
    actual_mask = np.argmax(y[batch_idx], axis=-1)
    plt.subplot(2, 2, 3)
    plt.imshow(X_images[batch_idx])
    plt.imshow(actual_mask, alpha=0.5, cmap="coolwarm")
    plt.title("Actual Mask (Ground Truth)", fontsize=font_size, fontweight="bold")
    plt.axis("off")

    # TP/FP/FN/TN overlay
    TP = (proposed_mask == actual_mask) & (actual_mask == 1)
    FP = (proposed_mask != actual_mask) & (proposed_mask == 1)
    FN = (proposed_mask != actual_mask) & (actual_mask == 1)
    TN = (proposed_mask == actual_mask) & (actual_mask == 0)

    image_with_metrics = X_images[batch_idx].copy()
    image_with_metrics[TP] = [0, 255, 0]
    image_with_metrics[FP] = [255, 0, 0]
    image_with_metrics[FN] = [0, 0, 255]
    image_with_metrics[TN] = [128, 128, 128]

    plt.subplot(2, 2, 4)
    plt.imshow(image_with_metrics)
    plt.text(
        0.35, 1.05, "TP", color="green", fontsize=font_size,
        fontweight="bold", ha="center", transform=plt.gca().transAxes,
    )
    plt.text(
        0.45, 1.05, "FP", color="red", fontsize=font_size,
        fontweight="bold", ha="center", transform=plt.gca().transAxes,
    )
    plt.text(
        0.55, 1.05, "FN", color="blue", fontsize=font_size,
        fontweight="bold", ha="center", transform=plt.gca().transAxes,
    )
    plt.text(
        0.65, 1.05, "TN", color="gray", fontsize=font_size,
        fontweight="bold", ha="center", transform=plt.gca().transAxes,
    )
    plt.axis("off")
    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.tight_layout()
    plt.savefig(
        f"results_comparison_{batch_idx + 1}_{k}.png", bbox_inches="tight"
    )
    plt.close()
    return k + 1


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def run_train(args):
    # 1. Split
    ids = os.listdir(os.path.join(args.data, "images"))
    train_ids, val_ids, test_ids = split_dataset(
        ids, train_frac=0.8, val_frac=0.1, test_frac=0.1
    )

    # 2. DataLoaders
    def make_loader(ids, mode):
        return DynamicDataLoader(
            data_dir=args.data,
            ids=ids,
            batch_size=args.bs,
            img_size=tuple(args.input_shape),
            mode=mode,
            image_dtype=np.float32,
            mask_dtype=np.int32,
            num_classes=args.num_classes,
            input_scale=args.input_scale,
            mask_scale=args.mask_scale,
        )

    train_loader = make_loader(train_ids, "train")
    val_loader = make_loader(val_ids, "val")
    test_loader = make_loader(test_ids, "test")

    # 3. Model
    model = get_model(
        input_size=tuple(args.input_shape),
        filter_num_begin=args.filter,
        depth=args.depth,
        stack_num_down=args.stack_down,
        stack_num_up=args.stack_up,
        patch_size=tuple(args.patch_size),
        num_heads=args.num_heads,
        window_size=args.window_size,
        num_mlp=args.num_mlp,
        num_classes=args.num_classes,
    )

    # 4. Loss
    loss_kwargs = {
        "alpha": getattr(args, "alpha", 0.25),
        "gamma": getattr(args, "gamma", 2.0),
    }
    loss_fn = get_loss(args.loss_type, **loss_kwargs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4, clipvalue=0.5),
        loss=loss_fn,
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )

    # 5. Train
    os.makedirs(args.model_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            "val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.keras"),
            "val_loss",
            save_best_only=True,
        ),
    ]
    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # 6. Evaluate
    y_true, y_logits = [], []
    k = 0
    for X, y in test_loader:
        preds = model.predict(X)
        y_true.extend(y)
        y_logits.extend(preds)
        if args.visualize:
            for i in range(min(args.visualize - k, X.shape[0])):
                k = visualize_comparison(k, X, y, preds, i, args.num_classes)

    y_t = np.concatenate([yt.argmax(-1).flatten() for yt in y_true])
    y_p = np.concatenate([pl.argmax(-1).flatten() for pl in y_logits])

    cm = confusion_matrix(y_t, y_p)
    metrics = {
        "Accuracy": accuracy_score(y_t, y_p),
        "F1": f1_score(y_t, y_p, average="weighted"),
        "Precision": precision_score(y_t, y_p, average="weighted"),
        "Recall": recall_score(y_t, y_p, average="weighted"),
        "AUC": roc_auc_score(
            keras.utils.to_categorical(y_t),
            keras.utils.to_categorical(y_p),
            multi_class="ovr",
        ),
        "Confusion Matrix": cm.tolist(),
    }
    metrics_path = os.path.join(args.model_dir, "model_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done. Metrics:", json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_infer(args):
    # Load model
    custom_objects = {
        "FocalLoss": FocalLoss,
        "DiceLoss": DiceLoss,
        "BCELoss": BCELoss,
        "BCEDiceLoss": BCEDiceLoss,
        "FocalDiceLoss": FocalDiceLoss,
        "FocalTverskyLoss": FocalTverskyLoss,
        "TverskyLoss": TverskyLoss,
        "auc_focal_loss_fixed": auc_focal_loss(alpha=args.alpha, gamma=args.gamma),
        **transformer_layers.__dict__,
        **swin_layers.__dict__,
    }
    model = load_model(
        os.path.join(args.model_dir, "best_model.keras"),
        custom_objects=custom_objects,
    )

    # --- Single-image mode ---
    if args.image:
        img = np.array(
            Image.open(args.image)
            .convert("RGB")
            .resize((args.input_shape[1], args.input_shape[0]))
        )
        inp = img.astype(np.float32)[None] / args.input_scale
        preds = model.predict(inp)[0]
        mask = preds.argmax(-1)

        if args.visualize > 1:
            visualize_comparison(
                0, inp, np.array([preds]), np.array([preds]), 0, args.num_classes
            )
        else:
            plt.imshow(img)
            plt.imshow(mask, alpha=0.5, cmap="coolwarm")
            plt.axis("off")
            plt.savefig(args.output, bbox_inches="tight")
            print(f"Saved overlay to {args.output}")
        return

    # --- Test-loader mode ---
    print("Running inference on test dataset...")
    all_ids = os.listdir(os.path.join(args.data, "images"))
    _, _, test_ids = split_dataset(
        all_ids, train_frac=0.8, val_frac=0.1, test_frac=0.1
    )

    test_loader = DynamicDataLoader(
        data_dir=args.data,
        ids=test_ids,
        batch_size=args.bs,
        img_size=tuple(args.input_shape),
        mode="test",
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=args.num_classes,
        input_scale=args.input_scale,
        mask_scale=args.mask_scale,
    )

    y_true_all, y_pred_all = [], []
    total_time, num_examples = 0, 0
    k = 0

    for X_batch, y_batch in test_loader:
        start_time = time.time()
        preds = model.predict(X_batch)
        total_time += time.time() - start_time
        num_examples += X_batch.shape[0]

        y_true_all.extend(np.argmax(y_batch, axis=-1).flatten())
        y_pred_all.extend(np.argmax(preds, axis=-1).flatten())

        if args.visualize and k < args.visualize:
            for i in range(min(args.visualize - k, X_batch.shape[0])):
                k = visualize_comparison(
                    k, X_batch, y_batch, preds, i, args.num_classes
                )

    accuracy = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    precision = precision_score(y_true_all, y_pred_all, average="macro")
    recall = recall_score(y_true_all, y_pred_all, average="macro")
    auc = roc_auc_score(
        keras.utils.to_categorical(y_true_all),
        keras.utils.to_categorical(y_pred_all),
        multi_class="ovr",
    )
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    avg_time_ms = (total_time / num_examples) * 1000

    metrics = {
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "Confusion Matrix": conf_matrix.tolist(),
        "Average Inference Time per Image (ms)": avg_time_ms,
    }

    metrics_path = os.path.join(args.model_dir, "model_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Inference completed. Metrics saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        prog="swin-unet", description="Train, evaluate+visualize, or infer"
    )
    sp = p.add_subparsers(dest="command", required=True)

    # --- train ---
    t = sp.add_parser("train", help="Train & evaluate")
    t.add_argument("--data", default="./data")
    t.add_argument("--model-dir", default="./checkpoint")
    t.add_argument("--num-classes", type=int, default=2)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--filter", type=int, default=128)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--stack-down", type=int, default=2)
    t.add_argument("--stack-up", type=int, default=2)
    t.add_argument("--patch-size", type=int, nargs=2, default=[4, 4])
    t.add_argument("--num-heads", type=int, nargs="+", default=[4, 8, 8, 8])
    t.add_argument("--window-size", type=int, nargs="+", default=[4, 2, 2, 2])
    t.add_argument("--num-mlp", type=int, default=512)
    t.add_argument("--gamma", type=float, default=2.0)
    t.add_argument("--alpha", type=float, default=0.25)
    t.add_argument(
        "--loss-type",
        default="focal",
        choices=["focal", "dice", "bce", "bce_dice", "focal_dice", "focal_tversky", "tversky"],
        help="Loss function to use",
    )
    t.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    t.add_argument("--input-scale", type=int, default=255)
    t.add_argument("--mask-scale", type=int, default=255)
    t.add_argument("--visualize", type=int, default=0)
    t.set_defaults(func=run_train)

    # --- infer ---
    i = sp.add_parser("infer", help="Run inference on a single image or test set")
    i.add_argument("--model-dir", default="./checkpoint")
    i.add_argument("--image", help="Path to single image (optional)")
    i.add_argument("--output", default="out.png")
    i.add_argument("--num-classes", type=int, default=2)
    i.add_argument("--gamma", type=float, default=2.0)
    i.add_argument("--alpha", type=float, default=0.25)
    i.add_argument("--loss-type", default="focal")
    i.add_argument("--input-scale", type=int, default=255)
    i.add_argument("--data", default="./data")
    i.add_argument("--bs", type=int, default=32)
    i.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    i.add_argument("--visualize", type=int, default=0)
    i.set_defaults(func=run_infer)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
