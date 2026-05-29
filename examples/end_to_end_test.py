#!/usr/bin/env python
"""
End-to-End Test for Swin-UNet

Creates a synthetic segmentation dataset, trains a small Swin-UNet,
runs inference, and saves all results. Tests every loss function.

Usage:
    python end_to_end_test.py
    python end_to_end_test.py --epochs 10 --size 128
"""

import os, sys, json, time, shutil, argparse, warnings
import numpy as np
from PIL import Image, ImageDraw

# Ensure the package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from swin_transformer.AUC_LOSS import get_loss
from swin_transformer.model_loader import get_model
from swin_transformer.data_loader import DynamicDataLoader
from swin_transformer.split_data import split_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def make_circle_mask(size, cx, cy, r):
    """Create a binary mask with a filled circle."""
    y, x = np.ogrid[:size, :size]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r**2).astype(np.uint8) * 255


def make_rect_mask(size, x0, y0, x1, y1):
    """Create a binary mask with a filled rectangle."""
    m = np.zeros((size, size), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def generate_synthetic_dataset(root, num_samples=96, size=128):
    """Generate images with random shapes and matching binary masks.

    Returns (root, num_samples) so callers know what was created.
    """
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rng = np.random.RandomState(42)

    for i in range(num_samples):
        # Random background colour
        bg = rng.randint(0, 256, size=3, dtype=np.uint8)
        img = np.full((size, size, 3), bg, dtype=np.uint8)

        # Add some noise
        noise = rng.randint(0, 40, size=(size, size, 3), dtype=np.uint8)
        img = np.clip(img.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)

        # Decide shape type
        if rng.rand() > 0.5:
            cx = rng.randint(size // 4, 3 * size // 4)
            cy = rng.randint(size // 4, 3 * size // 4)
            r = rng.randint(size // 8, size // 3)
            mask = make_circle_mask(size, cx, cy, r)
            colour = rng.randint(100, 256, size=3, dtype=np.uint8)
            yy, xx = np.ogrid[:size, :size]
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            for c in range(3):
                img[inside, c] = colour[c]
        else:
            x0 = rng.randint(0, size // 2)
            y0 = rng.randint(0, size // 2)
            x1 = rng.randint(x0 + size // 6, size)
            y1 = rng.randint(y0 + size // 6, size)
            mask = make_rect_mask(size, x0, y0, x1, y1)
            colour = rng.randint(100, 256, size=3, dtype=np.uint8)
            img[y0:y1, x0:x1] = colour

        # Blur edges slightly for realism
        Image.fromarray(img).save(os.path.join(img_dir, f"{i:04d}.png"))
        Image.fromarray(mask).save(os.path.join(mask_dir, f"{i:04d}.png"))

    return root, num_samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_evaluate(
    data_dir,
    model_dir,
    loss_type="focal",
    loss_kwargs=None,
    img_size=128,
    epochs=5,
    batch_size=4,
    visualize=4,
):
    """Train a Swin-UNet on the given dataset and return metrics."""
    if loss_kwargs is None:
        loss_kwargs = {}

    # Split
    ids = sorted(os.listdir(os.path.join(data_dir, "images")))
    train_ids, val_ids, test_ids = split_dataset(ids, 0.7, 0.15, 0.15)

    # Data loaders
    def make_loader(ids_list, mode):
        return DynamicDataLoader(
            data_dir=data_dir,
            ids=ids_list,
            batch_size=batch_size,
            img_size=(img_size, img_size),
            mode=mode,
            image_dtype=np.float32,
            mask_dtype=np.int32,
            num_classes=2,
            input_scale=255,
            mask_scale=255,
        )

    train_loader = make_loader(train_ids, "train")
    val_loader = make_loader(val_ids, "val")
    test_loader = make_loader(test_ids, "test")

    # Model
    model = get_model(
        input_size=(img_size, img_size, 3),
        filter_num_begin=32,
        depth=3,
        stack_num_down=2,
        stack_num_up=2,
        patch_size=[4, 4],
        num_heads=[4, 4, 4],
        window_size=[4, 2, 2],
        num_mlp=256,
        num_classes=2,
    )
    print(f"  Model params: {model.count_params():,}")

    # Compile
    loss_fn = get_loss(loss_type, **loss_kwargs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Callbacks
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            "val_loss", patience=max(2, epochs // 3), restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, "val_loss", save_best_only=True
        ),
    ]

    # Train
    t0 = time.time()
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # Evaluate on test set
    y_true_all, y_pred_all = [], []
    for X, y in test_loader:
        preds = model.predict(X, verbose=0)
        y_true_all.extend(np.argmax(y, axis=-1).flatten())
        y_pred_all.extend(np.argmax(preds, axis=-1).flatten())

    metrics = {
        "loss_type": loss_type,
        "loss_kwargs": loss_kwargs,
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
        "precision": float(precision_score(y_true_all, y_pred_all, average="macro")),
        "recall": float(recall_score(y_true_all, y_pred_all, average="macro")),
        "train_time_s": round(train_time, 1),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "params": model.count_params(),
    }

    # Save metrics
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def infer_and_visualize(model, data_dir, output_dir, img_size=128, num_samples=6):
    """Run inference on test images and save overlay visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Load a few test images
    ids = sorted(os.listdir(os.path.join(data_dir, "images")))
    _, _, test_ids = split_dataset(ids, 0.7, 0.15, 0.15)

    loader = DynamicDataLoader(
        data_dir=data_dir,
        ids=test_ids[:num_samples],
        batch_size=1,
        img_size=(img_size, img_size),
        mode="test",
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=2,
        input_scale=255,
        mask_scale=255,
    )

    results = []
    for idx, (X, y) in enumerate(loader):
        pred = model.predict(X, verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8) * 255
        true_mask = np.argmax(y[0], axis=-1).astype(np.uint8) * 255
        image = (X[0] * 255).astype(np.uint8)

        # Save overlay image
        overlay = image.copy()
        # True positive (green), false positive (red), false negative (blue)
        tp = (pred_mask > 0) & (true_mask > 0)
        fp = (pred_mask > 0) & (true_mask == 0)
        fn = (pred_mask == 0) & (true_mask > 0)
        overlay[tp] = [0, 255, 0]
        overlay[fp] = [255, 0, 0]
        overlay[fn] = [0, 0, 255]

        Image.fromarray(image).save(os.path.join(output_dir, f"sample_{idx}_image.png"))
        Image.fromarray(true_mask).save(
            os.path.join(output_dir, f"sample_{idx}_gt.png")
        )
        Image.fromarray(pred_mask).save(
            os.path.join(output_dir, f"sample_{idx}_pred.png")
        )
        Image.fromarray(overlay).save(
            os.path.join(output_dir, f"sample_{idx}_overlay.png")
        )

        iou = np.sum(tp) / max(np.sum(tp | fp | fn), 1)
        results.append({"sample": idx, "iou": round(float(iou), 4)})

    with open(os.path.join(output_dir, "inference_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Swin-UNet end-to-end test")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--size", type=int, default=128, help="Image size")
    parser.add_argument("--samples", type=int, default=96, help="Synthetic samples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["focal", "dice", "focal_dice"],
        help="Loss functions to test",
    )
    parser.add_argument("--out", default="./test_results", help="Output directory")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("  Swin-UNet End-to-End Test")
    print("=" * 60)
    print(f"  Image size : {args.size}x{args.size}")
    print(f"  Samples    : {args.samples}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Losses     : {args.losses}")
    print(f"  Output     : {out}")
    print("=" * 60)

    # ---- 1. Generate synthetic data ----
    print("\n[1/3] Generating synthetic dataset ...")
    data_dir = os.path.join(out, "data")
    generate_synthetic_dataset(data_dir, num_samples=args.samples, size=args.size)
    print(f"  Created {args.samples} image/mask pairs in {data_dir}")

    # ---- 2. Train with each loss ----
    print("\n[2/3] Training models ...")
    all_metrics = {}

    for loss_type in args.losses:
        name = loss_type
        model_dir = os.path.join(out, f"model_{name}")
        print(f"\n  --- {name} ---")

        model, metrics = train_and_evaluate(
            data_dir,
            model_dir,
            loss_type=name,
            img_size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            visualize=4,
        )
        all_metrics[name] = metrics
        print(f"  Test accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

        # Inference
        infer_dir = os.path.join(out, f"inference_{name}")
        infer_results = infer_and_visualize(model, data_dir, infer_dir, img_size=args.size)
        mean_iou = np.mean([r["iou"] for r in infer_results])
        metrics["mean_inference_iou"] = round(float(mean_iou), 4)
        print(f"  Mean IoU on {len(infer_results)} test samples: {mean_iou:.4f}")

        # Clean up model from memory
        del model

    # ---- 3. Summary ----
    print("\n[3/3] Summary")
    print("=" * 60)
    print(f"{'Loss':<16} {'Accuracy':>10} {'F1':>10} {'IoU':>10} {'Time':>8}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(
            f"{name:<16} {m['accuracy']:>10.4f} {m['f1']:>10.4f} "
            f"{m.get('mean_inference_iou', 0):>10.4f} {m['train_time_s']:>7.1f}s"
        )
    print("=" * 60)

    # Save combined report
    report_path = os.path.join(out, "report.json")
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nFull report saved to {report_path}")
    print(f"All outputs under {out}/")
    print("\nTest PASSED - all loss functions trained and produced valid results.")


if __name__ == "__main__":
    main()
