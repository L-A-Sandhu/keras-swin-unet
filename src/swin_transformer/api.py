from types import SimpleNamespace
from swin_transformer.cli import run_train, run_infer


# ─────────────────────────────────────────────
# Training wrapper
# ─────────────────────────────────────────────
def swin_train(**kwargs):
    """
    Keyword-friendly wrapper for training using default + user-specified args.
    """
    args = {
        "data": "./demo_data",
        "model_dir": "./checkpoint",
        "num_classes": 2,
        "bs": 4,
        "epochs": 5,
        "patience": 3,
        "filter": 128,
        "depth": 4,
        "stack_down": 2,
        "stack_up": 2,
        "patch_size": [4, 4],
        "num_heads": [4, 8, 8, 8],
        "window_size": [4, 2, 2, 2],
        "num_mlp": 512,
        "gamma": 2.0,
        "alpha": 0.25,
        "input_shape": [512, 512, 3],
        "input_scale": 255,
        "mask_scale": 255,
        "visualize": 2,
        **kwargs,
    }
    run_train(SimpleNamespace(**args))


# ─────────────────────────────────────────────
# Inference wrapper using test split
# ─────────────────────────────────────────────
def swin_infer(**kwargs):
    """
    Wrapper for inference using test split or single image.
    """
    args = {
        "data": "./demo_data",
        "model_dir": "./checkpoint",
        "num_classes": 2,
        "gamma": 0.25,
        "alpha": 2.0,
        "input_scale": 255,
        "mask_scale": 255,  # ✅ ADD THIS LINE
        "visualize": 0,
        "output": "out.png",
        "image": None,  # ← if provided, runs single image mode
    }
    args.update(kwargs)
    run_infer(SimpleNamespace(**args))
