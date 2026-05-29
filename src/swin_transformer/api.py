from types import SimpleNamespace


def swin_train(**kwargs):
    """Keyword-friendly wrapper for training.

    Parameters
    ----------
    data : str
        Dataset path (must contain images/ and masks/).
    model_dir : str
        Folder to save model + logs.
    num_classes : int
        Number of output classes (2 for binary).
    epochs : int
        Training epochs.
    bs : int
        Batch size.
    patience : int
        Early stopping patience.
    filter : int
        Initial number of filters / embedding dimension.
    depth : int
        Depth of encoder / decoder (number of stages).
    stack_down : int
        Number of Swin transformer blocks per encoder stage.
    stack_up : int
        Number of Swin transformer blocks per decoder stage.
    patch_size : list
        Patch size for initial embedding.
    num_heads : list
        Attention heads per stage (length must match depth).
    window_size : list
        Window size per stage (length must match depth).
    num_mlp : int
        MLP hidden size in Swin blocks.
    gamma : float
        Gamma for focal-based losses.
    alpha : float
        Alpha for focal-based losses.
    loss_type : str
        Loss function: "focal", "dice", "bce", "bce_dice",
        "focal_dice", "focal_tversky", "tversky".
    input_shape : list
        Input image shape [H, W, C].
    input_scale : int
        Divide input by this value.
    mask_scale : int
        Divide mask by this value.
    visualize : int
        Number of test samples to visualize.
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
        "loss_type": "focal",
        "input_shape": [512, 512, 3],
        "input_scale": 255,
        "mask_scale": 255,
        "visualize": 2,
        **kwargs,
    }
    from swin_transformer.cli import run_train
    run_train(SimpleNamespace(**args))


def swin_infer(**kwargs):
    """Flexible wrapper for inference on either a single image or the test dataset.

    If 'image' is provided, runs single-image inference.
    Otherwise, runs batch inference on the test set using the data loader.
    """
    args = {
        "model_dir": "./checkpoint",
        "image": None,
        "output": "output.png",
        "num_classes": 2,
        "gamma": 2.0,
        "alpha": 0.25,
        "loss_type": "focal",
        "input_scale": 255,
        "mask_scale": 255,
        "data": "./demo_data",
        "bs": 4,
        "input_shape": [512, 512, 3],
        "visualize": 1,
        **kwargs,
    }
    from swin_transformer.cli import run_infer
    run_infer(SimpleNamespace(**args))


__all__ = ["swin_train", "swin_infer"]
