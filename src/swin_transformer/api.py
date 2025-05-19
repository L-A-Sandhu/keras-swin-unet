from types import SimpleNamespace
from swin_transformer.cli import run_train, run_infer


def swin_train(**kwargs):
    """
    Keyword-friendly wrapper for training
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
        **kwargs,  # override defaults
    }
    run_train(SimpleNamespace(**args))


# def swin_infer(**kwargs):
#     args = {
#         "model_dir": "./checkpoint",
#         "image": "./demo_data/images/104.jpg",
#         "output": "output.png",
#         "num_classes": 2,
#         "gamma": 2.0,
#         "alpha": 0.25,
#         "input_scale": 255,
#         "visualize": 1,
#         **kwargs,
#     }
#     run_infer(SimpleNamespace(**args))
def swin_infer(**kwargs):
    """
    Flexible wrapper for inference on either a single image or the test dataset.
    If 'image' is provided in kwargs, it runs single-image inference.
    Otherwise, it runs batch inference on the test set using the data loader.
    """
    args = {
        "model_dir": "./checkpoint",
        "image": None,  # if None, use test loader
        "output": "output.png",
        "num_classes": 2,
        "gamma": 2.0,
        "alpha": 0.25,
        "input_scale": 255,
        "data": "./demo_data",  # required for test loader fallback
        "bs": 4,                # batch size for test loader
        "input_shape": [512, 512, 3],
        "visualize": 1,
        **kwargs,
    }
    run_infer(SimpleNamespace(**args))



__all__ = ["swin_train", "swin_infer"]
