import os
import numpy as np
import pytest
from swin_transformer.data_loader import DynamicDataLoader
from PIL import Image

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "demo_data"))
IMAGES_DIR = os.path.join(DEMO_DATA_DIR, "images")
MASKS_DIR = os.path.join(DEMO_DATA_DIR, "masks")

@pytest.fixture(scope="module")
def jpg_ids():
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(".jpg") and os.path.exists(os.path.join(MASKS_DIR, f.replace(".jpg", ".png")))
    ])
    if not image_files:
        pytest.skip("❌ No matching .jpg image + .png mask pairs found in demo_data/")
    return image_files

@pytest.fixture(scope="module")
def loader(jpg_ids):
    return DynamicDataLoader(
        data_dir=DEMO_DATA_DIR,
        ids=jpg_ids,
        batch_size=1,
        img_size=(128, 128),
        num_classes=2,
        input_scale=255,
        mask_scale=255
    )

def test_loader_length_matches_input(jpg_ids, loader):
    assert len(loader) == len(jpg_ids), "❌ Loader length doesn't match number of image-mask pairs"

def test_loader_reads_jpg_and_png(loader):
    X, y = loader[0]
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (1, 128, 128, 3), "❌ Image shape incorrect"
    assert y.shape == (1, 128, 128, 2), "❌ Mask shape incorrect or not one-hot encoded"

def test_mask_values_are_binary(loader):
    _, y = loader[0]
    y_sum = np.sum(y, axis=-1)
    assert np.all(np.isin(y_sum, [1])), "❌ Mask values are not valid one-hot encoded binary masks"

def test_grayscale_image_converted_to_rgb(jpg_ids):
    gray_img_path = os.path.join(IMAGES_DIR, jpg_ids[0])
    img = Image.open(gray_img_path).convert("L")
    img.save(gray_img_path)  # overwrite with grayscale for test

    loader = DynamicDataLoader(
        data_dir=DEMO_DATA_DIR,
        ids=[jpg_ids[0]],
        batch_size=1,
        img_size=(64, 64),
        num_classes=2,
        input_scale=255,
        mask_scale=255
    )
    X, _ = loader[0]
    assert X.shape[-1] == 3, "❌ Grayscale image was not converted to 3-channel RGB"
