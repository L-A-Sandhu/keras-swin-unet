import sys, os
import numpy as np
import pytest
import warnings
from PIL import Image

# Add project src to path for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from swin_transformer.data_loader import DynamicDataLoader

# Fixture to generate mixed-format datasets with exact unique values
@pytest.fixture(params=[
    ('png', 'png', [0, 255]),
    ('jpg', 'png', [0, 255]),
    ('tif', 'png', [0, 255]),
    ('npy', 'png', [0, 127, 255]),
])
def temp_dataset(tmp_path, request):
    img_ext, mask_ext, unique_vals = request.param
    demo = tmp_path / "demo_data"
    imgs = demo / "images"
    msks = demo / "masks"
    imgs.mkdir(parents=True)
    msks.mkdir()

    ids = []
    # Create two samples per format
    for idx in [1, 2]:
        name = f"img{idx}"
        img_path = imgs / f"{name}.{img_ext}"
        mask_path = msks / f"{name}.{mask_ext}"

        # Generate image
        if img_ext == 'npy':
            arr = (np.random.rand(32, 32, 3) * 65535).astype(np.uint16)
            np.save(img_path, arr)
        else:
            arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(img_path)

        # Generate mask with required unique values
        m = np.random.choice(unique_vals, size=(32, 32)).astype(np.uint8)
        # Force one pixel of each class
        for i, val in enumerate(unique_vals):
            m.flat[i] = val
        Image.fromarray(m).save(mask_path)

        ids.append(f"{name}.{img_ext}")

    return str(demo), ids, unique_vals

def test_loader_handles_all_formats(temp_dataset):
    demo_dir, ids, unique_vals = temp_dataset
    num_classes = len(unique_vals)
    loader = DynamicDataLoader(
        data_dir=demo_dir,
        ids=ids,
        batch_size=2,
        img_size=(32, 32),
        num_classes=num_classes,
        input_scale=65535 if ids[0].endswith('.npy') else 255,
        mask_scale=255
    )
    assert len(loader) == 1
    X, y = loader[0]
    assert X.shape == (2, 32, 32, 3)
    assert y.shape == (2, 32, 32, num_classes)
    assert np.allclose(np.sum(y, axis=-1), 1)

def test_error_on_too_few_classes(temp_dataset):
    demo_dir, ids, unique_vals = temp_dataset
    true_classes = len(unique_vals)
    if true_classes > 1:
        loader = DynamicDataLoader(
            data_dir=demo_dir,
            ids=[ids[1]],
            batch_size=1,
            img_size=(32, 32),
            num_classes=true_classes - 1,
            input_scale=65535 if ids[1].endswith('.npy') else 255,
            mask_scale=255
        )
        with pytest.raises(ValueError, match="exceeds the number of classes"):
            _ = loader[0]

@pytest.mark.parametrize("target_classes", [4, 5, 6])
def test_missing_classes_up_to_six(tmp_path, target_classes):
    # Create a mask with exactly 3 values: 0,127,255
    demo = tmp_path / "demo_multi"
    imgs = demo / "images"; msks = demo / "masks"
    imgs.mkdir(parents=True); msks.mkdir()

    Image.fromarray((np.random.rand(32,32,3)*255).astype(np.uint8)).save(imgs/"sample.png")
    mask = np.random.choice([0,127,255], size=(32,32)).astype(np.uint8)
    # Force one pixel of each class
    for i, v in enumerate([0,127,255]):
        mask.flat[i] = v
    Image.fromarray(mask).save(msks/"sample.png")

    loader = DynamicDataLoader(
        data_dir=str(demo),
        ids=["sample.png"],
        batch_size=1,
        img_size=(32,32),
        num_classes=target_classes,
        input_scale=255,
        mask_scale=255
    )

    # Warn on fewer unique mask values
    with pytest.warns(UserWarning, match="less than the number of classes"):
        X, y = loader[0]

    assert y.shape == (1, 32, 32, target_classes)
    # Missing channels (beyond index 0,2,255→mapped classes) should be zero
    present_indices = [0,1,2]  # mask values 0→class0, 127→class1,255→class2
    for cls in range(target_classes):
        if cls not in present_indices:
            assert np.all(y[0,:,:,cls] == 0)
    assert np.allclose(np.sum(y, axis=-1), 1)

def test_empty_data_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        DynamicDataLoader(
            data_dir=str(empty),
            ids=["none"],
            batch_size=1,
            img_size=(16, 16),
            num_classes=2
        )

def test_index_out_of_range(tmp_path):
    demo = tmp_path / "demo_idx"
    imgs = demo / "images"; msks = demo / "masks"
    imgs.mkdir(parents=True); msks.mkdir()
    Image.fromarray((np.random.rand(16,16,3)*255).astype(np.uint8)).save(imgs/"only.png")
    Image.fromarray((np.random.choice([0,255], size=(16,16))).astype(np.uint8)).save(msks/"only.png")

    loader = DynamicDataLoader(
        data_dir=str(demo),
        ids=["only.png"],
        batch_size=1,
        img_size=(16,16),
        num_classes=2,
        input_scale=255,
        mask_scale=255
    )
    with pytest.raises(IndexError):
        _ = loader[1]
