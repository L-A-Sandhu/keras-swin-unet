# import pytest
# import numpy as np
# from swin_transformer.model_loader import get_model

# def test_model_creation_default_shape():
#     model = get_model(
#         filter_num_begin=64,
#         depth=2,
#         stack_num_down=1,
#         stack_num_up=1,
#         patch_size=(4, 4),
#         num_heads=[2, 4],
#         window_size=[4, 2],
#         num_mlp=128,
#         num_classes=2
#     )
    
#     assert model.input_shape == (None, 512, 512, 3), "❌ Input shape mismatch"
#     assert model.output_shape == (None, 512, 512, 2), "❌ Output shape mismatch"
#     assert isinstance(model.count_params(), int), "❌ Model does not compile properly"

# def test_model_forward_pass():
#     model = get_model(
#         filter_num_begin=32,
#         depth=2,
#         stack_num_down=1,
#         stack_num_up=1,
#         patch_size=(4, 4),
#         num_heads=[2, 4],
#         window_size=[4, 2],
#         num_mlp=64,
#         num_classes=2
#     )

#     dummy_input = np.random.rand(1, 512, 512, 3).astype(np.float32)
#     output = model.predict(dummy_input)
#     assert output.shape == (1, 512, 512, 2), "❌ Forward output shape is incorrect"
#     assert np.allclose(np.sum(output, axis=-1), 1.0, atol=1e-2), "❌ Output is not softmax normalized"

# @pytest.mark.parametrize("num_classes", [1, 3, 4])
# def test_model_output_classes(num_classes):
#     model = get_model(
#         filter_num_begin=16,
#         depth=2,
#         stack_num_down=1,
#         stack_num_up=1,
#         patch_size=(4, 4),
#         num_heads=[2, 4],
#         window_size=[4, 2],
#         num_mlp=64,
#         num_classes=num_classes
#     )
#     assert model.output_shape[-1] == num_classes, f"❌ Output channels != {num_classes}"
import pytest
import numpy as np
from swin_transformer.model_loader import get_model

def test_model_creation_default_shape():
    model = get_model(
        input_size=(512, 512, 3),  # Pass input_size explicitly
        filter_num_begin=64,
        depth=2,
        stack_num_down=1,
        stack_num_up=1,
        patch_size=(4, 4),
        num_heads=[2, 4],
        window_size=[4, 2],
        num_mlp=128,
        num_classes=2
    )
    
    assert model.input_shape == (None, 512, 512, 3), "❌ Input shape mismatch"
    assert model.output_shape == (None, 512, 512, 2), "❌ Output shape mismatch"
    assert isinstance(model.count_params(), int), "❌ Model does not compile properly"

def test_model_forward_pass():
    model = get_model(
        input_size=(512, 512, 3),  # Pass input_size explicitly
        filter_num_begin=32,
        depth=2,
        stack_num_down=1,
        stack_num_up=1,
        patch_size=(4, 4),
        num_heads=[2, 4],
        window_size=[4, 2],
        num_mlp=64,
        num_classes=2
    )

    dummy_input = np.random.rand(1, 512, 512, 3).astype(np.float32)
    output = model.predict(dummy_input)
    assert output.shape == (1, 512, 512, 2), "❌ Forward output shape is incorrect"
    assert np.allclose(np.sum(output, axis=-1), 1.0, atol=1e-2), "❌ Output is not softmax normalized"

@pytest.mark.parametrize("num_classes", [1, 3, 4])
def test_model_output_classes(num_classes):
    model = get_model(
        input_size=(512, 512, 3),  # Pass input_size explicitly
        filter_num_begin=16,
        depth=2,
        stack_num_down=1,
        stack_num_up=1,
        patch_size=(4, 4),
        num_heads=[2, 4],
        window_size=[4, 2],
        num_mlp=64,
        num_classes=num_classes
    )
    assert model.output_shape[-1] == num_classes, f"❌ Output channels != {num_classes}"

@pytest.mark.parametrize("input_shape", [(256, 256), (128, 128), (64, 64)])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_model_input_output_variability(input_shape, num_classes):
    h, w = input_shape
    model = get_model(
        input_size=(h, w, 3),  # Pass input_size explicitly
        filter_num_begin=16,
        depth=2,
        stack_num_down=1,
        stack_num_up=1,
        patch_size=(4, 4),
        num_heads=[2, 4],
        window_size=[4, 2],
        num_mlp=64,
        num_classes=num_classes
    )
    
    dummy_input = np.random.rand(1, h, w, 3).astype(np.float32)
    output = model.predict(dummy_input)
    
    assert output.shape == (1, h, w, num_classes), f"❌ Output shape mismatch for input shape {input_shape} and num_classes {num_classes}"

