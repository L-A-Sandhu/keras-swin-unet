from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

# from keras_swin_unet import transformer_layers
from keras_swin_unet import transformer_layers, swin_layers

from keras_swin_unet import swin_layers


def get_model(
    input_size,
    filter_num_begin,
    depth,
    stack_num_down,
    stack_num_up,
    patch_size,
    num_heads,
    window_size,
    num_mlp,
    num_classes,
):
    """
    Create a model with flexible input size and output classes.

    Parameters:
    - input_size (tuple): The input image size as (height, width, channels).
    - filter_num_begin (int): Number of filters for the first convolutional layer.
    - depth (int): Depth of the network.
    - stack_num_down (int): Number of downsampling layers.
    - stack_num_up (int): Number of upsampling layers.
    - patch_size (tuple): Size of patches to be extracted from input.
    - num_heads (list): List of number of attention heads for each transformer block.
    - window_size (list): List of window sizes for each transformer block.
    - num_mlp (int): Size of the MLP layer.
    - num_classes (int): Number of output classes (this determines the number of channels in the final output).

    Returns:
    - model (keras.Model): The compiled model.
    """
    IN = Input(input_size)
    X = swin_unet_2d_base(
        IN,
        filter_num_begin,
        depth,
        stack_num_down,
        stack_num_up,
        patch_size,
        num_heads,
        window_size,
        num_mlp,
        shift_window=True,
        name="swin_unet",
    )
    n_labels = num_classes  # Number of output classes
    OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation="softmax")(
        X
    )  # Final output layer with num_classes channels
    model = Model(inputs=[IN], outputs=[OUT])
    return model


def swin_unet_2d_base(
    input_tensor,
    filter_num_begin,
    depth,
    stack_num_down,
    stack_num_up,
    patch_size,
    num_heads,
    window_size,
    num_mlp,
    shift_window=True,
    name="swin_unet",
):
    input_size = list(input_tensor.shape)[1:]
    num_patch_x = input_size[0] // patch_size[0]
    num_patch_y = input_size[1] // patch_size[1]
    embed_dim = filter_num_begin
    depth_ = depth
    X_skip = []
    X = input_tensor
    X = transformer_layers.patch_extract(patch_size)(X)
    X = transformer_layers.patch_embedding(num_patch_x * num_patch_y, embed_dim)(X)
    X = swin_transformer_stack(
        X,
        stack_num=stack_num_down,
        embed_dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads[0],
        window_size=window_size[0],
        num_mlp=num_mlp,
        shift_window=shift_window,
        name="{}_swin_down0".format(name),
    )
    X_skip.append(X)
    for i in range(depth_ - 1):
        X = transformer_layers.patch_merging(
            (num_patch_x, num_patch_y), embed_dim=embed_dim, name="down{}".format(i)
        )(X)
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2
        X = swin_transformer_stack(
            X,
            stack_num=stack_num_down,
            embed_dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads[i + 1],
            window_size=window_size[i + 1],
            num_mlp=num_mlp,
            shift_window=shift_window,
            name="{}_swin_down{}".format(name, i + 1),
        )
        X_skip.append(X)
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    X = X_skip[0]
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    for i in range(depth_decode):
        X = transformer_layers.patch_expanding(
            num_patch=(num_patch_x, num_patch_y),
            embed_dim=embed_dim,
            upsample_rate=2,
            return_vector=True,
        )(X)
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2
        X = concatenate([X, X_decode[i]], axis=-1, name="{}_concat_{}".format(name, i))
        X = Dense(
            embed_dim, use_bias=False, name="{}_concat_linear_proj_{}".format(name, i)
        )(X)
        X = swin_transformer_stack(
            X,
            stack_num=stack_num_up,
            embed_dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads[i],
            window_size=window_size[i],
            num_mlp=num_mlp,
            shift_window=shift_window,
            name="{}_swin_up{}".format(name, i),
        )
    X = transformer_layers.patch_expanding(
        num_patch=(num_patch_x, num_patch_y),
        embed_dim=embed_dim,
        upsample_rate=patch_size[0],
        return_vector=False,
    )(X)
    return X


def swin_transformer_stack(
    X,
    stack_num,
    embed_dim,
    num_patch,
    num_heads,
    window_size,
    num_mlp,
    shift_window=True,
    name="",
):
    mlp_drop_rate = 0
    attn_drop_rate = 0
    proj_drop_rate = 0
    drop_path_rate = 0
    qkv_bias = True
    qk_scale = None
    shift_size = window_size // 2 if shift_window else 0
    for i in range(stack_num):
        shift_size_temp = 0 if i % 2 == 0 else shift_size
        X = swin_layers.SwinTransformerBlock(
            dim=embed_dim,
            num_patch=num_patch,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size_temp,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_drop=mlp_drop_rate,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            drop_path_prob=drop_path_rate,
            name="name{}".format(i),
        )(X)
    return X
