# from __future__ import absolute_import

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
# from tensorflow.keras.activations import softmax
# from keras_swin_unet.util_layers import drop_path

# def window_partition(x, window_size):
#     _, H, W, C = x.get_shape().as_list()
#     patch_num_H = H // window_size
#     patch_num_W = W // window_size
#     x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
#     x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
#     windows = tf.reshape(x, shape=(-1, window_size, window_size, C))
#     return windows

# def window_reverse(windows, window_size, H, W, C):
#     patch_num_H = H // window_size
#     patch_num_W = W // window_size
#     x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
#     x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
#     x = tf.reshape(x, shape=(-1, H, W, C))
#     return x

# class Mlp(tf.keras.layers.Layer):
#     def __init__(self, filter_num, drop=0., name=''):
#         super().__init__()
#         self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
#         self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))
#         self.drop = Dropout(drop)
#         self.activation = tf.keras.activations.gelu

#     def call(self, x):
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class WindowAttention(tf.keras.layers.Layer):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., name=''):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads

#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.prefix = name

#         self.qkv = Dense(dim * 3, use_bias=qkv_bias, name='{}_attn_qkv'.format(self.prefix))
#         self.attn_drop = Dropout(attn_drop)
#         self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
#         self.proj_drop = Dropout(proj_drop)

#     def build(self, input_shape):
#         num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
#         self.relative_position_bias_table = self.add_weight(
#             name='{}_attn_pos'.format(self.prefix),
#             shape=(num_window_elements, self.num_heads),
#             initializer=tf.keras.initializers.Zeros(),
#             trainable=True
#         )

#         coords_h = np.arange(self.window_size[0])
#         coords_w = np.arange(self.window_size[1])
#         coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
#         coords = np.stack(coords_matrix)
#         coords_flatten = coords.reshape(2, -1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.transpose([1, 2, 0])
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)

#         with tf.init_scope():
#             self.relative_position_index = tf.Variable(
#                 initial_value=tf.convert_to_tensor(relative_position_index),
#                 trainable=False,
#                 name='{}_attn_pos_ind'.format(self.prefix)
#             )

#         self.built = True

#     def call(self, x, mask=None):
#         _, N, C = x.get_shape().as_list()
#         head_dim = C // self.num_heads

#         x_qkv = self.qkv(x)
#         x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
#         x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
#         q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

#         q = q * self.scale
#         k = tf.transpose(k, perm=(0, 1, 3, 2))
#         attn = tf.matmul(q, k)

#         num_window_elements = self.window_size[0] * self.window_size[1]
#         relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
#         relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
#         relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
#         relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
#         attn = attn + tf.expand_dims(relative_position_bias, axis=0)

#         if mask is not None:
#             nW = mask.get_shape()[0]
#             mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
#             attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
#             attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
#             attn = softmax(attn, axis=-1)
#         else:
#             attn = softmax(attn, axis=-1)

#         attn = self.attn_drop(attn)
#         x_qkv = tf.matmul(attn, v)
#         x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
#         x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
#         x_qkv = self.proj(x_qkv)
#         x_qkv = self.proj_drop(x_qkv)
#         return x_qkv

# class SwinTransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024,
#                  qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, name=''):
#         super().__init__()

#         self.dim = dim
#         self.num_patch = num_patch
#         self.num_heads = num_heads
#         self.window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
#         self.shift_size = shift_size if isinstance(shift_size, (list, tuple)) else [shift_size, shift_size]
#         self.num_mlp = num_mlp
#         self.prefix = name

#         # Ensure window size correctly fits within num_patch
#         if any(patch < window for patch, window in zip(self.num_patch, self.window_size)):
#             raise ValueError(f"Window size {self.window_size} is too large for num_patch {self.num_patch}.")

#         # Initialize Layers
#         self.norm1 = LayerNormalization(epsilon=1e-5, name='{}_norm1'.format(self.prefix))
#         self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
#                                     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
#         self.drop_path = drop_path(drop_path_prob)
#         self.norm2 = LayerNormalization(epsilon=1e-5, name='{}_norm2'.format(self.prefix))
#         self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)

#         # Assertions and conditions
#         assert 0 <= self.shift_size[0], 'shift_size >= 0 is required'

#         if min(self.num_patch) < min(self.window_size):
#             self.shift_size = [0, 0]
#             self.window_size = [min(self.num_patch), min(self.num_patch)]


#     def build(self, input_shape):
#         if any(s > 0 for s in self.shift_size):
#             H, W = self.num_patch
#             h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
#             w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))

#             mask_array = np.zeros((1, H, W, 1))
#             count = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     mask_array[:, h, w, :] = count
#                     count += 1
#             mask_array = tf.convert_to_tensor(mask_array)

#             mask_windows = window_partition(mask_array, self.window_size[0])
#             mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size[0] * self.window_size[1]])
#             attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
#             attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
#             attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
#             self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False, name='{}_attn_mask'.format(self.prefix))
#         else:
#             self.attn_mask = None

#         self.built = True

#     def call(self, x):
#         H, W = self.num_patch
#         B, L, C = x.get_shape().as_list()
#         assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'

#         x_skip = x
#         x = self.norm1(x)
#         x = tf.reshape(x, shape=(-1, H, W, C))

#         if any(s > 0 for s in self.shift_size):
#             shifted_x = tf.roll(x, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1, 2])
#         else:
#             shifted_x = x

#         x_windows = window_partition(shifted_x, self.window_size[0])
#         x_windows = tf.reshape(x_windows, shape=(-1, self.window_size[0] * self.window_size[1], C))

#         attn_windows = self.attn(x_windows, mask=self.attn_mask)

#         attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size[0], self.window_size[1], C))
#         shifted_x = window_reverse(attn_windows, self.window_size[0], H, W, C)

#         if any(s > 0 for s in self.shift_size):
#             x = tf.roll(shifted_x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1, 2])
#         else:
#             x = shifted_x

#         x = tf.reshape(x, shape=(-1, H * W, C))

#         x = self.drop_path(x)
#         x = x_skip + x

#         x_skip = x
#         x = self.norm2(x)
#         x = self.mlp(x)
#         x = self.drop_path(x)

#         return x_skip + x


from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.activations import softmax
from keras_swin_unet.util_layers import drop_path


def window_partition(x, window_size):
    _, H, W, C = x.get_shape().as_list()
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W, C):
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(
        windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C)
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, H, W, C))
    return x


class Mlp(tf.keras.layers.Layer):
    def __init__(self, filter_num, drop=0.0, name="", **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(filter_num[0], name="{}_mlp_0".format(name))
        self.fc2 = Dense(filter_num[1], name="{}_mlp_1".format(name))
        self.drop = Dropout(drop)
        self.activation = tf.keras.activations.gelu

    def call(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0.0,
        name="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.prefix = name

        self.qkv = Dense(
            dim * 3, use_bias=qkv_bias, name="{}_attn_qkv".format(self.prefix)
        )
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name="{}_attn_proj".format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            name="{}_attn_pos".format(self.prefix),
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        with tf.init_scope():
            self.relative_position_index = tf.Variable(
                initial_value=tf.convert_to_tensor(relative_position_index),
                trainable=False,
                name="{}_attn_pos_ind".format(self.prefix),
            )

        self.built = True

    def call(self, x, mask=None):
        _, N, C = x.get_shape().as_list()
        head_dim = C // self.num_heads

        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = tf.matmul(q, k)

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)

        attn = self.attn_drop(attn)
        x_qkv = tf.matmul(attn, v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.proj_drop(x_qkv)
        return x_qkv


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        qk_scale=None,
        mlp_drop=0,
        attn_drop=0,
        proj_drop=0,
        drop_path_prob=0,
        name="",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = (
            window_size
            if isinstance(window_size, (list, tuple))
            else [window_size, window_size]
        )
        self.shift_size = (
            shift_size
            if isinstance(shift_size, (list, tuple))
            else [shift_size, shift_size]
        )
        self.num_mlp = num_mlp
        self.prefix = name

        # Ensure window size correctly fits within num_patch
        if any(
            patch < window for patch, window in zip(self.num_patch, self.window_size)
        ):
            raise ValueError(
                f"Window size {self.window_size} is too large for num_patch {self.num_patch}."
            )

        # Initialize Layers
        self.norm1 = LayerNormalization(
            epsilon=1e-5, name="{}_norm1".format(self.prefix)
        )
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            name=self.prefix,
        )
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(
            epsilon=1e-5, name="{}_norm2".format(self.prefix)
        )
        self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)

        # Assertions and conditions
        assert 0 <= self.shift_size[0], "shift_size >= 0 is required"

        if min(self.num_patch) < min(self.window_size):
            self.shift_size = [0, 0]
            self.window_size = [min(self.num_patch), min(self.num_patch)]

    def build(self, input_shape):
        if any(s > 0 for s in self.shift_size):
            H, W = self.num_patch
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )

            mask_array = np.zeros((1, H, W, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            mask_windows = window_partition(mask_array, self.window_size[0])
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size[0] * self.window_size[1]]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                trainable=False,
                name="{}_attn_mask".format(self.prefix),
            )
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "Number of patches before and after Swin-MSA are mismatched."

        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, H, W, C))

        if any(s > 0 for s in self.shift_size):
            shifted_x = tf.roll(
                x, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size[0])
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size[0] * self.window_size[1], C)
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size[0], self.window_size[1], C)
        )
        shifted_x = window_reverse(attn_windows, self.window_size[0], H, W, C)

        if any(s > 0 for s in self.shift_size):
            x = tf.roll(
                shifted_x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, H * W, C))

        x = self.drop_path(x)
        x = x_skip + x

        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)

        return x_skip + x
