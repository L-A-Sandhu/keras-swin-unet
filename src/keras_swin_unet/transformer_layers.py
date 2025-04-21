# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Dense, Conv2D
# from tensorflow.keras.layers import Embedding

# class patch_extract(Layer):
#     def __init__(self, patch_size):
#         super(patch_extract, self).__init__()
#         self.patch_size_x = patch_size[0]
#         self.patch_size_y = patch_size[1]

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         img_height, img_width = tf.shape(images)[1], tf.shape(images)[2]
#         channels = tf.shape(images)[3]

#         patches_per_row = img_height // self.patch_size_x
#         patches_per_col = img_width // self.patch_size_y
#         patch_num = patches_per_row * patches_per_col

#         sizes = [1, self.patch_size_x, self.patch_size_y, 1]
#         strides = [1, self.patch_size_x, self.patch_size_y, 1]
#         rates = [1, 1, 1, 1]
#         patches = tf.image.extract_patches(images=images, sizes=sizes, strides=strides, rates=rates, padding='VALID')

#         patch_dim = self.patch_size_x * self.patch_size_y * channels
#         patches = tf.reshape(patches, [batch_size, patch_num, patch_dim])

#         return patches
# class patch_embedding(Layer):
#     def __init__(self, num_patch, embed_dim):
#         super(patch_embedding, self).__init__()
#         self.num_patch = num_patch
#         self.embed_dim = embed_dim
#         self.proj = Dense(embed_dim)
#         self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

#     def build(self, input_shape):
#         super().build(input_shape)

#     def call(self, patch):
#         # Handle 3D input (batch_size, num_patches, patch_dim)
#         batch_size = tf.shape(patch)[0]
#         pos = tf.range(start=0, limit=self.num_patch, delta=1)
#         pos_embed = self.pos_embed(pos)

#         # Project patches to embedding space
#         patch_embed = self.proj(patch)

#         # Add positional embedding to the patch embedding
#         embed = patch_embed + pos_embed

#         return embed


# class patch_merging(Layer):
#     def __init__(self, num_patch, embed_dim, name=''):
#         super().__init__()
#         self.num_patch = num_patch
#         self.embed_dim = embed_dim
#         self.linear_trans = Dense(2 * embed_dim, use_bias=False, name='{}_linear_trans'.format(name))

#     def call(self, x):
#         H, W = self.num_patch
#         B, L, C = x.get_shape().as_list()
#         assert (L == H * W), 'input feature has wrong size'
#         assert (H % 2 == 0 and W % 2 == 0), '{}-by-{} patches received, they are not even.'.format(H, W)

#         x = tf.reshape(x, shape=(-1, H, W, C))
#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = tf.concat((x0, x1, x2, x3), axis=-1)
#         x = tf.reshape(x, shape=(-1, (H//2) * (W//2), 4 * C))
#         x = self.linear_trans(x)

#         return x

# class patch_expanding(Layer):
#     def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name=''):
#         super().__init__()
#         self.num_patch = num_patch
#         self.embed_dim = embed_dim
#         self.upsample_rate = upsample_rate
#         self.return_vector = return_vector
#         self.prefix = name if name else "default"

#         self.linear_trans1 = Conv2D(upsample_rate * embed_dim, kernel_size=1, use_bias=False,
#                                     name='{}_linear_trans1'.format(self.prefix))
#         self.linear_trans2 = Conv2D(upsample_rate * embed_dim, kernel_size=1, use_bias=False,
#                                     name='{}_linear_trans2'.format(self.prefix))

#     def call(self, x):
#         H, W = self.num_patch
#         B, L, C = x.get_shape().as_list()
#         assert (L == H * W), 'Input feature has wrong size'

#         x = tf.reshape(x, (-1, H, W, C))
#         x = self.linear_trans1(x)
#         x = tf.nn.depth_to_space(x, self.upsample_rate, data_format='NHWC',
#                                  name='{}_d_to_space'.format(self.prefix))

#         if self.return_vector:
#             x = tf.reshape(x, (-1, L * self.upsample_rate * self.upsample_rate, C // self.upsample_rate))

#         return x


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Embedding


class patch_extract(Layer):
    def __init__(self, patch_size, **kwargs):
        super(patch_extract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        img_height, img_width = tf.shape(images)[1], tf.shape(images)[2]
        channels = tf.shape(images)[3]

        patches_per_row = img_height // self.patch_size_x
        patches_per_col = img_width // self.patch_size_y
        patch_num = patches_per_row * patches_per_col

        sizes = [1, self.patch_size_x, self.patch_size_y, 1]
        strides = [1, self.patch_size_x, self.patch_size_y, 1]
        rates = [1, 1, 1, 1]
        patches = tf.image.extract_patches(
            images=images, sizes=sizes, strides=strides, rates=rates, padding="VALID"
        )

        patch_dim = self.patch_size_x * self.patch_size_y * channels
        patches = tf.reshape(patches, [batch_size, patch_num, patch_dim])

        return patches

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": (self.patch_size_x, self.patch_size_y),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_embedding(Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(patch_embedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        pos_embed = self.pos_embed(pos)

        patch_embed = self.proj(patch)
        embed = patch_embed + pos_embed

        return embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patch": self.num_patch,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_merging(Layer):
    def __init__(self, num_patch, embed_dim, name="", **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = Dense(
            2 * embed_dim, use_bias=False, name="{}_linear_trans".format(name)
        )

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert (
            H % 2 == 0 and W % 2 == 0
        ), "{}-by-{} patches received, they are not even.".format(H, W)

        x = tf.reshape(x, shape=(-1, H, W, C))
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (H // 2) * (W // 2), 4 * C))
        x = self.linear_trans(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patch": self.num_patch,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_expanding(Layer):
    def __init__(
        self, num_patch, embed_dim, upsample_rate, return_vector=True, name="", **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.prefix = name if name else "default"

        self.linear_trans1 = Conv2D(
            upsample_rate * embed_dim,
            kernel_size=1,
            use_bias=False,
            name="{}_linear_trans1".format(self.prefix),
        )
        self.linear_trans2 = Conv2D(
            upsample_rate * embed_dim,
            kernel_size=1,
            use_bias=False,
            name="{}_linear_trans2".format(self.prefix),
        )

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "Input feature has wrong size"

        x = tf.reshape(x, (-1, H, W, C))
        x = self.linear_trans1(x)
        x = tf.nn.depth_to_space(
            x,
            self.upsample_rate,
            data_format="NHWC",
            name="{}_d_to_space".format(self.prefix),
        )

        if self.return_vector:
            x = tf.reshape(
                x,
                (
                    -1,
                    L * self.upsample_rate * self.upsample_rate,
                    C // self.upsample_rate,
                ),
            )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patch": self.num_patch,
                "embed_dim": self.embed_dim,
                "upsample_rate": self.upsample_rate,
                "return_vector": self.return_vector,
                "name": self.prefix,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
