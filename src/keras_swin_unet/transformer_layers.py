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
        config.update({"patch_size": (self.patch_size_x, self.patch_size_y)})
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
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        pos_embed = self.pos_embed(pos)
        patch_embed = self.proj(patch)
        embed = patch_embed + pos_embed
        return embed

    def get_config(self):
        config = super().get_config()
        config.update({"num_patch": self.num_patch, "embed_dim": self.embed_dim})
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
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (H // 2) * (W // 2), 4 * C))
        x = self.linear_trans(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"num_patch": self.num_patch, "embed_dim": self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class patch_expanding(Layer):
    def __init__(
        self,
        num_patch,
        embed_dim,
        upsample_rate,
        return_vector=True,
        output_dim=None,
        name="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.prefix = name if name else "default"

        # output_dim: desired channels after depth_to_space.
        # If None, defaults to embed_dim // upsample_rate (channel halving for r=2).
        _output_dim = (
            output_dim if output_dim is not None else embed_dim // upsample_rate
        )
        self.output_dim = _output_dim

        # depth_to_space with block_size=r divides channels by r².
        # To get output_dim channels out, we need r² * output_dim channels in.
        self.linear_trans = Conv2D(
            upsample_rate * upsample_rate * _output_dim,
            kernel_size=1,
            use_bias=False,
            name="{}_linear_trans".format(self.prefix),
        )

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "Input feature has wrong size"

        x = tf.reshape(x, (-1, H, W, C))
        x = self.linear_trans(x)
        x = tf.nn.depth_to_space(
            x,
            self.upsample_rate,
            data_format="NHWC",
            name="{}_d_to_space".format(self.prefix),
        )

        if self.return_vector:
            new_H = H * self.upsample_rate
            new_W = W * self.upsample_rate
            x = tf.reshape(x, (-1, new_H * new_W, self.output_dim))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patch": self.num_patch,
                "embed_dim": self.embed_dim,
                "upsample_rate": self.upsample_rate,
                "return_vector": self.return_vector,
                "output_dim": self.output_dim,
                "name": self.prefix,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
