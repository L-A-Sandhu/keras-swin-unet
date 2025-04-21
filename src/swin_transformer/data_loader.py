# from tensorflow.keras.utils import Sequence
# import numpy as np
# import os
# from PIL import Image
# from tensorflow.keras.utils import to_categorical
# import warnings

# class DynamicDataLoader(Sequence):
#     def __init__(self, data_dir, ids, batch_size=32, img_size=(256, 256), mode='train',
#                  image_dtype=np.float32, mask_dtype=np.int32, num_classes=None, input_scale=65536, mask_scale=65536):
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.mode = mode
#         self.image_dir = os.path.join(data_dir, 'images')
#         self.mask_dir = os.path.join(data_dir, 'masks')
#         self.ids = ids
#         self.image_dtype = image_dtype
#         self.mask_dtype = mask_dtype
#         self.num_classes = num_classes
#         self.input_scale = input_scale
#         self.mask_scale = mask_scale

#         self.image_ext = self._determine_extension(self.image_dir)
#         self.mask_ext = self._determine_extension(self.mask_dir)

#     def _determine_extension(self, directory):
#         for filename in os.listdir(directory):
#             return os.path.splitext(filename)[1]
#         raise FileNotFoundError(f"No files found in {directory}")

#     def __len__(self):
#         return int(np.ceil(len(self.ids) / float(self.batch_size)))

#     def __getitem__(self, index):
#         if index >= len(self):
#             raise IndexError(f"Index {index} is out of range for {len(self)} batches.")

#         start_index = index * self.batch_size
#         end_index = min(start_index + self.batch_size, len(self.ids))

#         batch_ids = self.ids[start_index:end_index]

#         if len(batch_ids) == 0:
#             raise ValueError(f"Batch {index} has no data. This should not happen with correct batch calculation.")

#         X, y = self._data_generation(batch_ids)
#         # print(f"Batch {index} - X min: {X.min()}, X max: {X.max()}, y min: {y.min()}, y max: {y.max()}")
#         return X, y

#     def _load_image(self, image_id):
#         image_path = os.path.join(self.image_dir, image_id)
#         mask_path = os.path.join(self.mask_dir, image_id.replace(self.image_ext, self.mask_ext))

#         if image_path.endswith('.npy'):
#             image = np.load(image_path)
#         else:
#             image = Image.open(image_path).resize(self.img_size)
#             if image.mode == 'L':
#                 image = np.stack((np.array(image),) * 3, axis=-1)
#             image = np.array(image, dtype=self.image_dtype)
#         image = image.astype(self.image_dtype) / self.input_scale

#         mask = Image.open(mask_path).resize(self.img_size, resample=Image.NEAREST).convert('L')
#         mask = np.array(mask, dtype=self.mask_dtype)
#         mask = (mask / self.mask_scale).astype(int)

#         unique_values_image = np.unique(image)
#         unique_values_mask = np.unique(mask)

#         if len(unique_values_mask) > self.num_classes:
#             raise ValueError(f"Number of unique values in mask ({len(unique_values_mask)}) exceeds the number of classes ({self.num_classes}). Please check the mask or the num_classes argument.")
#         elif len(unique_values_mask) < self.num_classes:
#             warnings.warn(f"Number of unique values in mask ({len(unique_values_mask)}) is less than the number of classes ({self.num_classes}). Proceeding with training.")

#         return image, mask

#     def _data_generation(self, batch_ids):
#         X = np.empty((len(batch_ids), *self.img_size, 3), dtype=self.image_dtype)
#         y = np.empty((len(batch_ids), *self.img_size), dtype=self.mask_dtype)

#         for i, ID in enumerate(batch_ids):
#             image, mask = self._load_image(ID)
#             if image.ndim == 2:
#                 image = np.stack((image,) * 3, axis=-1)
#             X[i,] = image
#             y[i,] = mask

#         y = to_categorical(y, num_classes=self.num_classes) if self.num_classes > 1 else y
#         return X, y
import os
import warnings
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence, to_categorical


class DynamicDataLoader(Sequence):
    def __init__(
        self,
        data_dir,
        ids,
        batch_size=2,
        img_size=(256, 256),
        mode="train",
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=None,
        input_scale=65536,
        mask_scale=65536,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.ids = ids
        self.image_dtype = image_dtype
        self.mask_dtype = mask_dtype
        self.num_classes = num_classes
        self.input_scale = input_scale
        self.mask_scale = mask_scale

        self.image_ext = self._determine_extension(self.image_dir)
        self.mask_ext = self._determine_extension(self.mask_dir)

    def _determine_extension(self, directory):
        for filename in os.listdir(directory):
            return os.path.splitext(filename)[1]
        raise FileNotFoundError(f"No files found in {directory}")

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} is out of range for {len(self)} batches.")

        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.ids))
        batch_ids = self.ids[start_index:end_index]

        if len(batch_ids) == 0:
            raise ValueError(
                f"Batch {index} has no data. This should not happen with correct batch calculation."
            )

        X, y = self._data_generation(batch_ids)
        return X, y

    def _load_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        mask_path = os.path.join(
            self.mask_dir, image_id.replace(self.image_ext, self.mask_ext)
        )

        # Load image
        if image_path.endswith(".npy"):
            image = np.load(image_path)
        else:
            image = Image.open(image_path).resize(self.img_size[:2])

            if image.mode == "L":
                image = np.stack((np.array(image),) * 3, axis=-1)
            image = np.array(image, dtype=self.image_dtype)
        image = image.astype(self.image_dtype) / self.input_scale

        # Load mask
        mask_img = (
            Image.open(mask_path)
            .resize(self.img_size[:2], resample=Image.NEAREST)
            .convert("L")
        )
        mask_arr = np.array(mask_img, dtype=self.mask_dtype)

        # Determine raw unique values and map each to a class index
        unique_raw = np.unique(mask_arr)
        # Error if more raw classes than expected
        if len(unique_raw) > self.num_classes:
            raise ValueError(
                f"Number of unique values in mask ({len(unique_raw)}) exceeds the number of classes ({self.num_classes})."
            )
        # Warn if fewer raw classes
        if len(unique_raw) < self.num_classes:
            warnings.warn(
                f"Number of unique values in mask ({len(unique_raw)}) is less than the number of classes ({self.num_classes}). Proceeding with training."
            )
        # Create mapping from raw value to 0..len(unique_raw)-1
        mapping = {val: idx for idx, val in enumerate(unique_raw)}
        # Map mask to indices
        mask_indices = np.vectorize(lambda v: mapping[v])(mask_arr)

        return image, mask_indices

    def _data_generation(self, batch_ids):
        if len(self.img_size) == 2:
            X = np.empty((len(batch_ids), *self.img_size, 3), dtype=self.image_dtype)
        else:
            X = np.empty((len(batch_ids), *self.img_size), dtype=self.image_dtype)

        y = np.empty((len(batch_ids), *self.img_size[:2]), dtype=self.mask_dtype)
        for i, ID in enumerate(batch_ids):
            image, mask = self._load_image(ID)
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            X[i,] = image
            y[i,] = mask

        # One-hot encode if more than one class
        y = (
            to_categorical(y, num_classes=self.num_classes)
            if self.num_classes and self.num_classes > 1
            else y
        )
        return X, y
