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

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(
                f"Index {index} is out of range for {len(self)} batches."
            )

        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.ids))
        batch_ids = self.ids[start_index:end_index]

        if len(batch_ids) == 0:
            raise ValueError(
                f"Batch {index} has no data."
            )

        X, y = self._data_generation(batch_ids)
        return X, y

    def _load_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)

        # Match mask by finding the corresponding file with a known mask extension
        base_name = os.path.splitext(image_id)[0]
        mask_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.isfile(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for image '{image_id}' in {self.mask_dir}"
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

        # Validate and map mask values
        unique_raw = np.unique(mask_arr)
        if len(unique_raw) > self.num_classes:
            raise ValueError(
                f"Number of unique values in mask ({len(unique_raw)}) "
                f"exceeds num_classes ({self.num_classes}). "
                f"Unique values: {unique_raw.tolist()}"
            )
        if len(unique_raw) < self.num_classes:
            warnings.warn(
                f"Number of unique values in mask ({len(unique_raw)}) "
                f"is less than num_classes ({self.num_classes})."
            )

        # Map raw values to sequential class indices (0, 1, 2, ...)
        mapping = {val: idx for idx, val in enumerate(sorted(unique_raw))}
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

        if self.num_classes and self.num_classes > 1:
            y = to_categorical(y, num_classes=self.num_classes)
        return X, y
