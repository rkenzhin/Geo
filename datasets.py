"""
Handles loading large binary geo-datasets efficiently for PyTorch training/evaluation.

Supports loading single images on-the-fly or pre-loading a subset into RAM.
Includes various normalization options.
"""

import os
import sys
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# --- Dataset Configuration ---

DATA_DIR = "/path/to/your/data/please/edit/me"
# Input file is often the same for different output types (Diffraction/Reflection)
DATA_INPUT_BASENAMES = ["data_training_input_ext.bin"]
# Output files correspond to different processing results (D=Diffraction, R=Reflection)
# Use_second_data flag means, that we train for reflection output only
DATA_OUTPUT_BASENAMES = [
    "data_training_output_D_ext.bin",
    "data_training_output_R_ext.bin",
]

NX = 481
NY = 751
NZ = 481 * 6  # Total number of images (slices) in the binary files
# indexes range = 481 * 6 - 281 = 2405, get 5 cubes from 6. 1st one will miss


# Ensure DATA_INPUT_NAME and DATA_OUTPUT_NAME are lists of full paths
# Handle cases where only one input file is defined but multiple outputs exist
if len(DATA_INPUT_BASENAMES) == 1 and len(DATA_OUTPUT_BASENAMES) > 1:
    DATA_INPUT_NAME = [os.path.join(DATA_DIR, DATA_INPUT_BASENAMES[0])] * len(
        DATA_OUTPUT_BASENAMES
    )
else:
    DATA_INPUT_NAME = [os.path.join(DATA_DIR, name) for name in DATA_INPUT_BASENAMES]
DATA_OUTPUT_NAME = [os.path.join(DATA_DIR, name) for name in DATA_OUTPUT_BASENAMES]


# --- Constants ---
DATA_SHAPE: Tuple[int, int] = (NX, NY)
OFFSET_STEP: int = NX * NY * 4  # Step between images in bytes (float32 = 4 bytes)
COUNT_SIZE: int = NX * NY  # Number of float32 values per image
CROP_SIZE: int = 500  # Default crop size if cropping is enabled
EPS: float = 1e-21  # Small epsilon for safe division

# --- Normalization Function Names ---
# Used for validation and dynamic function calling
NORMALIZE_FUNC_NAMES = (
    "normalize_minmax",
    "unnormalize_minmax",
    "normalize_mean_std",
    "normalize_input_to_max1",
    "normalize_output_to_max1",
    "unnormalize_output_to_max1",
)


# --- Data Loading Functions ---


# Cache loaded images to avoid redundant disk reads if the same image is needed again quickly.
@lru_cache(maxsize=128)  # Adjust maxsize based on available RAM and access patterns
def load_one_image(filename: str, offset: int = 0) -> torch.Tensor:
    """
    Loads a single 2D image (slice) from a large binary file.
    Uses numpy.fromfile for efficient reading.

    Args:
        filename: The path to the binary data file.
        offset: The byte offset within the file to start reading the image.

    Returns:
        A torch.Tensor representing the loaded image (shape [NX, NY]).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        with open(filename, "rb") as fp:
            # Ensure count doesn't exceed file size implicitly, handle potential partial reads if needed
            data = np.fromfile(fp, dtype=np.float32, count=COUNT_SIZE, offset=offset)

        if data.size != COUNT_SIZE:
            raise ValueError(
                f"Read {data.size} elements, expected {COUNT_SIZE} for shape {DATA_SHAPE} from {filename} at offset {offset}."
            )

        # Reshape directly into torch tensor for potentially better memory handling
        image_tensor = torch.from_numpy(data).reshape(NX, NY)
        return image_tensor
    except Exception as e:
        raise IOError(f"Error reading image from {filename} at offset {offset}: {e}")


def load_file(filename: str, indexes_range: int) -> torch.Tensor:
    """
    Loads a specified number of images from the beginning of a binary file.
    Warning: This can consume significant RAM if indexes_range is large.
             Consider using GeoDataset with is_load_data_to_RAM=False for large datasets.

    Args:
        filename: Path to the binary data file.
        indexes_range: The number of images (slices) to load from the start.

    Returns:
        A torch.Tensor containing the loaded images (shape [indexes_range, NX, NY]).
    """
    count = COUNT_SIZE * indexes_range
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        with open(filename, "rb") as fp:
            # Consider checking file size vs count request here
            data = np.fromfile(fp, dtype=np.float32, count=count)

        if data.size != count:
            raise ValueError(
                f"Read {data.size} elements, expected {count} for {indexes_range} images from {filename}."
            )

        image_tensor = torch.from_numpy(data).reshape(indexes_range, NX, NY)
        return image_tensor
    except Exception as e:
        raise IOError(f"Error reading {indexes_range} images from {filename}: {e}")


# --- Normalization Functions (Single Image) ---
# These operate on single [C, H, W] or [H, W] tensors, typically used within the Dataset __getitem__.
# The corresponding batch versions are in utils.py for inference.


def normalize_minmax(image: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Normalizes a 2D tensor ([H, W] or [1, H, W]) to the [0, 1] interval."""
    img = image.squeeze()  # Work with 2D tensor
    maximum = img.max()
    minimum = img.min()
    return (img - minimum) / (maximum - minimum + EPS)


def unnormalize_minmax(
    output_img: torch.Tensor, input_img: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Unnormalizes a 2D output tensor using the min/max of the corresponding input tensor."""
    maximum = input_img.max()
    minimum = input_img.min()
    return output_img * (maximum - minimum) + minimum


def normalize_input_to_max1(
    image: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Normalizes a 2D tensor ([H, W] or [1, H, W]) to [-1, 1] by its max absolute value."""
    return image / image.abs().max() + EPS


def normalize_output_to_max1(
    output_img: torch.Tensor, input_img: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Normalizes a 2D output tensor using the max absolute value of the input tensor."""
    return output_img / input_img.abs().max() + EPS


def unnormalize_output_to_max1(
    output_img: torch.Tensor, input_img: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Unnormalizes a 2D output tensor using the max absolute value of the input tensor."""
    return output_img * input_img.abs().max()


def normalize_mean_std(image: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Normalizes a 2D tensor ([H, W] or [1, H, W]) to mean = 0 and std = 1."""
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + EPS)


# --- Augmentation/Noise Functions ---


def add_gaussian_noise(
    image: torch.Tensor, mean: float = 0.0, std: float = 1e-15, p: float = 1.0
) -> torch.Tensor:
    """
    Adds Gaussian noise to tensor with a given probability.

    Args:
        image: Input tensor.
        mean: Mean of the Gaussian noise.
        std: Standard deviation of the Gaussian noise.
        p: Probability of adding noise (0 to 1).

    Returns:
        Tensor with potentially added noise.
    """
    if np.random.rand() < p:  # Check probability first
        noise = torch.randn_like(image) * std + mean
        return image + noise
    return image


# --- Dataset Class ---


class GeoDataset(Dataset):
    """
    PyTorch Dataset for loading large binary geo-datasets.
    Handles on-the-fly loading or pre-loading to RAM, subset selection (train/val),
    and applies normalization and noise augmentation during __getitem__.

    Args:
        data_dir: Directory containing the binary data files.
        data_input_name: List of input data filenames.
        data_output_name: List of output data filenames.
        val_ratio: Fraction of data to reserve for validation. Default is 0.2.
        is_val: If True, use the validation subset; otherwise, use the training subset. Default is False.
        indexes_range: Total number of indices to consider from the dataset. Cut first N numbers from the beginning of data.
                       Usefull for debugging and testing (if we don't want load and train all data). Default is NZ.
        indexes: Specific numpy array of indices to use (overrides val_ratio/is_val splitting).
                 Useful for creating test sets or specific evaluation sets. Default is None.
        no_data_split: If True use all indexes (all data), if False - split indexes for validation. Default is False.
        normalize_input: Name of the normalization function for input images (from NORMALIZE_FUNC_NAMES).
        normalize_output: Name of the normalization function for output/target images.
        add_noise: If True, add Gaussian noise to input images (typically only for training). Default is False.
        use_second_data: If True, use the second pair of input/output files from the lists. Default is False.
        NY_crop: Crop the Y dimension of images to this size (from the top). Default is NY (no crop).
        is_load_data_to_RAM: If True, load all specified data indices into RAM at initialization.
                             Requires significant RAM for large datasets. Default is False.
        is_crop: Performs height (NY) splitting (or cropping) of already loaded (and possibly cropped with NY_crop) images into smaller fragments (patches).
                 This is done only for the training set (is_val=False) and only if the data is loaded into RAM (is_load_data_to_RAM=True).
                 Default is False.
        crop_size: The size to crop along the height (NY) dimension if is_crop is True.
        starting_index : The first index (inclusive) from which to start reading the data.
                         Default is 0 (beginning of the dataset).

        ending_index :  The last index (exclusive) at which to stop reading the data.
                        Default is NZ (end of the dataset).
                        Note: ending_index must be greater than starting_index.
        noise_std: Mean of the Gaussian noise.
        noise_mean: Standard deviation of the Gaussian noise.
        noise_probability: Probability of adding noise (0 to 1).

    """

    def __init__(
        self,
        data_dir: str,
        data_input_name: List[str],
        data_output_name: List[str],
        val_ratio: float = 0.2,
        is_val: bool = False,
        indexes_range: int = NZ,
        indexes: Optional[np.ndarray] = None,
        no_data_split: bool = False,
        normalize_input: Optional[str] = None,
        normalize_output: Optional[str] = None,
        add_noise: bool = False,
        use_second_data: bool = False,
        NY_crop: int = NY,
        is_load_data_to_RAM: bool = False,
        is_crop: bool = False,
        crop_size: int = CROP_SIZE,
        starting_index: int = 0,
        ending_index: int = NZ,
    ):
        self.data_dir = data_dir
        self.is_val = is_val
        self.is_crop = is_crop and not is_val  # Only crop training data
        self.crop_size = crop_size
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.is_load_data_to_RAM = is_load_data_to_RAM
        self.add_noise = add_noise and not is_val  # Only add noise to training data
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noise_probability = noise_probability
        
        # Select input/output filenames
        file_pair_index = 1 if use_second_data else 0
        if file_pair_index >= len(data_input_name) or file_pair_index >= len(
            data_output_name
        ):
            raise ValueError(
                f"Invalid file_pair_index {file_pair_index} for dataset lists."
            )
        self.input_filename = data_input_name[file_pair_index]
        self.output_filename = data_output_name[file_pair_index]

        self.NY_crop = min(NY_crop, NY)  # Ensure crop doesn't exceed original size

        if self.is_crop and not self.is_load_data_to_RAM:
            raise ValueError(
                "Cropping requires loading data to RAM (is_load_data_to_RAM=True)."
            )

        if starting_index >= ending_index:
            raise ValueError(
                f"Index range validation failed. Required: starting_index < ending_index. "
                f"Got: {starting_index} >= {ending_index}"
            )
        # Determine indices to use
        if indexes is not None:
            # Use provided indices directly (e.g., for test or evaluation set)
            # Don't cut it with indexes_range parameter.
            self.indexes = indexes
            indexes_range = len(self.indexes)  # Update range based on provided indices
            print(f"Using provided {indexes_range} indices.")
        elif no_data_split:
            #  Don't split data on train-val set, use all.
            self.indexes = np.arange(starting_index, ending_index)[:indexes_range]
        else:
            # Use train/val split logic
            print(
                f"Performing train/val split on range {indexes_range} with val_ratio {val_ratio}."
            )
            np.random.seed(42)  # Ensure reproducible splits
            all_considered_indices = np.random.permutation(
                np.arange(starting_index, ending_index)
            )[:indexes_range]  # permute indices
            val_len = int(val_ratio * indexes_range)
            train_len = indexes_range - val_len

            if is_val:
                self.indexes = all_considered_indices[train_len:]
                print(f"Using validation set: {len(self.indexes)} indices.")
            else:
                self.indexes = all_considered_indices[:train_len]
                print(f"Using training set: {len(self.indexes)} indices.")

        self.size = len(self.indexes)  # Initial size

        # Pre-load data to RAM if requested
        self.input_data: Optional[torch.Tensor] = None
        self.output_data: Optional[torch.Tensor] = None

        if self.is_load_data_to_RAM:
            print(f"Loading {self.size} images into RAM...")
            self.input_data = torch.zeros(self.size, NX, self.NY_crop)
            self.output_data = torch.zeros(self.size, NX, self.NY_crop)

            for i, idx in enumerate(self.indexes):
                offset = idx * OFFSET_STEP
                try:
                    self.input_data[i] = load_one_image(
                        self.input_filename, offset=offset
                    )[:, : self.NY_crop]
                    self.output_data[i] = load_one_image(
                        self.output_filename, offset=offset
                    )[:, : self.NY_crop]
                except Exception as e:
                    print(f"Error loading image index {idx} (offset {offset}): {e}")
                    # Handle error appropriately (e.g., skip index, raise)
                    raise

            print("Data loaded into RAM.")

            # Apply cropping (only for train split) after loading if enabled for training set.
            if self.is_crop and not self.is_val:
                print(f"Applying cropping with size {self.crop_size}...")

                self.input_data = torch.cat(
                    self.input_data.split(crop_size, dim=2), dim=0
                )
                self.output_data = torch.cat(
                    self.output_data.split(crop_size, dim=2), dim=0
                )

                self.size = self.input_data()[0]  # Update size after cropping

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset (considering cropping)."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample (input image X, target image y) from the dataset.
        Handles on-the-fly loading or retrieval from RAM, applies noise and normalization.
        Args:
            idx: The index of the sample to retrieve.
        Returns:
            A tuple (X, y) where X and y are torch tensors with shape [1, NX, NY_crop_or_crop_size].
        """
        if self.is_load_data_to_RAM:
            X = self.input_data[idx]
            y = self.output_data[idx]
        else:
            # Load data on-the-fly
            if self.is_crop:
                # Calculate original image index and crop offset if cropping is done on-the-fly.
                # This logic requires careful implementation if is_load_data_to_RAM=False and is_crop=True.
                # Currently, is_crop requires is_load_data_to_RAM=True, so this path isn't taken.
                raise NotImplementedError(
                    "On-the-fly cropping is not implemented. Use is_load_data_to_RAM=True for cropping."
                )
            else:
                # Load full (potentially NY_cropped) image
                offset = self.indexes[idx] * OFFSET_STEP

                X = load_one_image(self.input_filename, offset=offset)[
                    :, : self.NY_crop
                ]
                y = load_one_image(self.output_filename, offset=offset)[
                    :, : self.NY_crop
                ]

        # Add channel dimension: [NX, NY_crop] -> [1, NX, NY_crop]
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

        # Apply noise augmentation (only during training)
        if self.add_noise:
            X = add_gaussian_noise(
                X, mean=self.noise_mean, std=self.noise_std, p=self.noise_probability
            )

        # Apply normalization (output first, as it might depend on original input X).
        if self.normalize_output in NORMALIZE_FUNC_NAMES:
            normalize_function = getattr(sys.modules[__name__], self.normalize_output)
            y = normalize_function(y, X)  # Pass original X if needed by function

        if self.normalize_input in NORMALIZE_FUNC_NAMES:
            normalize_function = getattr(sys.modules[__name__], self.normalize_input)
            X = normalize_function(X)

        return X, y
