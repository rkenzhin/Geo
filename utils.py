"""
Normalization utility functions designed for BATCH processing during inference.
These functions mirror the logic in datasets.py but operate on tensors
with a batch dimension ([B, C, H, W]) without needing access to dataset stats.

Also helping fucntions for other modules (e.g. predict_image for inference)
"""

import sys
from typing import Optional

import torch

import datasets

# Tuple of recognized normalization/unnormalization function names
# Used for validation in other modules (e.g., evaluation.py)
NORMALIZE_FUNC_NAMES = (
    "normalize_minmax",
    "unnormalize_minmax",
    "normalize_mean_std",
    "normalize_input_to_max1",
    "normalize_output_to_max1",
    "unnormalize_output_to_max1",
)

# Small epsilon to prevent division by zero
EPS = 1e-21


def _check_and_expand_dims(image: torch.Tensor) -> torch.Tensor:
    """
    Ensures the input tensor has batch and channel dimensions [B, C, H, W].

    - If input is [H, W], expands to [1, 1, H, W].
    - If input is [C, H, W] (C must be 1), expands to [1, 1, H, W].
    - If input is [B, H, W] (implies C=1), expands to [B, 1, H, W].
    - If input is [B, C, H, W] (C must be 1), returns unchanged.

    Raises ValueError for invalid input shapes.

    Args:
        image: Input tensor.

    Returns:
        Tensor reshaped to [B, 1, H, W].
    """
    if image.dim() == 2:  # Single image (H, W)
        return image.unsqueeze(0).unsqueeze(0)  # Expand to [1, 1, H, W]
    elif image.dim() == 3:  # Single image with channel (C, H, W)
        if image.size(0) == 1:
            return image.unsqueeze(0)  # Expand C=1 to [1, 1, H, W]
        else:
            raise ValueError(
                "Input tensor has 3 dims but channel != 1. Expected [1, H, W] or [H, W]."
            )
    elif image.dim() == 4:  # Batch of images
        if image.size(1) == 1:
            return image  # Already [B, 1, H, W]
        else:
            raise ValueError(
                "Input tensor has 4 dims but channel != 1. Expected [B, 1, H, W]."
            )
    else:
        raise ValueError(
            f"Input tensor has unsupported number of dimensions: {image.dim()}. Expected 2, 3 or 4."
        )


def normalize_minmax(image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Normalizes a batch of images to the [0, 1] interval using min-max scaling.
    Each image in the batch is normalized independently based on its own min/max.

    Args:
        image: Input tensor batch (shape [B, 1, H, W] or compatible).

    Returns:
        Normalized tensor batch (shape [B, 1, H, W]).
    """
    image = _check_and_expand_dims(image)
    batch_size = image.size(0)

    # Calculate min/max per image in the batch
    view_shape = (batch_size, -1)
    minimum = image.view(view_shape).min(dim=1).values  # Shape [B,]
    maximum = image.view(view_shape).max(dim=1).values

    # Reshape min/max for broadcasting: [B, 1, 1, 1]
    minimum = minimum[:, None, None, None]
    maximum = maximum[:, None, None, None]

    normalized_images = (image - minimum) / (maximum - minimum + EPS)
    return normalized_images


def unnormalize_minmax(
    output_img: torch.Tensor, input_img: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Unnormalizes a batch of output images using min-max scaling derived from the input batch.
    Each output image is unnormalized independently using the min/max of the corresponding input image.

    Args:
        output_img: The normalized output tensor batch (shape [B, 1, H, W] or compatible).
        input_img: The original input tensor batch (shape [B, 1, H, W] or compatible),
                   used to get the min/max for scaling.

    Returns:
        Unnormalized tensor batch (shape [B, 1, H, W]).
    """
    input_img = _check_and_expand_dims(input_img)
    output_img = _check_and_expand_dims(output_img)
    batch_size = input_img.size(0)

    # Calculate min/max per image in the INPUT batch
    view_shape = (batch_size, -1)
    minimum = input_img.view(view_shape).min(dim=1).values  # Shape [B,]
    maximum = input_img.view(view_shape).max(dim=1).values

    # Reshape for broadcasting: [B, 1, 1, 1]
    minimum = minimum[:, None, None, None]
    maximum = maximum[:, None, None, None]

    unnormalized_output = output_img * (maximum - minimum) + minimum
    return unnormalized_output


def normalize_input_to_max1(image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Normalizes a batch of images to the [-1, 1] interval by dividing by the maximum absolute value.
    Each image in the batch is normalized independently based on its own max absolute value.

    Args:
        image: Input tensor batch (shape [B, 1, H, W] or compatible).

    Returns:
        Normalized tensor batch (shape [B, 1, H, W]).
    """
    image = _check_and_expand_dims(image)
    batch_size = image.size(0)

    # Calculate max absolute value per image
    view_shape = (batch_size, -1)
    # Add epsilon to max_abs to prevent division by zero if max_abs is 0
    max_abs = image.view(view_shape).abs().max(dim=1).values  # Shape [B,]

    # Reshape for broadcasting
    max_abs = max_abs[:, None, None, None]

    # Normalize: image / max_abs
    normalized_image = image / (max_abs + EPS)
    return normalized_image


def normalize_output_to_max1(
    output_img: torch.Tensor, input_img: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Normalizes a batch of output images using the maximum absolute value derived from the input batch.
    This is typically used when the *target* data needs normalization based on the *input* scale,
    similar to `normalize_input_to_max1`. Each output image is normalized using the max absolute
    value of the corresponding input image.

    Args:
        output_img: The output tensor batch to normalize (shape [B, 1, H, W] or compatible).
        input_img: The original input tensor batch (shape [B, 1, H, W] or compatible),
                   used to get the max absolute value for scaling.

    Returns:
        Normalized output tensor batch (shape [B, 1, H, W]).
    """
    input_img = _check_and_expand_dims(input_img)
    output_img = _check_and_expand_dims(output_img)
    batch_size = input_img.size(0)

    # Calculate max absolute value per image in the INPUT batch
    view_shape = (batch_size, -1)
    max_abs = input_img.view(view_shape).abs().max(dim=1).values  # Shape [B]

    # Reshape for broadcasting
    max_abs = max_abs[:, None, None, None]

    normalized_output = output_img / (max_abs + EPS)
    return normalized_output


def unnormalize_output_to_max1(
    output_img: torch.Tensor, input_img: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Unnormalizes a batch of output images using the maximum absolute value derived from the input batch.

    This reverses the effect of `normalize_output_to_max1` or standardizes the output scale based on input.
    Each output image is unnormalized using the max absolute value of the corresponding input image.

    Args:
        output_img: The normalized output tensor batch (shape [B, 1, H, W] or compatible).
        input_img: The original input tensor batch (shape [B, 1, H, W] or compatible),
                   used to get the max absolute value for scaling.

    Returns:
        Unnormalized output tensor batch (shape [B, 1, H, W]).
    """
    input_img = _check_and_expand_dims(input_img)
    output_img = _check_and_expand_dims(output_img)
    batch_size = input_img.size(0)

    # Calculate max absolute value per image in the INPUT batch
    view_shape = (batch_size, -1)
    max_abs = input_img.view(view_shape).abs().max(dim=1).values  # Shape [B,]

    # Reshape for broadcasting
    max_abs = max_abs[:, None, None, None]

    unnormalized_output = output_img * max_abs
    return unnormalized_output


def normalize_mean_std(image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Normalizes a batch of images to have zero mean and unit standard deviation (z-score normalization).
    Each image in the batch is normalized independently based on its own mean/std.

    Args:
        image: Input tensor batch (shape [B, 1, H, W] or compatible).

    Returns:
        Normalized tensor batch (shape [B, 1, H, W]).
    """
    image = _check_and_expand_dims(image)
    batch_size = image.size(0)

    # Calculate mean/std per image
    view_shape = (batch_size, -1)
    mean = image.view(view_shape).mean(dim=1)  # Shape [B]
    std = image.view(view_shape).std(dim=1)

    # Reshape for broadcasting
    mean = mean[:, None, None, None]
    std = std[:, None, None, None]

    normalized_image = (image - mean) / (std + EPS)
    return normalized_image


def predict_image(
    model: torch.nn.Module,
    input_img: torch.Tensor,
    normalize_input: Optional[str] = None,
    unnorm_output: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Performs inference on a single (batch=1) image using the provided model.

    Args:
        model: The trained PyTorch model.
        input_img: The input image tensor (usually shape [1, C=1, H, W] or [C=1, H, W]).
        normalize_input: Name of the normalization function from datasets.py to apply to the input.
        unnorm_output: Name of the unnormalization function from datasets.py to apply to the output.
        device: The device (CPU or CUDA) to run inference on. Detects automatically if None.

    Returns:
        The predicted output image tensor (usually shape [1, C, H, W]) detached on cpu.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if input_img.dim() == 3:  # Add batch dimension if missing
        input_img = input_img.unsqueeze(0)

    input_img = input_img.to(device)

    with torch.no_grad():
        normalized_input = input_img
        if normalize_input and normalize_input in datasets.NORMALIZE_FUNC_NAMES:
            # Use the single-image version from datasets for consistency if needed
            # Note: predict_batch uses batch-aware utils versions
            normalize_function = getattr(sys.modules["datasets"], normalize_input)
            # Assuming single image normalization takes (C, H, W)
            normalized_input = normalize_function(input_img.squeeze(0)).unsqueeze(0)
        elif normalize_input:
            print(
                f"Warning: Normalization function '{normalize_input}' not found in datasets.py. Skipping.",
                file=sys.stderr,
            )

        output = model(normalized_input)
        if unnorm_output and unnorm_output in datasets.NORMALIZE_FUNC_NAMES:
            unnormalize_function = getattr(sys.modules["datasets"], unnorm_output)
            # Assuming single image unnormalization needs original input (C,H,W) for scaling
            output = unnormalize_function(output, input_img.squeeze(0))
        elif unnorm_output:
            print(
                f"Warning: Unnormalization function '{unnorm_output}' not found in datasets.py. Skipping.",
                file=sys.stderr,
            )

    return output.detach().cpu()


def predict_batch(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    normalize_input: Optional[str] = None,
    unnorm_output: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Performs inference on a batch of images using the provided model.

    Uses batch-aware normalization/unnormalization from utils.py.

    Args:
        model: The trained PyTorch model.
        input_batch: The input batch tensor (shape [B, C, H, W]).
        normalize_input: Name of the normalization function from utils.py to apply to the input batch.
        unnorm_output: Name of the unnormalization function from utils.py to apply to the output batch.
        device: The device (CPU or CUDA) to run inference on. Detects automatically if None.

    Returns:
        The predicted output batch tensor (shape [B, C, H, W]) detached on cpu.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    input_batch = input_batch.to(device)

    with torch.no_grad():
        normalized_input_batch = input_batch
        if normalize_input and normalize_input in NORMALIZE_FUNC_NAMES:
            normalize_function = getattr(sys.modules["utils"], normalize_input)
            normalized_input_batch = normalize_function(input_batch)
        elif normalize_input:
            print(
                f"Warning: Normalization function '{normalize_input}' not found in utils.py. Skipping.",
                file=sys.stderr,
            )

        output_batch = model(normalized_input_batch)

        if unnorm_output and unnorm_output in NORMALIZE_FUNC_NAMES:
            unnormalize_function = getattr(sys.modules["utils"], unnorm_output)
            output_batch = unnormalize_function(
                output_batch, input_batch
            )  # Pass original input_batch for scaling
        elif unnorm_output:
            print(
                f"Warning: Unnormalization function '{unnorm_output}' not found in utils.py. Skipping.",
                file=sys.stderr,
            )

    return output_batch.detach().cpu()


def transform_img_to_2d(input_img: torch.Tensor):
    """
    Transform torch.tensor image from shape [1, C=1, H, W] or shape [C=1, H, W])
    to shape [H, W] required for matplotlib show.

    Args:
        input_img: The input image tensor (shape [1, C=1, H, W] or [C=1, H, W]).

    Returns:
        Torch tensor [H, W] shape.
    """
    if input_img.dim() not in (2, 3, 4):
        raise ValueError("Input image dim should be in (2, 3, 4).")

    if input_img.dim() == 3:
        input_img = input_img.squeeze(0)
    elif input_img.dim() == 4:
        input_img = input_img.squeeze(0).squeeze(0)

    return input_img
