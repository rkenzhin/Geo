import json
import os
import sys
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from datasets import NX, NY, GeoDataset
from utils import predict_batch

# Constants
PREDICTION_SUFFIX = "prediction_of_{}_{}images.bin"
INPUT_SUFFIX = "input_of_{}_{}images.bin"
GT_SUFFIX = "model_output_of_{}_{}images.bin"
INDEX_FILE = "train_val_test_indexes.txt"

# --- Paths (MODIFY THESE) ---
DATA_DIR_TRAIN = "/path/data"
DATA_DIR_REAL = "/path/data"

DATA_INPUT_BASENAMES = ["data_training_input_ext.bin"]
# Output files correspond to different processing results (D=Diffraction, R=Reflection)
# Use_second_data flag means, that we train for reflection output only
DATA_OUTPUT_BASENAMES = [
    "data_training_output_D_ext.bin",
    "data_training_output_R_ext.bin",
]


def inference_with_groundtruth(
    model_dir: str,
    data_dir: str,
    normalize_input: Optional[str] = None,
    unnorm_output: Optional[str] = None,
    indexes_range: Optional[int] = 100,
    mode_str: Literal["val", "test"] = "val",
    use_second_data: bool = False,
    NY_crop: int = NY,  # if NY than we use all NY size of img (no CROP)
    save_bin: bool = False,
    batch_size: int = 12,  # Added batch size parameter
    load_to_ram: bool = False,  # Control RAM loading for dataset
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Performs inference on a validation or test set, compares with ground truth,
    calculates MSE loss metrics, and optionally saves results. Uses batch processing.

    Args:
        model_dir: Path to the saved PyTorch model (.pth file).
        data_dir: Directory containing the binary data files.
        normalize_input: Name of the input normalization function (from utils.py).
        unnorm_output: Name of the output unnormalization function (from utils.py).
        indexes_range: Number of images to evaluate from the specified set. If None, use all.
        mode_str: Which set to evaluate ('val' or 'test').
        use_second_data: Flag to use the second dataset file pair (defined in datasets.py).
        NY_crop: Vertical dimension cropping for images.
        save_bin: Whether to save input, ground truth, and prediction tensors to binary files.
        batch_size: Number of images to process in each batch.
        load_to_ram: Whether the dataset should load all data into RAM (faster but uses more memory).

    Returns:
        A tuple containing:
        - input_tensor: Tensor of input images used ([N, NX, NY_crop]).
        - model_tensor: Tensor of ground truth images ([N, NX, NY_crop]).
        - result_tensor: Tensor of predicted images ([N, NX, NY_crop]).
        - loss_dict: Dictionary containing average, min, and max MSE loss and corresponding indices.
    """
    model_parent_dir = os.path.dirname(model_dir)
    index_file_path = os.path.join(model_parent_dir, INDEX_FILE)

    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Index file not found: {index_file_path}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model file not found: {model_dir}")

    with open(index_file_path) as f:
        indexes_data = json.load(f)
        eval_indices = np.array(indexes_data[f"{mode_str}_indexes"])

    if indexes_range is not None and indexes_range < len(eval_indices):
        # Select a subset if requested, ensure reproducibility if needed by fixing seed elsewhere
        # For simplicity, just taking the first 'indexes_range' indices
        eval_indices = eval_indices[:indexes_range]
    elif indexes_range is None:
        indexes_range = len(eval_indices)  # Use all indices if None

    print(f"Evaluating {indexes_range} images from '{mode_str}' set.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_dir, map_location=device)
    model.eval()

    # --- Dataset and DataLoader Setup ---

    data_input_paths = [os.path.join(data_dir, DATA_INPUT_BASENAMES[0])] * len(
        DATA_OUTPUT_BASENAMES
    )
    data_output_paths = [os.path.join(data_dir, name) for name in DATA_OUTPUT_BASENAMES]

    # Create a GeoDataset instance specifically for evaluation
    # Note: Normalization/unnormalization happens *outside* the dataset during batch prediction
    eval_dataset = GeoDataset(
        data_dir,
        data_input_paths,
        data_output_paths,
        indexes=eval_indices,  # Pass the specific indices
        is_val=True,  # Treat like validation data (no shuffling, no train-specific aug)
        use_second_data=use_second_data,
        NY_crop=NY_crop,
        is_load_data_to_RAM=load_to_ram,
        # Pass None for normalization here, handled in predict_batch
        normalize_input=None,
        normalize_output=None,
        add_noise=False,  # Typically no noise for evaluation
        is_crop=False,  # Typically no cropping for evaluation unless intended
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,  #  No shuffling for evaluation/testing
        num_workers=4 if torch.cuda.is_available() else 0,  # Adjust based on system
        pin_memory=True if torch.cuda.is_available() else False,
    )

    mse_loss_fn = torch.nn.MSELoss()

    # Initialize tensors to store results
    result_tensor = torch.zeros(indexes_range, NX, NY_crop)
    input_tensor = torch.zeros(indexes_range, NX, NY_crop)
    model_tensor = torch.zeros(indexes_range, NX, NY_crop)

    total_loss = 0.0
    min_loss = float("inf")
    min_idx = -1
    max_loss = -float("inf")
    max_idx = -1
    current_idx = 0

    print(f"Starting inference on {device} with batch size {batch_size}...")
    for input_batch_ds, output_batch_ds in tqdm(eval_dataloader):
        # input_batch_ds, output_batch_ds are [B, 1, NX, NY_crop] from DataLoader

        # Ensure tensors are float32 (model expects this)
        input_batch_ds = input_batch_ds.float()
        output_batch_ds = output_batch_ds.float()

        # Predict batch
        prediction_batch = predict_batch(
            model, input_batch_ds, normalize_input, unnorm_output, device
        )  # Returns [B, 1, NX, NY_crop] on CPU

        # Calculate loss for the batch (ensure ground truth is on the same device as prediction if calculating loss on GPU)
        # For simplicity, loss calculated on CPU after prediction is moved to CPU
        batch_mse = mse_loss_fn(prediction_batch, output_batch_ds).item()
        total_loss += batch_mse * input_batch_ds.size(0)  # Accumulate total loss

        # Find min/max loss within the *batch* and update overall min/max
        # Note: MSELoss reduces the batch already, so batch_mse is a single value
        # To find per-image loss, calculate MSE individually if needed:
        individual_losses = torch.mean(
            (prediction_batch - output_batch_ds) ** 2, dim=(1, 2, 3)
        )
        batch_min_loss, batch_min_indices = torch.min(individual_losses, dim=0)
        batch_max_loss, batch_max_indices = torch.max(individual_losses, dim=0)

        batch_min_loss_val = batch_min_loss.item()
        batch_max_loss_val = batch_max_loss.item()
        batch_min_idx_in_batch = batch_min_indices.item()
        batch_max_idx_in_batch = batch_max_indices.item()

        if batch_min_loss_val < min_loss:
            min_loss = batch_min_loss_val
            min_idx = current_idx + batch_min_idx_in_batch
        if batch_max_loss_val > max_loss:
            max_loss = batch_max_loss_val
            max_idx = current_idx + batch_max_idx_in_batch

        # Store results
        start = current_idx
        end = current_idx + prediction_batch.size(0)
        result_tensor[start:end] = prediction_batch.squeeze(1)  # Remove channel dim
        input_tensor[start:end] = input_batch_ds.squeeze(1)
        model_tensor[start:end] = output_batch_ds.squeeze(1)

        current_idx += prediction_batch.size(0)

    avg_loss = total_loss / indexes_range if indexes_range > 0 else 0.0

    loss_dict = {
        "avg_loss": avg_loss,
        "min_loss": min_loss if min_idx != -1 else None,
        "min_idx": min_idx if min_idx != -1 else None,  # Index within the evaluated set
        "max_loss": max_loss if max_idx != -1 else None,
        "max_idx": max_idx if max_idx != -1 else None,  # Index within the evaluated set
    }

    if save_bin:
        dirname = os.path.dirname(model_dir)
        pred_filename = os.path.join(
            dirname, PREDICTION_SUFFIX.format(mode_str, indexes_range)
        )
        in_filename = os.path.join(
            dirname, INPUT_SUFFIX.format(mode_str, indexes_range)
        )
        gt_filename = os.path.join(dirname, GT_SUFFIX.format(mode_str, indexes_range))

        print(f"Saving predictions to {pred_filename}")
        result_tensor.numpy().astype(np.float32).tofile(pred_filename)
        print(f"Saving inputs to {in_filename}")
        input_tensor.numpy().astype(np.float32).tofile(in_filename)
        print(f"Saving ground truth to {gt_filename}")
        model_tensor.numpy().astype(np.float32).tofile(gt_filename)

    return input_tensor, model_tensor, result_tensor, loss_dict


if __name__ == "__main__":
    # Ensure the model path is correct relative to where you run the script
    # Example using the latest model path from the original file:
    model_path = "models/name/_best.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        print("Please update the 'model_path' variable in the script.", file=sys.stderr)
        sys.exit(1)

    print(f"Using model: {model_path}")

    # Define evaluation parameters
    eval_norm_input = "normalize_input_to_max1"
    eval_unnorm_output = "unnormalize_output_to_max1"
    eval_range = 100  # Number of validation images to test
    eval_mode = "val"
    eval_use_second = (
        False  # Use first dataset pair (diffraction output for Feb 2025 data)
    )
    eval_ny_crop = datasets.NY  # Use full height unless specified
    eval_save = True
    eval_batch_size = 12  # Adjust based on GPU memory
    eval_load_ram = True  # Set to True if RAM allows and you want speed

    print("\nStarting evaluation...")
    try:
        input_data, gt_data, pred_data, losses = inference_with_groundtruth(
            model_dir=model_path,
            data_dir=DATA_DIR_TRAIN,
            normalize_input=eval_norm_input,
            unnorm_output=eval_unnorm_output,
            indexes_range=eval_range,
            mode_str=eval_mode,
            use_second_data=eval_use_second,
            NY_crop=eval_ny_crop,
            save_bin=eval_save,
            batch_size=eval_batch_size,
            load_to_ram=eval_load_ram,
        )

        print("\nEvaluation finished.")
        print("Loss Metrics:")
        for key, value in losses.items():
            if isinstance(value, (float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print("\nOutput tensor shapes:")
        print(f"  Input:        {input_data.shape}")
        print(f"  Ground Truth: {gt_data.shape}")
        print(f"  Prediction:   {pred_data.shape}")

    except FileNotFoundError as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)

