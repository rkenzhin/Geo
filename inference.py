import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import predict_batch


class InferenceDataset(Dataset):
    """
    Simple PyTorch Dataset for inference on a pre-loaded tensor.
    Assumes input tensor shape [N, H, W] where N is the number of images.
    Adds the channel dimension [1, H, W] in __getitem__.
    """

    def __init__(self, input_imgs: torch.Tensor):
        if input_imgs.dim() != 3:
            raise ValueError(
                f"Input tensor must have 3 dimensions ([N, H, W]), got {input_imgs.dim()}"
            )
        self.input_imgs = input_imgs

    def __len__(self):
        return self.input_imgs.size(0)

    def __getitem__(self, idx):
        # Returns shape [1, H, W]
        return self.input_imgs[idx].unsqueeze(0).float()  # Ensure float


def load_real_binary_file(filename: str, nx: int, ny: int, nz: int) -> torch.Tensor:
    """Loads a binary file containing a stack of real images."""
    try:
        with open(filename, "rb") as fp:
            data = np.fromfile(fp, dtype=np.float32)
        expected_elements = nz * nx * ny
        if data.size != expected_elements:
            warnings.warn(
                f"Warning: Read {data.size} elements, expected {expected_elements} from {filename}"
            )
            # Attempt to reshape anyway, might raise error if size mismatch is too large
        return torch.from_numpy(data).reshape(nz, nx, ny)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filename}")
        raise
    except Exception as e:
        print(f"Error loading or reshaping data from {filename}: {e}")
        raise


def save_tensor_to_binary(tensor_data: torch.Tensor, filename: str):
    """Saves a torch tensor to a binary file as float32."""
    try:
        # Ensure tensor is on CPU and contiguous before converting to numpy
        tensor_data.cpu().contiguous().numpy().astype(np.float32).tofile(filename)
        print(f"Successfully saved tensor to {filename}")
    except Exception as e:
        print(f"Error saving tensor to {filename}: {e}")
        raise


def run_batch_inference(
    model: torch.nn.Module,
    input_imgs: torch.Tensor,
    batch_size: int,
    normalize_input: str,
    unnorm_output: str,
    device: torch.device,
) -> torch.Tensor:
    """Runs batch inference on the input tensor stack."""

    dataset = InferenceDataset(input_imgs)
    # Use num_workers=0 for simplicity in notebook, adjust if needed
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Initialize result tensor on CPU first
    result = torch.zeros_like(input_imgs)

    current_idx = 0
    model.to(device)
    model.eval()

    print(f"Starting batch inference with batch size {batch_size}...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # batch shape is [B, 1, H, W]
            batch_on_device = batch.to(device)

            # Use the predict_batch function from utils
            predictions_batch = predict_batch(
                model,
                batch_on_device,  # Pass batch already on device
                normalize_input,
                unnorm_output,
                device,  # Pass device explicitly
            )  # Returns predictions on CPU

            batch_len = predictions_batch.size(0)
            # Store result (predictions are [B, 1, H, W], store as [B, H, W])
            result[current_idx : current_idx + batch_len] = predictions_batch.squeeze(1)

            current_idx += batch_len

    print("Inference complete.")
    return result
