import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from utils import predict_image, transform_img_to_2d


def plot_history_simple(
    history: List[Dict[str, Optional[float]]], filename: Optional[str] = None
):
    """
    Plots the training and validation loss history.

    Args:
        history: A list of dictionaries, where each dict contains
                 'train_loss' (float) and 'val_loss' (float or None).
        filename: If provided, saves the plot to this file path.
                  Otherwise, displays the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    epochs_train = list(range(1, len(history) + 1))
    epochs_val = []
    train_loss = []
    val_loss = []

    for epoch_ndx, epoch in enumerate(history, 1):
        train_loss.append(epoch["train_loss"])
        if epoch["val_loss"] is not None:
            epochs_val.append(epoch_ndx)
            val_loss.append(epoch["val_loss"])

    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(epochs_train, train_loss, label="Train Loss", marker="o", linestyle="-")
    if epochs_val:  # Only plot if validation loss exists
        ax.plot(
            epochs_val, val_loss, label="Validation Loss", marker="x", linestyle="--"
        )

    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    if filename is None:
        plt.show()
    else:
        try:
            fig.savefig(filename, bbox_inches="tight")
            print(f"Saved training history plot to {filename}")
            plt.close(fig)  # Close the figure after saving to free memory
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}", file=sys.stderr)


def show_images(
    *images: torch.Tensor, titles: Optional[List[str]] = None, colorbar: bool = True
):
    """
    Displays multiple images side-by-side using matplotlib.

    Args:
        *images: A variable number of image tensors. Expected shape [C=1, H, W] or [H, W].
        titles: An optional list of titles corresponding to the images.
        colorbar: Show colorbar. Default is True.
    """
    num_images = len(images)
    if num_images == 0:
        print("No images provided to show.")
        return

    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(5 * num_images, 6))
    fig.tight_layout()

    if num_images == 1:
        axes = [axes]  # Make it iterable

    for i, img in enumerate(images):
        # Squeeze dimensions to get [H, W] for imshow
        img_to_show = img.squeeze().cpu().numpy()
        im = axes[
            i
        ].imshow(
            img_to_show.T,  # Transpose assuming original is [W, H] or similar from dataset
            interpolation="nearest",
            aspect="auto",
        )
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    if colorbar:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.04)
    plt.show()


def show_inference(
    input_img: torch.Tensor,
    model: torch.nn.Module,
    title_suffix: str = "",
    normalize_input: Optional[str] = None,
    unnorm_output: Optional[str] = None,
    filename: Optional[str] = None,
    device: Optional[torch.device] = None,
    colorbar: bool = True,
) -> torch.Tensor:
    """
    Shows single input image and its corresponding model prediction side-by-side.

    Args:
        input_img: The input image tensor (shape [1, C=1, H, W] or [C=1, H, W]).
        model: The trained PyTorch model.
        title_suffix: Optional string to append to plot titles (e.g., image index).
        normalize_input: Name of input normalization function (from datasets/utils).
        unnorm_output: Name of output unnormalization function (from datasets/utils).
        filename: If provided, saves the plot to this file. Otherwise, displays it.
        device: Device to run inference on. Auto-detected if None.
        colorbar: Show colorbar. Default is True.

    Returns:
        The predicted output tensor [C, H, W] shape.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use the refactored predict_image function
    predicted_output = predict_image(
        model, input_img, normalize_input, unnorm_output, device
    )  # Returns tensor [C, H, W] on CPU

    # --- Plotting ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    fig.tight_layout(pad=3.0)

    # Transform to matplotlib format 2d images (H, W)
    input_img = transform_img_to_2d(input_img)

    # Input image
    img_in_np = input_img.cpu().numpy()
    axes[0].imshow(img_in_np.T, interpolation="nearest", aspect="auto")
    axes[0].set_title(f"Input Image {title_suffix}")

    # Predicted image
    img_out_np = (
        predicted_output.squeeze().squeeze().cpu().numpy()
    )  # Transform to matplotlib format 2d images (H, W)
    im2 = axes[1].imshow(img_out_np.T, interpolation="nearest", aspect="auto")
    axes[1].set_title(f"Predicted Image {title_suffix}")

    if colorbar:
        fig.colorbar(im2, ax=axes.tolist(), fraction=0.046, pad=0.04)

    if filename is None:
        plt.show()
    else:
        try:
            fig.savefig(filename, bbox_inches="tight")
            print(f"Saved inference plot to {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}", file=sys.stderr)

    return predicted_output.squeeze()


def show_inference_with_groundtruth(
    input_img: torch.Tensor,
    model_img: torch.Tensor,  # Ground truth
    model: torch.nn.Module,
    title_suffix: str = "",
    normalize_input: Optional[str] = None,
    unnorm_output: Optional[str] = None,
    filename: Optional[str] = None,
    device: Optional[torch.device] = None,
    colorbar: bool = True,
) -> torch.Tensor:
    """
    Shows input, prediction, and ground truth images side-by-side.

    Args:
        input_img: Input image tensor ([1, C=1, H, W] or [C=1, H, W]).
        model_img: Ground truth image tensor ([1, C=1, H, W] or [C=1, H, W]).
        model: Trained PyTorch model.
        title_suffix: Optional string for plot titles.
        normalize_input: Name of input normalization function.
        unnorm_output: Name of output unnormalization function.
        filename: File path to save plot, or None to display.
        device: Device for inference.
        colorbar: Show colorbar. Default is True.

    Returns:
        The predicted output tensor [C, H, W] shape.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use the refactored predict_image function
    predicted_output = predict_image(
        model, input_img, normalize_input, unnorm_output, device
    )  # Returns tensor on CPU

    # --- Plotting ---
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 8))
    fig.tight_layout(pad=3.0)

    # Transform to matplotlib format 2d images (H, W)
    input_img = transform_img_to_2d(input_img)

    # Input image
    img_in_np = input_img.squeeze().cpu().numpy()
    axes[0].imshow(img_in_np.T, interpolation="nearest", aspect="auto")
    axes[0].set_title(f"Input Image {title_suffix}")

    # Predicted image
    img_pred_np = predicted_output.squeeze().squeeze().cpu().numpy()
    axes[1].imshow(img_pred_np.T, interpolation="nearest", aspect="auto")
    axes[1].set_title(f"Predicted Image {title_suffix}")

    # Ground truth image
    img_gt_np = model_img.squeeze().cpu().numpy()
    im3 = axes[2].imshow(img_gt_np.T, interpolation="nearest", aspect="auto")
    axes[2].set_title(f"Ground Truth {title_suffix}")

    if colorbar:
        fig.colorbar(im3, ax=axes.tolist(), fraction=0.046, pad=0.04)

    if filename is None:
        plt.show()
    else:
        try:
            fig.savefig(filename, bbox_inches="tight")
            print(f"Saved inference+GT plot to {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}", file=sys.stderr)

    return predicted_output.squeeze()
