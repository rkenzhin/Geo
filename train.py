import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.optim as optim
import torchvision
import yaml
from datasets import (
    CROP_SIZE,
    DATA_DIR,
    DATA_INPUT_BASENAMES,
    DATA_OUTPUT_BASENAMES,
    NY,
    NZ,
    GeoDataset,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import logger
from unet import UNet
from viz import plot_history_simple

MODELS_DIR = "models"
RUNS_DIR = "runs"
BEST_MODEL_SUFFIX = "_best.pth"
EPOCH_MODEL_SUFFIX = "_epoch{}.pth"
LOSS_HISTORY_FILE = "loss_history.txt"
TRAINING_HISTORY_IMG = "training_history.png"
INDEX_FILE = "train_val_test_indexes.txt"
DEFAULT_CONFIG_FILE = "config.yaml"
FINAL_CONFIG_SAVE_FILE = "final_config.yaml"

# --- Default configuration (used if file not found or parameter missing) ---
DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "models_dir": MODELS_DIR,
        "runs_dir": RUNS_DIR,
        "data_dir": "/home/kenzhin/Data/TrainingData2025_Feb",
    },
    "data": {
        "input_basenames": ["data_training_input_ext.bin"],
        "output_basenames": [
            "data_training_output_D_ext.bin",
            "data_training_output_R_ext.bin",
        ],
        "indexes_range": NZ,
        "starting_index": 0,
        "ending_index": NZ,
        "val_ratio": 0.1,
        "is_load_to_RAM": False,
        "use_second_data": False,
        "is_crop": False,
        "NY_crop": NY,
        "add_noise": False,
        "normalize_input": "normalize_input_to_max1",
        "normalize_output": "normalize_output_to_max1",
        "crop_size": CROP_SIZE,
    },
    "model": {
        "name": "UNet",
        "params": {"n_channels": 1, "n_classes": 1, "bilinear": False},
    },
    "training": {
        "epochs": 1,
        "batch_size": 1,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "loss_function": "MSELoss",
        "validation_cadence": 1,
        "save_epoch_cadence": 10,
        "early_stop_patience": 50,
        "log_images_cadence": 1,
    },
    "logging": {"tb_prefix": "Unet", "comment": "", "log_2_images": False},
    "run_management": {"save_config": True},
}


# --- Helper functions for configuration ---
def load_config(config_path: str) -> Dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None:
            print(
                f"Warning: Config file {config_path} is empty. Using defaults.",
                file=sys.stderr,
            )
            return {}
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at {config_path}. Using defaults.",
            file=sys.stderr,
        )
        return {}
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}", file=sys.stderr)
        return {}


def merge_configs(default: Dict, loaded: Dict) -> Dict:
    """Recursively merges the loaded configuration with the default one."""
    merged = default.copy()
    for key, value in loaded.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested_config_value(
    config_dict: Dict, key_path: List[str], default: Any = None
) -> Any:
    """Safely extracts a nested value from the configuration dictionary."""
    current = config_dict

    for key in key_path:
        if not isinstance(current, dict):
            print(f"Error: '{key}' cannot be applied to {type(current)}")
            return default
        if key not in current:
            print(f"Key '{key}' not found")
            return default
        current = current[key]
    return current


def setup_parser(cfg: Dict) -> argparse.ArgumentParser:
    """Configures the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train UNet using config file and CLI overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Parameters that can be overridden via CLI ---
    # Take default from loaded cfg
    parser.add_argument(
        "--epochs",
        default=get_nested_config_value(cfg, ["training", "epochs"], 1),
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=get_nested_config_value(cfg, ["training", "batch_size"], 1),
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=get_nested_config_value(cfg, ["training", "learning_rate"]),
        type=float,
    )
    parser.add_argument(
        "--save_config",
        default=get_nested_config_value(cfg, ["run_management", "save_config"], True),
        type=bool,
        help="Save final config to run directory",
    )
    parser.add_argument(
        "--is_load_to_RAM",
        default=get_nested_config_value(cfg, ["data", "is_load_to_RAM"]),
        action="store_true",
    )
    parser.add_argument(
        "--add_noise",
        default=get_nested_config_value(cfg, ["data", "add_noise"]),
        action="store_true",
    )
    parser.add_argument(
        "--indexes_range",
        default=get_nested_config_value(cfg, ["data", "indexes_range"]),
        type=int,
    )
    parser.add_argument(
        "--starting_index",
        help="Starting_index to load images",
        type=int,
        default=get_nested_config_value(cfg, ["data", "starting_index"], 0),
    )
    parser.add_argument(
        "--ending_index",
        help="Ending_index to load images",
        type=int,
        default=get_nested_config_value(cfg, ["data", "ending_index"], NZ),
    )
    default_optimizer = get_nested_config_value(cfg, ["training", "optimizer"])
    parser.add_argument(
        "--optimizer",
        default=default_optimizer,
        type=str,
        help=f"Optimizer name from torch.optim (default: {default_optimizer})",
    )

    default_loss = get_nested_config_value(cfg, ["training", "loss_function"])
    parser.add_argument(
        "--loss_function",
        default=default_loss,
        type=str,
        help=f"Loss function name from torch.nn or custom (default: {default_loss})",
    )
    # Define choices explicitly for normalization args
    norm_choices = [
        "normalize_minmax",
        "normalize_mean_std",
        "normalize_input_to_max1",
    ]
    parser.add_argument(
        "--normalize_input",
        help="Normalization for input data (from datasets.py)",
        type=str,
        choices=norm_choices,
        default=get_nested_config_value(cfg, ["data", "normalize_input"]),
    )
    parser.add_argument(
        "--normalize_output",
        help="Normalization for output data (from datasets.py)",
        type=str,
        default=get_nested_config_value(cfg, ["data", "normalize_output"]),
    )
    parser.add_argument(
        "--val_ratio",
        help="Ratio of data for validation set",
        type=float,
        default=get_nested_config_value(cfg, ["data", "val_ratio"], 0.1),
    )
    parser.add_argument(
        "--use_second_data",
        help="Use the second dataset file pair (defined in datasets.py)",
        action="store_true",
        default=get_nested_config_value(cfg, ["data", "use_second_data"]),
    )
    parser.add_argument(
        "--is_crop",
        help="Crop training images (requires --is_load_to_RAM True)",
        action="store_true",
        default=get_nested_config_value(cfg, ["data", "is_crop"]),
    )
    parser.add_argument(
        "--crop_size",
        help="Crop training images (requires --is_load_to_RAM True)",
        type=int,
        default=get_nested_config_value(cfg, ["data", "crop_size"], CROP_SIZE),
    )
    parser.add_argument(
        "--NY_crop",
        help="Vertical cropping size (up to this value from the top)",
        type=int,
        default=get_nested_config_value(cfg, ["data", "NY_crop"], NY),
    )
    parser.add_argument(
        "comment",
        nargs="?",
        default=get_nested_config_value(cfg, ["logging", "comment"], ""),
        type=str,
    )
    # Add parameters that are ONLY from the config to args via set_defaults
    parser.set_defaults(
        data_dir=get_nested_config_value(cfg, ["paths", "data_dir"], DATA_DIR),
        models_dir=get_nested_config_value(cfg, ["paths", "models_dir"], MODELS_DIR),
        runs_dir=get_nested_config_value(cfg, ["paths", "runs_dir"], RUNS_DIR),
        data_input_basenames=get_nested_config_value(
            cfg, ["data", "input_basenames"], DATA_INPUT_BASENAMES
        ),
        data_output_basenames=get_nested_config_value(
            cfg, ["data", "output_basenames"], DATA_OUTPUT_BASENAMES
        ),
        model_name=get_nested_config_value(cfg, ["model", "name"]),
        model_params=get_nested_config_value(cfg, ["model", "params"]),
        validation_cadence=get_nested_config_value(
            cfg, ["training", "validation_cadence"]
        ),
        log_images_cadence=get_nested_config_value(
            cfg, ["training", "log_images_cadence"]
        ),
        save_epoch_cadence=get_nested_config_value(
            cfg, ["training", "save_epoch_cadence"]
        ),
        early_stop_patience=get_nested_config_value(
            cfg, ["training", "early_stop_patience"]
        ),
        tb_prefix=get_nested_config_value(cfg, ["logging", "tb_prefix"]),
        tb_log_2_images=get_nested_config_value(cfg, ["logging", "log_2_images"]),
    )
    return parser


def load_config_with_cli():
    """Two-stage config loading with CLI override support"""
    # Stage 1: Only parse config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Path to the YAML configuration file.",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Load and merge config file if config.yaml is empty (for example)
    cfg = merge_configs(DEFAULT_CONFIG, load_config(pre_args.config))

    # Stage 2: Full parsing with config defaults
    parser = setup_parser(cfg)
    args = parser.parse_args(remaining_argv)

    return args


# Custom Loss
class L1_L2_Loss(torch.nn.Module):
    """Calculates a combined L1 (MAE) and L2 (MSE) loss."""

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined loss.

        Args:
            prediction: The model's output tensor.
            target: The ground truth tensor.

        Returns:
            The combined L1 + L2 loss tensor.
        """
        return self.l1_loss(prediction, target) + self.mse_loss(prediction, target)


class GeoTrainer:
    """
    Handles the training process for the Unet model on Geo datasets.

    Includes argument parsing, data loading, training loop, validation,
    TensorBoard logging, model saving, and history plotting.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the trainer based on the args object from argparse (contains the merged config from config.yaml and CLI).

        Args:
        args: Namespace object from argparse with final parameters.
        """
        self.args = args
        self.log = logger.logging.getLogger(__name__)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # We use paths from the config if they are there, otherwise from constants
        self.models_base_dir = self.args.models_dir
        self.runs_base_dir = self.args.runs_dir
        self.data_dir = self.args.data_dir

        self._init_model()
        self._init_optimizer()
        self._init_loss()

        self.history: List[Dict[str, Optional[float]]] = []

        self.validation_cadence = self.args.validation_cadence  # Validate every epoch
        self.log_images_cadence = self.args.log_images_cadence  # Log images every epoch
        self.save_epoch_cadence = (
            self.args.save_epoch_cadence
        )  # Save model checkpoint every {N} epochs
        self.model_checkpoint_dict = {
            "best_val_loss": float("inf"),
            "epochs_no_improve": 0,
            "early_stop_patience": self.args.early_stop_patience,  # Stop after N epochs with no improvement
        }

        self.time_str = datetime.now().strftime("%Y-%m-%d_%H.%M.")
        self.total_samples = self.args.indexes_range  # Take from args
        self.run_name = (
            f"{self.args.tb_prefix}_{self.loss_fn._get_name()}_{self.time_str}_"
            f"{self.total_samples}img_{self.args.epochs}epochs_{self.args.comment}"
        )
        self.run_dir = os.path.join(self.models_base_dir, self.run_name)
        self.log_dir = os.path.join(self.runs_base_dir, self.run_name)

        self.train_writer: Optional[SummaryWriter] = None
        self.val_writer: Optional[SummaryWriter] = None

        self._validate_args()  # Validate final args
        self._log_initial_setup()

    def _validate_args(self):
        """Performs basic validation of parsed arguments."""
        if self.args.is_load_to_RAM and self.args.indexes_range > 2000:
            self.log.warning(
                "Loading > 2000 images to RAM might exceed memory limits. "
                f"Current indexes_range: {self.args.indexes_range}"
            )
        # Depends on RAM memory this assertion might be needed:
        # assert not (self.args.indexes_range >= 2000 and self.args.is_load_to_RAM), \
        #     "Loading >= 2000 images to RAM might cause overflow."

        assert 0 <= self.args.val_ratio < 1, "Validation ratio must be in [0, 1)."
        if self.args.is_crop and not self.args.is_load_to_RAM:
            raise ValueError("--is_crop requires --is_load_to_RAM True to be enabled.")
        if self.args.NY_crop > NY:
            self.log.warning(
                f"--NY_crop ({self.args.NY_crop}) > dataset NY ({NY}). Using full height."
            )
            self.args.NY_crop = NY

        if self.args.starting_index >= self.args.ending_index:
            raise ValueError(
                f"starting_index ({self.args.starting_index}) must be < ending_index ({self.args.ending_index})."
            )

    def _log_initial_setup(self):
        """Logs the parsed arguments and training parameters."""
        self.log.info("----- Configuration -----")
        for arg, value in vars(self.args).items():
            self.log.info(f"  {arg:<20}: {value}")
        self.log.info("-------------------------\n")
        self.log.info("----- Training Setup -----")
        self.log.info(f"  Device              : {self.device}")
        self.log.info(f"  Total samples       : {self.total_samples}")
        self.log.info(f"  Run Directory       : {self.run_dir}")
        self.log.info(f"  TensorBoard Log Dir : {self.log_dir}")
        self.log.info("-------------------------\n")

    def _init_model(self):
        """Initialize model with config parameters and safety checks."""
        if self.args.model_name == "UNet":
            self.model = UNet(**self.args.model_params).to(
                self.device, dtype=torch.float32
            )
        else:
            raise ValueError(f"Unknown model name: {self.args.model_name}")

    def _init_optimizer(self):
        """Initialize optimizer with safety checks."""
        try:
            # Basic initialization with lr. For other parameters (momentum, weight_decay, etc.)
            # you can add 'optimizer_params' section to config.yaml and pass them like this:
            # optim_params_cfg = get_nested_config_value(cfg, ['training', 'optimizer_params'], {})
            # self.optimizer = optimizer_class(model_params, lr=learning_rate, **optim_params_cfg)
            optimizer_name = self.args.optimizer
            learning_rate = self.args.learning_rate
            model_params = self.model.parameters()
            optimizer_class = getattr(optim, optimizer_name)

            self.optimizer = optimizer_class(model_params, lr=learning_rate)
        except AttributeError:
            self.log.error(f"Optimizer '{optimizer_name}' not found in torch.optim.")

            available_optimizers = [
                name
                for name in dir(optim)
                if isinstance(getattr(optim, name), type)
                and issubclass(getattr(optim, name), optim.Optimizer)
                and name != "Optimizer"
            ]
            self.log.info(
                f"Available optimizers in torch.optim might include: {', '.join(available_optimizers[:10])}..."
            )
            raise ValueError(f"Unknown optimizer specified: {optimizer_name}")
        except Exception as e:
            self.log.error(f"Error instantiating optimizer '{optimizer_name}': {e}")
            raise

    def _init_loss(self):
        try:
            loss_function_name = self.args.loss_function
            if hasattr(torch.nn, loss_function_name):
                loss_class = getattr(torch.nn, loss_function_name)

                self.loss_fn = loss_class()
            # Trying to find the loss function in the current module (for custom ones)
            elif loss_function_name in globals() and isinstance(
                globals()[loss_function_name], type
            ):
                loss_class = globals()[loss_function_name]
                self.loss_fn = loss_class()
            else:
                raise AttributeError
        except AttributeError:
            self.log.error(
                f"Loss function '{loss_function_name}' not found in torch.nn or current module."
            )
            available_losses = [
                name
                for name in dir(torch.nn)
                if "Loss" in name and not name.startswith("_")
            ]
            self.log.info(
                f"Available standard losses might include: {', '.join(available_losses[:10])}..."
            )
            raise ValueError(f"Unknown loss function specified: {loss_function_name}")
        except Exception as e:
            self.log.error(
                f"Error instantiating loss function '{loss_function_name}': {e}"
            )
            raise

    def _init_tensorboard_writers(self):
        """Initializes TensorBoard SummaryWriters."""
        if self.train_writer is None:
            self.train_writer = SummaryWriter(
                log_dir=os.path.join(self.log_dir, "train")
            )
            self.val_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "val"))
            self.log.info(f"TensorBoard writers initialized in '{self.log_dir}'")

    def _init_dataloaders(self) -> Tuple[DataLoader, DataLoader, List[int]]:
        """
        Initializes training and validation datasets and dataloaders.
        Also determines and saves test indices.

        Returns:
            Tuple of (train_dataloader, val_dataloader, test_indices).
        """
        input_basenames = self.args.data_input_basenames
        output_basenames = self.args.data_output_basenames

        if len(input_basenames) == 1 and len(output_basenames) > 1:
            data_input_paths = [os.path.join(self.data_dir, input_basenames[0])] * len(
                output_basenames
            )
        else:
            data_input_paths = [
                os.path.join(self.data_dir, name) for name in input_basenames
            ]
        data_output_paths = [
            os.path.join(self.data_dir, name) for name in output_basenames
        ]

        # Simplified dataset selection
        train_dataset = GeoDataset(
            data_dir=self.data_dir,
            data_input_name=data_input_paths,
            data_output_name=data_output_paths,
            val_ratio=self.args.val_ratio,
            is_val=False,
            indexes_range=self.args.indexes_range,
            normalize_input=self.args.normalize_input,
            add_noise=self.args.add_noise,
            normalize_output=self.args.normalize_output,
            use_second_data=self.args.use_second_data,
            NY_crop=self.args.NY_crop,
            is_load_data_to_RAM=self.args.is_load_to_RAM,
            is_crop=self.args.is_crop,
            starting_index=self.args.starting_index,
            ending_index=self.args.ending_index,
        )

        # Use same settings for validation with noise and crop if True on train set.
        val_dataset = GeoDataset(
            data_dir=self.data_dir,
            data_input_name=data_input_paths,
            data_output_name=data_output_paths,
            val_ratio=self.args.val_ratio,
            is_val=True,
            indexes_range=self.args.indexes_range,
            normalize_input=self.args.normalize_input,
            add_noise=self.args.add_noise,
            normalize_output=self.args.normalize_output,
            use_second_data=self.args.use_second_data,
            NY_crop=self.args.NY_crop,
            is_load_data_to_RAM=self.args.is_load_to_RAM,
            is_crop=self.args.is_crop,
            starting_index=self.args.starting_index,
            ending_index=self.args.ending_index,
        )

        # Save train/val/test indices
        os.makedirs(self.run_dir, exist_ok=True)  # Ensure directory exists
        index_file_path = os.path.join(self.run_dir, INDEX_FILE)
        with open(index_file_path, "w") as fp:
            all_indices = set(range(NZ))  # Assuming NZ is total dataset size
            train_indices_list = train_dataset.indexes.tolist()
            val_indices_list = val_dataset.indexes.tolist()
            test_indices = list(
                all_indices - set(train_indices_list) - set(val_indices_list)
            )
            test_indices.sort()  # Keep indices sorted

            index_data = {
                "train_indexes": train_indices_list,
                "val_indexes": val_indices_list,
                "test_indexes": test_indices,
            }
            json.dump(index_data, fp, indent=4)
            self.log.info(f"Train/Val/Test indices saved to '{index_file_path}'")

        # Create DataLoaders
        num_workers = 4 if self.device.type == "cuda" else 0
        pin_memory = True if self.device.type == "cuda" else False

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_dl, val_dl

    def _train_epoch(self, train_dl: DataLoader) -> float:
        """Performs one epoch of training."""
        train_loss = 0.0
        self.model.train()  # Set model to training mode

        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(self.device, dtype=torch.float32)
            y_batch = y_batch.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.loss_fn(predictions, y_batch)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_dl)  # Average loss for the epoch

    def _validate_epoch(self, val_dl: DataLoader) -> float:
        """Performs one epoch of validation."""
        val_loss = 0.0
        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.float32)

                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)

                val_loss += loss.item()

        return val_loss / len(val_dl)  # Average loss for the epoch

    def _log_metrics(self, epoch: int, mode: Literal["train", "val"], loss: float):
        """Logs loss metrics to console and TensorBoard."""
        self._init_tensorboard_writers()
        self.log.info(f"Epoch {epoch:03d}: {mode}_loss = {loss:.5f}")
        writer = getattr(self, f"{mode}_writer")
        if writer:
            writer.add_scalar("Loss", loss, epoch)
            writer.flush()

    def _log_images(self, epoch: int, mode: Literal["train", "val"], dl: DataLoader):
        """Logs sample images to TensorBoard."""
        self._init_tensorboard_writers()
        self.model.eval()  # Ensure model is in eval mode for prediction

        num_images_to_log = 2 if self.args.tb_log_2_images else 1
        #  if num_images_to_log = 2 we show 1st and last image in dataset.
        indices_to_log = (
            [0, len(dl.dataset) - 1]
            if num_images_to_log == 2 and len(dl.dataset) > 1
            else [0]
        )

        if not dl.dataset:
            self.log.warning(f"Cannot log images for empty {mode} dataset.")
            return

        inputs = []
        ground_truths = []
        for i in indices_to_log:
            try:
                img_input, img_gt = dl.dataset[i]
                inputs.append(img_input)
                ground_truths.append(img_gt)
            except IndexError:
                self.log.warning(f"Index {i} out of bounds for {mode} dataset logging.")
                return  # Don't proceed if indices are invalid

        input_batch = torch.stack(inputs).to(self.device, dtype=torch.float32)
        gt_batch = torch.stack(ground_truths)  # Keep GT on CPU for grid creation

        with torch.no_grad():
            predicted_batch = self.model(input_batch).detach().cpu()

        # Create grids
        input_grid = torchvision.utils.make_grid(
            input_batch.cpu(),
            nrow=1,
            padding=5,
            pad_value=1,
            normalize=True,  # Normalize for viz
        )
        gt_grid = torchvision.utils.make_grid(
            gt_batch,
            nrow=1,
            padding=5,
            pad_value=1,
            normalize=True,  # Normalize for viz
        )
        pred_grid = torchvision.utils.make_grid(
            predicted_batch,
            nrow=1,
            padding=5,
            pad_value=1,
            normalize=True,  # Normalize for viz
        )

        writer = getattr(self, f"{mode}_writer")
        if writer:
            # Log input/GT only once (or less frequently) if desired
            if epoch == 1:  # Only log input/GT on first image log call
                writer.add_image(f"{mode}/Input_Image", input_grid, epoch)
                writer.add_image(f"{mode}/Ground_Truth_Image", gt_grid, epoch)
            writer.add_image(f"{mode}/Predicted_Image", pred_grid, epoch)
            writer.flush()

    def _save_history_loss(self, train_loss: float, val_loss: Optional[float]):
        """Appends epoch losses to the history list."""
        self.history.append({"train_loss": train_loss, "val_loss": val_loss})

    def _save_model(self, epoch: int, is_best: bool = False):
        """Saves the model checkpoint."""
        suffix = BEST_MODEL_SUFFIX if is_best else EPOCH_MODEL_SUFFIX.format(epoch)
        filename = f"{self.time_str}_totalSamples{self.total_samples}{suffix}"
        filepath = os.path.join(self.run_dir, filename)

        # torch.save(self.model.state_dict(), filepath) # Saving state_dict for more flexibility
        torch.save(self.model, filepath)  # Saving entire model

        status = f"Best epoch {epoch} model" if is_best else f"Epoch {epoch} model"
        self.log.info(f"{status} saved to '{filepath}'")

    def _save_final_config(self):
        """Saves the final launch configuration to YAML."""
        config_save_path = os.path.join(self.run_dir, FINAL_CONFIG_SAVE_FILE)
        try:
            args_dict = vars(self.args)

            with open(config_save_path, "w") as f:
                yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)
            self.log.info(f"Final configuration saved to '{config_save_path}'")
        except Exception as e:
            self.log.error(f"Error saving final config: {e}")

    def train(self):
        """Runs the main training loop."""
        start_time = datetime.now()
        if self.args.save_config:  # Saving dinal_config
            self._save_final_config()
        self.log.info(
            f"Starting training on {self.device} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        self.log.info("Initializing datasets and dataloaders...")
        train_dl, val_dl = self._init_dataloaders()
        self.log.info(
            f"  Train samples: {len(train_dl.dataset)}, Val samples: {len(val_dl.dataset)}"
        )
        self.log.info(
            f"  Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Batch size: {self.args.batch_size}\n"
        )

        self.log.info("Starting training loop...\n")
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = datetime.now()
            self.log.info(f"--- Epoch {epoch}/{self.args.epochs} ---")

            # --- Training ---
            train_loss = self._train_epoch(train_dl)
            self._log_metrics(epoch, "train", train_loss)

            # --- Validation ---
            val_loss = None
            if epoch == 1 or epoch % self.validation_cadence == 0:
                val_loss = self._validate_epoch(val_dl)
                self._log_metrics(epoch, "val", val_loss)

                # Checkpointing and Early Stopping
                if val_loss < self.model_checkpoint_dict["best_val_loss"]:
                    self.log.info(
                        f"Validation loss improved ({self.model_checkpoint_dict['best_val_loss']:.5f} -> {val_loss:.5f}). Saving best model."
                    )
                    self.model_checkpoint_dict["best_val_loss"] = val_loss
                    self.model_checkpoint_dict["epochs_no_improve"] = 0
                    self._save_model(epoch, is_best=True)
                else:
                    self.model_checkpoint_dict["epochs_no_improve"] += 1
                    self.log.info(
                        f"Validation loss did not improve for {self.model_checkpoint_dict['epochs_no_improve']} epoch(s)."
                    )
                    if (
                        self.model_checkpoint_dict["epochs_no_improve"]
                        >= self.model_checkpoint_dict["early_stop_patience"]
                    ):
                        self.log.info(f"Early stopping triggered after {epoch} epochs.")
                        break

            # --- Image Logging ---
            if epoch == 1 or epoch % self.log_images_cadence == 0:
                try:
                    self._log_images(epoch, "train", train_dl)
                    self._log_images(epoch, "val", val_dl)
                except Exception as e:
                    self.log.error(f"Error logging images for epoch {epoch}: {e}")

            # --- Periodic Checkpointing (Optional) ---
            if self.save_epoch_cadence and epoch % self.save_epoch_cadence == 0:
                self._save_model(epoch, is_best=False)

            self._save_history_loss(train_loss, val_loss)

            epoch_duration = datetime.now() - epoch_start_time
            self.log.info(
                f"Epoch {epoch} duration: {str(epoch_duration).split('.')[0]}\n"
            )

        # --- Post-Training ---
        self._save_model(epoch, is_best=False)  # Save last epoch model
        train_duration = datetime.now() - start_time
        self.log.info(f"Total training duration: {str(train_duration).split('.')[0]}\n")

        # Save history and plot
        history_path = os.path.join(self.run_dir, LOSS_HISTORY_FILE)
        try:
            with open(history_path, "w") as fp:
                json.dump(self.history, fp, indent=4)
            plot_path = os.path.join(self.run_dir, TRAINING_HISTORY_IMG)
            plot_history_simple(self.history, filename=plot_path)
            self.log.info(f"Loss history saved to '{history_path}'")
            self.log.info(f"Training plot saved to '{plot_path}'")
        except Exception as e:
            self.log.error(f"Error saving/plotting history: {e}")

        # Copy log file to run directory
        log_filename = logger.DEFAULT_LOG_FILENAME  # Use the default from logger
        try:
            shutil.copy(log_filename, os.path.join(self.run_dir, log_filename))
            self.log.info(f"Log file copied to '{self.run_dir}'")
        except Exception as e:
            self.log.error(f"Error copying log file: {e}")

        # Close TensorBoard writers
        if self.train_writer:
            self.train_writer.close()
        if self.val_writer:
            self.val_writer.close()

    def main(self):
        """Main execution function."""
        # Ensure run directory exists and set up file logging there
        os.makedirs(self.run_dir, exist_ok=True)
        self.log.info("Starting GeoTrainer...")
        self.train()
        self.log.info(
            f"Training finished successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )


if __name__ == "__main__":
    args = load_config_with_cli()
    GeoTrainer(args=args).main()
