# UNet for Geo-Dataset Image Restoration

This repository contains the code for our research on **3D Seismic Inversion for Fracture Model Reconstruction Based on Machine Learning paper**. This project implements and trains a UNet neural network for image 3D fracture model reconstruction tasks on binary syntetic geo-datasets. It includes scripts for flexible experiments (with hyperparameters, image sizes tuning etc.), training, evaluation, and inference. Supports flexible configuration via YAML files with command-line overrides.

**Research Paper**: [3D Seismic Inversion for Fracture Model Reconstruction Based on Machine Learning](https://doi.org/10.1007/978-3-031-49435-2_8)

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Setup and Installation](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [License](#license)
- [Contact](#contact)

## Features

*   **UNet Model:** Standard UNet architecture suitable for image processing tasks.
*   **Large Data Handling:** Efficiently loads data from large binary files either on-the-fly or by pre-loading into RAM.
*   **Flexible Configuration:** Core parameters are set via `config.yaml`, but can be easily overridden using command-line arguments (`argparse`).
*   **Normalization:** Supports various normalization methods for input and output data.
*   **Training:** Includes a script for training the model using specified data and parameters:
    *   Choice of optimizers (`Adam`, `SGD`, etc.) and loss functions (`MSELoss`, `L1Loss`, etc.).
    *   Logging of metrics and image samples to TensorBoard.
    *   Model checkpoint saving (best model, per-epoch).
    *   Early Stopping.
    *   Saving training history and the final run configuration.
*   **Evaluation:** `evaluation.py` script for evaluating the trained model's performance on a hold-out set using batch processing.
*   **Inference:** Example script/notebook for applying the trained model to new data.
*   **Visualization:** Utilities for plotting training history and displaying images.

## File Structure

<pre>
project-root/
├── config.yaml # Main configuration file (YAML)
├── requirements.txt # Python dependencies (or requirements.in/.txt)
├── datasets.py # Data loading, single-image normalization
├── evaluation.py # Model evaluation script (batch processing)
├── inference.py # Script with inference functions
├── inference.ipynb # Jupyter Notebook with inference example
├── logger.py # Logging setup
├── train.py # Main training script (config + argparse)
├── unet.py # UNet model definition
├── utils.py # Utilities (batch normalization functions for inference)
├── viz.py # Visualization functions
├── models/ # Directory for saving trained models
│ └── <run_name>/ # Directory for a specific training run
│ ├── *_best.pth
│ ├── *_epochX.pth
│ ├── final_config.yaml # Final config for the run
│ ├── loss_history.txt
│ ├── training_history.png
│ └── training_log.txt
├── runs/ # Directory for TensorBoard logs
│ └── <run_name>/
│ ├── train/
│ └── val/
└── README.md # This file
</pre>

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/rkenzhin/Geo.git>
    cd <Geo>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **For GPU:** Ensure you have compatible NVIDIA drivers and CUDA Toolkit installed. You might need to install a specific PyTorch version for your CUDA version (see [PyTorch official website](https://pytorch.org/)).

4.  **Data:**
    *   **Download** the required binary data files (`.bin`).
    *   **Place** them in a directory on your machine (or server).
    *   **Edit `config.yaml`:** Update the `paths.data_dir` section with the **correct absolute path** to your data directory. Also, ensure `input_basenames`, `output_basenames`, and the dimension parameters (`NX`, `NY`, `NZ`) in `config.yaml` match your dataset.

## Configuration

The primary configuration is managed through the `config.yaml` file. It includes sections for paths (`paths`), data parameters (`data`), model details (`model`), training settings (`training`), logging options (`logging`), and run management (`run_management`).

Most parameters set in `config.yaml` can be overridden when running `train.py` via command-line arguments. For example, to change the number of epochs: `python train.py --epochs 50`.

When training starts (and if `save_config: true`), the final effective configuration (including CLI overrides) will be saved to `final_config.yaml` within the run directory (`models/<run_name>/`).

## Usage

### Training

1.  **Configure `config.yaml`** for your dataset and desired training parameters.
2.  **Run the training script:**
    ```bash
    # Use default config.yaml
    python train.py

    # Override parameters from the command line
    python train.py --epochs 100 --batch_size 8 --learning_rate 1e-5 --comment "lr_experiment"
    ```
3.  **Monitoring (TensorBoard):**
    *   Logs are saved to `runs/<run_name>` during training.
    *   Launch TensorBoard:
        ```bash
        tensorboard --logdir runs
        ```
    *   Open `http://localhost:6006` (or the specified address) in your browser.

4.  **Results:** Trained models, logs, plots, and the final config are saved in `models/<run_name>`.

### Evaluation

The `evaluation.py` script evaluates a saved model's performance on the validation or test set using batch processing.

*   **Adaptation:** Currently, the script requires manually setting the model path and parameters within the `if __name__ == "__main__":` block. Adapt it to accept command-line arguments or read a separate evaluation config file for better usability.
*   **Running (after adaptation):**
    ```bash
    python evaluation.py --model_path models/<run_name>/<model_file.pth> --mode test --batch_size 16 --save_bin True ...
    ```
*   **Output:** Calculates loss metrics (MSE) and can optionally save predictions, inputs, and ground truth tensors to binary files.

### Inference

The `inference.ipynb` notebook (or a similar `inference.py` script) demonstrates how to load a trained model and apply it to new ("real") data.

*   **Adaptation:** Modify the paths to the model and input data within the notebook/script.
*   **Running:** Execute the notebook cells or run the script.
*   **Output:** Model predictions for the new data.

## License

[MIT License]

## Contact

[romankenzhin@gmail.com](mailto:romankenzhin@gmail.com)
