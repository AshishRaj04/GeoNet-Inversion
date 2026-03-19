import torch
import os

# Paths
BASE_DATA_DIR = "data"
SEISMIC_DATA_PATH = os.path.join(BASE_DATA_DIR, "seis2_1_0.npy")
VELOCITY_DATA_PATH = os.path.join(BASE_DATA_DIR, "vel2_1_0.npy")
MODEL_SAVE_PATH = os.path.join("results", "model.pth")

# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training weights
L1_WEIGHT = 1.0
MSE_WEIGHT = 0.1
GRADIENT_WEIGHT = 0.2
SSIM_WEIGHT = 0.3
TOTAL_LOSS_SCALE = 100.0
