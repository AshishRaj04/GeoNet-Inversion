import torch
import os

# Paths
BASE_DATA_DIR = "/dataset/dataset"
RESULTS_DIR = "/results_v5"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "uNet_v5.pth")

ALL_FILE_PAIRS = [
    (os.path.join(BASE_DATA_DIR, "CurveFault_A_seis2_1_0.npy"), os.path.join(BASE_DATA_DIR, "CurveFault_A_vel2_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveFault_A_seis4_1_0.npy"), os.path.join(BASE_DATA_DIR, "CurveFault_A_vel4_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveFault_B_seis6_1_0.npy"), os.path.join(BASE_DATA_DIR, "CurveFault_B_vel6_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveFault_B_seis8_1_0.npy"), os.path.join(BASE_DATA_DIR, "CurveFault_B_vel8_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveVel_A_data1.npy"), os.path.join(BASE_DATA_DIR, "CurveVel_A_model1.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveVel_A_data2.npy"), os.path.join(BASE_DATA_DIR, "CurveVel_A_model2.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveVel_B_data1.npy"), os.path.join(BASE_DATA_DIR, "CurveVel_B_model1.npy")),
    (os.path.join(BASE_DATA_DIR, "CurveVel_B_data2.npy"), os.path.join(BASE_DATA_DIR, "CurveVel_B_model2.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatFault_A_seis2_1_0.npy"), os.path.join(BASE_DATA_DIR, "FlatFault_A_vel2_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatFault_A_seis4_1_0.npy"), os.path.join(BASE_DATA_DIR, "FlatFault_A_vel4_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatFault_B_seis6_1_0.npy"), os.path.join(BASE_DATA_DIR, "FlatFault_B_vel6_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatFault_B_seis8_1_0.npy"), os.path.join(BASE_DATA_DIR, "FlatFault_B_vel8_1_0.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatVel_A_data1.npy"), os.path.join(BASE_DATA_DIR, "FlatVel_A_model1.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatVel_A_data2.npy"), os.path.join(BASE_DATA_DIR, "FlatVel_A_model2.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatVel_B_data1.npy"), os.path.join(BASE_DATA_DIR, "FlatVel_B_model1.npy")),
    (os.path.join(BASE_DATA_DIR, "FlatVel_B_data2.npy"), os.path.join(BASE_DATA_DIR, "FlatVel_B_model2.npy")),
    (os.path.join(BASE_DATA_DIR, "Style_A_data1.npy"), os.path.join(BASE_DATA_DIR, "Style_A_model1.npy")),
    (os.path.join(BASE_DATA_DIR, "Style_A_data2.npy"), os.path.join(BASE_DATA_DIR, "Style_A_model2.npy")),
    (os.path.join(BASE_DATA_DIR, "Style_B_data1.npy"), os.path.join(BASE_DATA_DIR, "Style_B_model1.npy")),
    (os.path.join(BASE_DATA_DIR, "Style_B_data2.npy"), os.path.join(BASE_DATA_DIR, "Style_B_model2.npy")),
]

# ─── Hyperparameters ───
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
EPOCHS = 400
WARMUP_EPOCHS = 50
VAL_FREQ = 1
CHECKPOINT_FREQ = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Y_MAX = 4000.0

# Loss weights
L1_WEIGHT = 1.0
MSE_WEIGHT = 0.1
GRADIENT_WEIGHT = 0.5
SSIM_WEIGHT = 0.3
TV_WEIGHT = 0.01
TOTAL_LOSS_SCALE = 100.0
