import torch
import os

# Paths
BASE_DATA_DIR = "/dataset/dataset"
RESULTS_DIR = "/results_v4"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "uNet_v4.pth")

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

# ─── Architecture ───
BASE_CH = 64
BOTTLENECK_DEPTH = 6
BOTTLENECK_HEADS = 8
WINDOW_SIZE = (5, 4)
MLP_RATIO = 4.0
DROP_PATH_RATE = 0.1

# ─── V4 Physics (Deepwave / OpenFWI Acquisition Geometry) ───
DX = 10.0                     # Grid spacing (meters)
DT = 0.001                    # Time step (seconds)
NT = 1000                     # Number of time samples
FREQ = 15.0                   # Ricker wavelet central frequency (Hz)
N_SOURCES = 5                 # Sources per sample
N_RECEIVERS = 70              # Receivers per sample
SOURCE_DEPTH = 1              # Source depth (grid index)
RECEIVER_DEPTH = 1            # Receiver depth (grid index)
SOURCE_POSITIONS = [0, 17, 34, 52, 69]  # Verified from OpenFWI data
PML_WIDTH = 20                # Absorbing boundary width

# Physics loss schedule
PHYSICS_WEIGHT = 0.1          # Weight for physics loss term
PHYSICS_START_EPOCH = 50      # Activate physics loss after warmup
PHYSICS_RAMP_EPOCHS = 100     # Linearly ramp weight over this many epochs
PHYSICS_MAX_WEIGHT = 0.5      # Maximum physics loss weight after ramp

# V2 legacy paths
V2_MODEL_PATH = os.path.join("results/v2", "uNet_v2.pth")
V2_SEISMIC_PATH = os.path.join(BASE_DATA_DIR, "CurveFault_A_seis2_1_0.npy")
V2_VELOCITY_PATH = os.path.join(BASE_DATA_DIR, "CurveFault_A_vel2_1_0.npy")



