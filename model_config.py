import torch

# Paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = f"{DATA_DIR}/models"
RESULTS_DIR = f"{DATA_DIR}/results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
METRICS_DIR = f"{RESULTS_DIR}/metrics"

# Processed data
X_TRAIN_PATH = f"{PROCESSED_DATA_DIR}/X_train.pt"
Y_TRAIN_PATH = f"{PROCESSED_DATA_DIR}/y_train.pt"
X_VAL_PATH = f"{PROCESSED_DATA_DIR}/X_val.pt"
Y_VAL_PATH = f"{PROCESSED_DATA_DIR}/y_val.pt"
X_TEST_PATH = f"{PROCESSED_DATA_DIR}/X_test.pt"
Y_TEST_PATH = f"{PROCESSED_DATA_DIR}/y_test.pt"

# Hyperparameters
INPUT_SIZE = 20  # Number features
OUTPUT_SIZE = 3  # Buy/sell/hold

# TCN channel sizes. Each element is a layer.
NUM_CHANNELS = [32, 64, 128]
KERNEL_SIZE = 3
DROPOUT = 0.2
SEQUENCE_LENGTH = 100

# Training hyperparameters
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01  # For AdamW

# For logging/saving
MODEL_NAME = "tcn_model_v1.pth"
MODEL_SAVE_PATH = f"{MODELS_DIR}/{MODEL_NAME}"
PLOT_NAME = "training_curves.png"
PLOT_SAVE_PATH = f"{PLOTS_DIR}/{PLOT_NAME}"
