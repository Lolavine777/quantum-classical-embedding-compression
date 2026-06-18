import os
import torch

# Reproducibility
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PhoBERT
PHOBERT_MODEL = "vinai/phobert-base"
MAX_LENGTH = 256
EMBED_DIM = 768

# Dataset
DATASET_NAME = "uitnlp/vietnamese_students_feedback"
NUM_CLASSES = 3
LABEL_NAMES = ["negative", "neutral", "positive"]

# Compression dimension
D_OUT = 8
D_OUT_LIST = [8, 16, 32, 64]  # For multi-dimension ablation study

# PQC
N_QUBITS = 8
N_LAYERS = 2

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(BASE_DIR, "data", "embeddings")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAINING_LOGS_DIR = os.path.join(RESULTS_DIR, "training_logs")

# Autoencoder hyperparameters (joint training)
AE_EPOCHS = 50
AE_LR = 1e-3
AE_BATCH_SIZE = 256
AE_LAMBDA_RECON = 0.1  # weight for reconstruction loss: loss = lambda * MSE + CE

# PQC hyperparameters
PQC_EPOCHS = 30
PQC_LR = 1e-2
PQC_BATCH_SIZE = 256

# Classifier hyperparameters (PCA downstream)
CLF_EPOCHS = 100
CLF_LR = 1e-3
CLF_BATCH_SIZE = 256
CLF_PATIENCE = 10
