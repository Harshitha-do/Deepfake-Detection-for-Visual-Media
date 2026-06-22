import os

# ==================================================
# PATHS – UPDATE THESE TO MATCH YOUR DATASET
# ==================================================

# ----- IMAGE DATASET (using folder structure) -----
USE_FOLDER_STRUCTURE = True   # Set True if you have train/val/test subfolders with REAL/FAKE
DATA_ROOT = "C:/Users/itsva/Projects/DeepFake_project/image_dataset"    # Folder containing train/, val/, test/
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

# ----- Or if using CSV metadata (set USE_FOLDER_STRUCTURE = False) -----
METADATA_FILE = "C:/Users/itsva/Projects/DeepFake_project/Deepfake_videos/List_of_testing_videos.txt"
IMAGE_DIR = "C:/Users/itsva/Projects/DeepFake_project/real_and_fake_face"

# ----- VIDEO DATASET -----
VIDEO_FOLDER = "C:/Users/itsva/Projects/DeepFake_project/Deepfake_videos"   # Main folder with subfolders (Celeb-synthesis, etc.)

# ----- Training settings -----
BATCH_SIZE = 32
IMG_SIZE = 224
RANDOM_SEED = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.3          # fraction of training set to use for validation

# ----- Model training -----
EPOCHS_FROZEN = 3
EPOCHS_FINETUNE = 10
LEARNING_RATE_FROZEN = 0.1
LEARNING_RATE_FINETUNE = 0.01
MOMENTUM = 0.9