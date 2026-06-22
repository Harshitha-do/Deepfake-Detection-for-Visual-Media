import os
import shutil
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
FRAME_ROOT = "extracted_frames"          # folder created by extract_all_frames.py
DEST_ROOT = "frame_training_data"        # output folder with train/val/test
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42
# ====================

# Map subfolder names to labels (0=REAL, 1=FAKE)
label_map = {
    "Celeb-synthesis": 1,   # FAKE
    "Celeb-real": 0,        # REAL
    "YouTube-real": 0       # REAL
}

# Gather all video folders with their labels
videos = []
for subfolder, label in label_map.items():
    subfolder_path = os.path.join(FRAME_ROOT, subfolder)
    if not os.path.exists(subfolder_path):
        print(f"Warning: {subfolder_path} not found, skipping")
        continue
    for video_name in os.listdir(subfolder_path):
        video_path = os.path.join(subfolder_path, video_name)
        if os.path.isdir(video_path):
            videos.append((video_path, label))

print(f"Found {len(videos)} videos")

# Split videos into train/val/test (by video, not by frame)
video_paths = [v[0] for v in videos]
labels = [v[1] for v in videos]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    video_paths, labels, test_size=VAL_SIZE+TEST_SIZE, random_state=RANDOM_SEED, stratify=labels)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), random_state=RANDOM_SEED, stratify=temp_labels)

def copy_frames(video_paths, labels, split_name):
    for vpath, lbl in zip(video_paths, labels):
        label_dir = "FAKE" if lbl == 1 else "REAL"
        dest_dir = os.path.join(DEST_ROOT, split_name, label_dir)
        os.makedirs(dest_dir, exist_ok=True)
        frames = [f for f in os.listdir(vpath) if f.lower().endswith('.jpg')]
        for fname in frames:
            src = os.path.join(vpath, fname)
            dst = os.path.join(dest_dir, f"{os.path.basename(vpath)}_{fname}")
            shutil.copy(src, dst)
        print(f"Copied {len(frames)} frames from {vpath} to {dest_dir}")

print("Copying train frames...")
copy_frames(train_paths, train_labels, "train")
print("Copying validation frames...")
copy_frames(val_paths, val_labels, "val")
print("Copying test frames...")
copy_frames(test_paths, test_labels, "test")

print(f"Done! Dataset created in {DEST_ROOT}")