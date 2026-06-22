import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

def load_images_from_folder(folder, label, limit=None):
    """Load all images from a folder (e.g., REAL or FAKE) with given label."""
    images, labels = [], []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
                if limit and len(images) >= limit:
                    break
    return np.array(images), np.array(labels)

def load_dataset_with_folder_structure():
    """Load train/val/test from directories: train/REAL, train/FAKE, etc."""
    print("Loading dataset using folder structure...")
    def load_split(split_dir):
        real_dir = os.path.join(split_dir, "REAL")
        fake_dir = os.path.join(split_dir, "FAKE")
        X, y = [], []
        if os.path.exists(real_dir):
            Xr, yr = load_images_from_folder(real_dir, 0)
            X.extend(Xr); y.extend(yr)
        if os.path.exists(fake_dir):
            Xf, yf = load_images_from_folder(fake_dir, 1)
            X.extend(Xf); y.extend(yf)
        return np.array(X), np.array(y)
    X_train, y_train = load_split(TRAIN_DIR)
    X_val, y_val = load_split(VAL_DIR)
    X_test, y_test = load_split(TEST_DIR)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_dataset_from_csv(csv_path, image_dir, sample_size=None):
    meta = pd.read_csv(csv_path)
    real_df = meta[meta["label"] == "REAL"]
    fake_df = meta[meta["label"] == "FAKE"]
    if sample_size:
        real_df = real_df.sample(sample_size//2, random_state=RANDOM_SEED)
        fake_df = fake_df.sample(sample_size//2, random_state=RANDOM_SEED)
    sample_meta = pd.concat([real_df, fake_df])
    
    train_val, test = train_test_split(sample_meta, test_size=TEST_SPLIT,
                                       random_state=RANDOM_SEED, stratify=sample_meta['label'])
    train, val = train_test_split(train_val, test_size=VAL_SPLIT,
                                  random_state=RANDOM_SEED, stratify=train_val['label'])
    
    def extract(split_df):
        images, labels = [], []
        for idx, row in split_df.iterrows():
            img_name = row['videoname'][:-4] + '.jpg'
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(1 if row['label'] == 'FAKE' else 0)
        return np.array(images), np.array(labels)
    
    X_train, y_train = extract(train)
    X_val, y_val = extract(val)
    X_test, y_test = extract(test)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_dataset():
    if USE_FOLDER_STRUCTURE:
        return load_dataset_with_folder_structure()
    else:
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
        return load_dataset_from_csv(METADATA_FILE, IMAGE_DIR)