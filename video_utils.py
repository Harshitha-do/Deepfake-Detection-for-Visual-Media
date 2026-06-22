import cv2
import numpy as np
import tensorflow as tf
from config import IMG_SIZE

def extract_frames(video_path, num_frames=None, sample_every_k=1):
    """
    Extract frames from a video file.
    
    Args:
        video_path: path to video file
        num_frames: total number of frames to extract (uniform sampling)
        sample_every_k: sample one frame every k frames (if num_frames is None)
    
    Returns:
        numpy array of frames (RGB, resized to IMG_SIZE x IMG_SIZE)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.array([])
    
    if num_frames is not None and num_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frames.append(frame)
    else:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_every_k == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frames.append(frame)
            count += 1
    cap.release()
    return np.array(frames)

def preprocess_frames_for_model(frames, model_type='xception'):
    """Preprocess frames for model input."""
    frames = frames.astype(np.float32)
    if model_type == 'xception':
        frames = tf.keras.applications.xception.preprocess_input(frames)
    elif model_type == 'custom':
        frames = frames / 255.0
    return frames