import cv2
import numpy as np
from deepface import DeepFace

def detect_emotion_from_image(image_array):
    """
    Detect dominant emotion from an image (numpy array, BGR format from OpenCV).
    Returns a string (emotion label) or None if detection fails.
    """
    try:
        # DeepFace expects RGB, convert from BGR
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(rgb_img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        return None

def detect_emotion_from_file(image_path):
    """
    Load an image from file path and detect emotion.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    return detect_emotion_from_image(img)