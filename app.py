import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
from emotion import detect_emotion_from_image

app = Flask(__name__)
CORS(app)

# Path to your trained deepfake model
MODEL_PATH = "xception_deepfake_image.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 224

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_array):
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = tf.keras.applications.xception.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def predict_image(img_array):
    input_tensor = preprocess_image(img_array)
    prob = model.predict(input_tensor, verbose=0)[0][0]
    # INVERTED: because model learned opposite mapping
    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if label == "FAKE" else 1 - prob
    return {"label": label, "confidence": float(confidence)}

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def predict_video(video_path):
    frames = extract_frames(video_path, num_frames=10)
    if not frames:
        return {"label": "ERROR", "confidence": 0.0}
    probs = []
    for frame in frames:
        inp = preprocess_image(frame)
        prob = model.predict(inp, verbose=0)[0][0]
        probs.append(prob)
    avg_prob = np.mean(probs)
    # INVERTED
    label = "REAL" if avg_prob >= 0.5 else "FAKE"
    confidence = avg_prob if label == "FAKE" else 1 - avg_prob
    return {"label": label, "confidence": float(confidence)}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if ext in ['mp4', 'avi', 'mov', 'mkv']:
            result = predict_video(tmp_path)
        else:
            img = cv2.imread(tmp_path)
            if img is None:
                return jsonify({"error": "Could not read image"}), 400
            result = predict_image(img)
            # Emotion detection only for images
            emotion = detect_emotion_from_image(img)
            if emotion:
                result['emotion'] = emotion
    finally:
        os.unlink(tmp_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)