import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Paths to required artifacts
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Load artifacts
scaler = joblib.load(SCALER_PATH)
model = joblib.load(KNN_MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        hist = cv2.calcHist([gray], [0], None, [3], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges > 0)

        # Return only the 6 features used during training
        return np.array([mean_intensity, std_intensity, *hist, edge_count]).reshape(1, -1)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def predict_image(image_path):
    features = extract_features(image_path)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    label = label_encoder.inverse_transform(prediction)[0]
    return label

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify an image as military or normal.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    try:
        result = predict_image(args.image_path)
        print(f"Predicted Label: {result}")
    except Exception as e:
        print(f"Error: {e}")
