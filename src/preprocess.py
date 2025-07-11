# src/preprocess.py

import os
import cv2
import numpy as np
import pandas as pd
from skimage import exposure

DATA_DIR = "data/raw"
OUTPUT_CSV = "data/processed/vehicle_features.csv"   # Updated to /processed
IMAGE_SIZE = (128, 128)
VALID_EXTS = (".jpg", ".jpeg", ".png")

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMAGE_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Feature calculations
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        aspect_ratio = img.shape[1] / img.shape[0]
        hist = cv2.calcHist([gray], [0], None, [3], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges > 0)
        contrast = exposure.is_low_contrast(gray, fraction_threshold=0.35)

        return [mean_intensity, std_intensity, aspect_ratio, *hist, edge_count, int(contrast)]
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def main():
    rows = []
    labels = []

    for label in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        print(f"📂 Processing '{label}' images...")
        for file in os.listdir(class_dir):
            if not file.lower().endswith(VALID_EXTS):
                continue
            path = os.path.join(class_dir, file)
            features = extract_features(path)
            if features:
                rows.append(features)
                labels.append(label)

    columns = [
        "mean_intensity", "std_intensity", "aspect_ratio",
        "hist_bin1", "hist_bin2", "hist_bin3",
        "edge_count", "low_contrast", "label"
    ]

    df = pd.DataFrame(rows, columns=columns[:-1])
    df["label"] = labels

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
