# scripts/flatten_folder.py

import os
import shutil

def flatten_folder(target_dir):
    for root, _, files in os.walk(target_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            # Avoid overwriting duplicates
            if not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)

    # Optionally remove empty folders
    for root, dirs, _ in os.walk(target_dir, topdown=False):
        for d in dirs:
            full_path = os.path.join(root, d)
            if not os.listdir(full_path):
                os.rmdir(full_path)

if __name__ == "__main__":
    flatten_folder("data/raw/normal")
    print("✅ All images have been flattened into 'data/raw/normal'")
