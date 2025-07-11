import os
import shutil

SRC_DIR = r"C:\Users\dylan\Workspace\military-target-id-system\data\raw\military\images"
DEST_DIR = r"C:\Users\dylan\Workspace\military-target-id-system\data\raw\military"
VALID_EXTS = (".jpg", ".jpeg", ".png")

def move_and_flatten():
    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if file.lower().endswith(VALID_EXTS):
                src_path = os.path.join(root, file)
                new_filename = file
                dest_path = os.path.join(DEST_DIR, new_filename)
                
                # Rename if file already exists
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    new_filename = f"{name}_{counter}{ext}"
                    dest_path = os.path.join(DEST_DIR, new_filename)
                    counter += 1

                shutil.move(src_path, dest_path)
                print(f"✅ Moved: {src_path} → {dest_path}")

if __name__ == "__main__":
    move_and_flatten()
