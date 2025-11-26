import cv2
import numpy as np
from pathlib import Path
import shutil
import yaml
import math
import random

# ============================
# USER SETTINGS (EDIT HERE)
# ============================

INPUT_DIR = Path("./images")          # root containing class subfolders
OUTPUT_DIR = Path("./out")        # output root
VAL_SPLIT = 0.2                          # 20% validation
TEST_SPLIT = 0.05                         # usually 0 unless you want test data
THRESHOLD = 127                          # white threshold

# roboflow fields made dynamic but can be overwritten here
ROBOFLOW_INFO = {
    "workspace": "auto",
    "project": "auto",
    "version": 1,
    "license": "CC BY 4.0",
    "url": "auto"
}

# ===============================================
# Helper: find square bbox around white region
# ===============================================
def find_hand_bbox(img, threshold=127):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = (mask > 0).astype(np.uint8) * 255  # force-pure-white mask

    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1
    side = max(w, h)

    cx = x_min + w / 2
    cy = y_min + h / 2

    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = x1 + side - 1
    y2 = y1 + side - 1

    return x1, y1, x2, y2


# ===============================================
# Clamp square bbox to image boundaries
# ===============================================
def clamp(bbox, W, H):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    side = max(w, h)

    cx = (x1 + x2)/2
    cy = (y1 + y2)/2

    x1 = int(cx - side/2)
    y1 = int(cy - side/2)
    x2 = x1 + side - 1
    y2 = y1 + side - 1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)

    return x1, y1, x2, y2


# ===============================================
# Convert to YOLO format
# ===============================================
def to_yolo(bbox, W, H):
    x1, y1, x2, y2 = bbox
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w/2
    cy = y1 + h/2
    return cx/W, cy/H, w/W, h/H


# ===============================================
# Main processing
# ===============================================
def main():

    # Reset output folder
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "train/images").mkdir(parents=True)
    (OUTPUT_DIR / "train/labels").mkdir(parents=True)
    (OUTPUT_DIR / "valid/images").mkdir(parents=True)
    (OUTPUT_DIR / "valid/labels").mkdir(parents=True)
    (OUTPUT_DIR / "test/images").mkdir(parents=True)
    (OUTPUT_DIR / "test/labels").mkdir(parents=True)

    class_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_id = {name: i for i, name in enumerate(class_names)}

    annotated = 0
    missing = 0
    sizes = []

    # process images
    for class_dir in class_dirs:
        cid = class_to_id[class_dir.name]

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print("Unreadable:", img_path)
                continue

            H, W = img.shape[:2]
            bbox = find_hand_bbox(img, THRESHOLD)

            if bbox is None:
                print("[NO HAND FOUND]", img_path)
                missing += 1
                continue

            bbox = clamp(bbox, W, H)
            cx, cy, w, h = to_yolo(bbox, W, H)

            # pick split bucket
            r = random.random()
            if r < VAL_SPLIT:
                split = "valid"
            elif r < VAL_SPLIT + TEST_SPLIT:
                split = "test"
            else:
                split = "train"

            # copy image
            out_image = OUTPUT_DIR / split / "images" / img_path.name
            shutil.copy(img_path, out_image)

            # write label
            out_label = OUTPUT_DIR / split / "labels" / (img_path.stem + ".txt")
            with open(out_label, "w") as f:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            side = bbox[2] - bbox[0]
            sizes.append((side, str(img_path)))

            annotated += 1

    # Outlier detection
    if sizes:
        arr = np.array([s for s, _ in sizes])
        m, sd = arr.mean(), arr.std()
        lower, upper = m - 2*sd, m + 2*sd

        print("\n=== OUTLIER REPORT ===")
        print("Thresholds:", lower, upper)
        for side, path in sizes:
            if side < lower or side > upper:
                print("[OUTLIER]", side, path)

    # Write YAML
    yaml_data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(class_names),
        "names": class_names,
        "roboflow": ROBOFLOW_INFO
    }

    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print("\n=== DONE ===")
    print("Annotated:", annotated)
    print("Missing:", missing)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
