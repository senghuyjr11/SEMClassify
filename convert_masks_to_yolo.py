import os
import cv2
import numpy as np
from glob import glob

def mask_to_boxes(mask_path):
    mask = cv2.imread(mask_path, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > 1 and bh > 1:
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            bw, bh = bw / w, bh / h
            boxes.append([0, cx, cy, bw, bh])  # class 0 = corrosion
    return boxes

def convert_split(split='train'):
    image_root = f"processed_dataset_split/{split}"
    mask_root = f"processed_dataset_masks"
    out_img_dir = f"yolo_data/images/{split}"
    out_lbl_dir = f"yolo_data/labels/{split}"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for dirpath, _, files in os.walk(image_root):
        for file in files:
            if not file.endswith(".png"):
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, file), image_root)
            img_path = os.path.join(image_root, rel_path)
            mask_path = os.path.join(mask_root, rel_path)

            if not os.path.exists(mask_path):
                continue

            boxes = mask_to_boxes(mask_path)
            if not boxes:
                continue

            # Save image
            save_img_path = os.path.join(out_img_dir, rel_path)
            os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            img = cv2.imread(img_path)
            cv2.imwrite(save_img_path, img)

            # Save label
            label_path = os.path.join(out_lbl_dir, rel_path.replace(".png", ".txt"))
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w") as f:
                for box in boxes:
                    f.write(" ".join([str(round(b, 6)) for b in box]) + "\n")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        convert_split(split)
    print("✅ YOLO data generated successfully.")
