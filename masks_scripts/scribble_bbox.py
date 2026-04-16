import os
import cv2
import numpy as np
import math
from tqdm import tqdm
from datetime import date
from PIL import Image
import matplotlib.pyplot as plt

base_path = "/Users/annakudela/Skinny/"
today = date.today().strftime("%Y_%m_%d")

input_mask_dir = os.path.join(base_path, "SkinBin")
bbox_dir = os.path.join(base_path, "bbox_masks")
scribble_dir = os.path.join(base_path, "scribble_masks")
output_mask_dir = os.path.join(base_path, today + str(" masks"))
os.makedirs(output_mask_dir, exist_ok=True)

TARGET_RATIO    = 0.10                 # docelowa powierzchnia
KERNEL_SHAPE    = cv2.MORPH_ELLIPSE    # albo cv2.MORPH_RECT
MIN_BLOB_AREA   = 30                   # px 
CONNECTIVITY    = 8                    # kierunki

image_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

def preview_grid(scr, bbox, merged, title="Scribble + bbox"):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(scr, cmap='gray', vmin=0, vmax=255)
    plt.title("Maska scribble")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(bbox, cmap='gray', vmin=0, vmax=255)
    plt.title("Bounding box")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(merged, cmap='gray', vmin=0, vmax=255)
    plt.title("Merge")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

all_files = [f for f in sorted(os.listdir(input_mask_dir))
             if os.path.isfile(os.path.join(input_mask_dir, f))
             and f.lower().endswith(image_extensions)]

kernel = cv2.getStructuringElement(KERNEL_SHAPE, (3, 3))

for fname in tqdm(all_files, desc="Scribble + bbox"):
    scribble_path = os.path.join(scribble_dir, fname)
    bbox_path = os.path.join(bbox_dir, fname)
    s = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(bbox_path, cv2.IMREAD_GRAYSCALE)
    if s is None or s is None:
        continue

    if s.shape != b.shape:
        b = cv2.resize(b, (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST)

    skin = (s == 0)
    bg = (b == 255)
    bg = bg & (~skin)

    merged = np.full(s.shape, 128, dtype=np.uint8)
    merged[bg] = 255
    merged[skin] = 0

    out_path = os.path.join(output_mask_dir, os.path.splitext(fname)[0] + ".bmp")
    Image.fromarray(merged).convert('L').save(out_path)

preview_grid(s, b, merged)
