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

input_mask_dir = os.path.join(base_path, "bbox_masks")
output_mask_dir = os.path.join(base_path, today + str(" masks"))
os.makedirs(output_mask_dir, exist_ok=True)

SKIN_REMOVE_RATIO = 0.7
np.random.seed(42)

image_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

all_files = [f for f in sorted(os.listdir(input_mask_dir))
             if os.path.isfile(os.path.join(input_mask_dir, f))
             and f.lower().endswith(image_extensions)]

for fname in tqdm(all_files, desc="Zmniejszanie bounding boxów"):
    in_path = os.path.join(input_mask_dir, fname)
    mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    drop_mask = mask.copy()
    skin_pixels = np.where(mask == 0)
    rand_vals = np.random.rand(len(skin_pixels[0]))
    drop_mask[skin_pixels[0][rand_vals <= SKIN_REMOVE_RATIO], skin_pixels[1][rand_vals <= SKIN_REMOVE_RATIO]] = 255

    out_path = os.path.join(output_mask_dir, os.path.splitext(fname)[0] + ".bmp")
    Image.fromarray(drop_mask.astype(np.uint8), mode="L").save(out_path)
