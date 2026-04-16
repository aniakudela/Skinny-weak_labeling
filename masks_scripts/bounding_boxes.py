import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import date
from PIL import Image

base_path = "/Users/annakudela/Skinny/"
today = date.today().strftime("%Y_%m_%d")

input_mask_dir = os.path.join(base_path, "SkinBin")
output_mask_dir = os.path.join(base_path, today + str(" masks"))
os.makedirs(output_mask_dir, exist_ok=True)

list_path = os.path.join(output_mask_dir, "train.txt")
try:
    all_filenames = [f for f in sorted(os.listdir(input_mask_dir)) 
                     if os.path.isfile(os.path.join(input_mask_dir, f))]
except FileNotFoundError:
    exit() 

image_extensions = ('.bmp', '.jpg', '.jpeg', '.png')

with open(list_path, 'w', encoding='utf-8') as f:
    for filename in tqdm(all_filenames, desc='Processing masks'):
        clean_filename = filename.strip()
        if not clean_filename.lower().endswith(image_extensions):
            continue 

        in_path = os.path.join(input_mask_dir, clean_filename) 
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue
        
        H, W = mask.shape

        border = np.concatenate([
            mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]
        ])
        bg_is_light = np.median(border) >= 127

        if bg_is_light:
            _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = H * W
        MIN_AREA = max(50, int(0.001 * img_area))   
        MAX_FRAC = 0.95                             
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if (w * h) / img_area > MAX_FRAC:
                continue
            boxes.append((x, y, w, h))

        filled_rects = np.full((H, W), 255, dtype=np.uint8) 
        for (x, y, w, h) in boxes:
            cv2.rectangle(filled_rects, (x, y), (x + w, y + h), 0, thickness=-1)

        base_name, _ = os.path.splitext(filename)
        new_filename = base_name + ".bmp"
        out_path = os.path.join(output_mask_dir, clean_filename)
        Image.fromarray(filled_rects).convert('1').save(out_path)
        f.write(base_name + '\n')
