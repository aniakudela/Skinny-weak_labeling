import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import date
from PIL import Image
import matplotlib.pyplot as plt

base_path = "/Users/annakudela/Skinny/"
today = date.today().strftime("%Y_%m_%d")

bbox_mask_dir = os.path.join(base_path, "bbox_masks")
rule_mask_dir   = os.path.join(base_path, "rule_masks")
output_mask_dir = os.path.join(base_path, today + " masks")
os.makedirs(output_mask_dir, exist_ok=True)

image_extensions = ('.bmp', '.jpg', '.jpeg', '.png')
all_filenames = [f for f in sorted(os.listdir(bbox_mask_dir))
                 if os.path.isfile(os.path.join(bbox_mask_dir, f))
                 and f.lower().endswith(image_extensions)]

example_image = None

for filename in tqdm(all_filenames, desc="Łączenie masek"):
    bbox_path = os.path.join(bbox_mask_dir, filename)
    rule_path = os.path.join(rule_mask_dir, filename)

    if not os.path.exists(rule_path):
        continue

    bbox = cv2.imread(bbox_path, cv2.IMREAD_GRAYSCALE)
    rule = cv2.imread(rule_path, cv2.IMREAD_GRAYSCALE)

    if bbox is None or rule is None:
        continue

    if bbox.shape != rule.shape:
        rule = cv2.resize(rule, (bbox.shape[1], bbox.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    result = np.full(bbox.shape, 128, dtype=np.uint8)

    is_background = bbox > 128
    result[is_background] = 255

    is_bbox_area = bbox <= 128
    is_rule_skin = rule <= 128  
    
    is_skin = is_bbox_area & is_rule_skin
    result[is_skin] = 0


    filename_bmp = os.path.splitext(filename)[0] + ".bmp"
    out_path = os.path.join(output_mask_dir, filename_bmp)
    
    Image.fromarray(result).convert('L').save(out_path)

    if example_image is None:
        example_image = (bbox, rule, result)

if example_image is not None:
    bbox, rule, result = example_image

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(bbox, cmap='gray')
    plt.title("Maska zgrubna")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rule, cmap='gray')
    plt.title("Maska regułowa")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='gray')
    plt.title("Iloraz masek")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Brak masek do podglądu.")
