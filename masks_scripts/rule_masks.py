import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import date
from PIL import Image

# --- filtr Kovac et al. ---
def skin_rule(r, g, b):
    r, g, b = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
    return (r > 95) & (g > 40) & (b > 20) & \
           ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 15) & \
           (np.abs(r - g) > 15) & (r > g) & (r > b)

base_path = "/Users/annakudela/Skinny/"
today = date.today().strftime("%Y_%m_%d")

input_dir = os.path.join(base_path, "dataset/org/features")
output_mask_dir = os.path.join(base_path, today + str(" masks"))
os.makedirs(output_mask_dir, exist_ok=True)


try:
    all_filenames = [f for f in sorted(os.listdir(input_dir)) 
                     if os.path.isfile(os.path.join(input_dir, f))]
except FileNotFoundError:
    exit() 

image_extensions = ('.bmp', '.jpg', '.jpeg', '.png')


for filename in tqdm(all_filenames, desc='Processing masks'):
    clean_filename = filename.strip()
    if not clean_filename.lower().endswith(image_extensions):
        continue 

    in_path = os.path.join(input_dir, clean_filename)
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # konwersja BGR -> RGB
    if img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if img.ndim < 3 or img.shape[2] < 3:
        continue
        
    skin_bool = skin_rule(r, g, b)
    mask_uint8 = np.where(skin_bool, 0, 255).astype(np.uint8)

    out_name = os.path.splitext(clean_filename)[0] + '_s.bmp'
    out_path = os.path.join(output_mask_dir, out_name)
    Image.fromarray(mask_uint8).convert('1').save(out_path)

# ['im03301_s.bmp', 'im03331_s.bmp']
