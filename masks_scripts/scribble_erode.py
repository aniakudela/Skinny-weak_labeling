import os
import cv2
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import date
import matplotlib.pyplot as plt

base_path = "/Users/annakudela/Skinny/"
today = date.today().strftime("%Y_%m_%d")

input_mask_dir = os.path.join(base_path, "SkinBin")
output_mask_dir = os.path.join(base_path, today + str(" masks"))
os.makedirs(output_mask_dir, exist_ok=True)


TARGET_RATIO    = 0.10                 # docelowa powierzchnia
KERNEL_SHAPE    = cv2.MORPH_ELLIPSE    # albo cv2.MORPH_RECT
MIN_BLOB_AREA   = 30                   # px 
CONNECTIVITY    = 8                    # kierunki

image_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

def erosion_history(blob: np.ndarray, kernel, max_iters=512):
    hist = [blob.copy()]
    cur = blob.copy()
    for _ in range(max_iters):
        nxt = cv2.erode(cur, kernel, iterations=1)
        if np.count_nonzero(nxt == 255) == 0:
            break
        hist.append(nxt)
        cur = nxt
    return hist

def pick_iter_closest_to_ratio(history, target_ratio):
    # dla białych blobów wybiera iterację o liczbie białych pikseli
    # najbliższej target_ratio * area0
    areas = [int(np.count_nonzero(h == 255)) for h in history]
    area0 = areas[0]
    target = max(1, int(round(area0 * target_ratio)))
    idx = int(np.argmin([abs(a - target) for a in areas]))
    return idx, history[idx]


def preview_grid(history, n_show=10, title="Blob erosion"):
    k = min(n_show, len(history))
    cols = 5
    rows = math.ceil(k / cols)
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(k):
        plt.subplot(rows, cols, i+1)
        plt.imshow(255 - history[i], cmap='gray', vmin=0, vmax=255)
        plt.title(f"Iteration {i+1}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

all_files = [f for f in sorted(os.listdir(input_mask_dir))
             if os.path.isfile(os.path.join(input_mask_dir, f))
             and f.lower().endswith(image_extensions)]

kernel = cv2.getStructuringElement(KERNEL_SHAPE, (3, 3))

for fname in tqdm(all_files, desc="Scribble - erozja per blob"):
    in_path = os.path.join(input_mask_dir, fname)
    m = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue

    H, W = m.shape[:2]
    skin01 = (m == 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(skin01, connectivity=CONNECTIVITY)
    scribble = np.full((H, W), 255, dtype=np.uint8)
    largest_hist = None
    biggest_area = -1

    for lab in range(1, num):
        comp = (labels == lab).astype(np.uint8)
        area = int(comp.sum())
        if area < MIN_BLOB_AREA:
            continue

        comp_white = (comp * 255).astype(np.uint8)
        hist = erosion_history(comp_white, kernel)
        _, best = pick_iter_closest_to_ratio(hist, TARGET_RATIO)

        # naniesienie czarnych pikseli z wybranej iteracji
        scribble[best == 255] = 0

        if area > biggest_area:
            biggest_area = area
            largest_hist = hist

    out_path = os.path.join(output_mask_dir, os.path.splitext(fname)[0] + ".bmp")
    Image.fromarray(scribble).convert('1').save(out_path)

preview_grid(largest_hist)


    # out_skin = os.path.join(output_scribble_dir, os.path.splitext(fname)[0] + ".bmp")
    # out_bg   = os.path.join(output_bg_dir,       os.path.splitext(fname)[0] + "_bg.bmp")
    # Image.fromarray(scribble).convert('1').save(out_skin)
    # Image.fromarray(bg_only).convert('1').save(out_bg)
