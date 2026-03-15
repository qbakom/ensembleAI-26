"""
KROK 2: Izolacja sygnału EKG przez próg jasności.

Sygnał jest zawsze czarny (gray < 50).
Tło, siatka — jaśniejsze.

Test:
  python step2_binarize.py --image SCIEZKA/ecg_train_XXXX.png --out_dir ./debug2
"""

import os
import argparse
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from step1_deskew import deskew

SIGNAL_THRESH = 50  # piksele ciemniejsze niż to = sygnał


def binarize(img: np.ndarray) -> np.ndarray:
    """
    Zwraca binarną maskę: 255 = sygnał, 0 = tło/siatka.
    Wejście: wyprostowany obraz BGR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < SIGNAL_THRESH).astype(np.uint8) * 255
    return mask


def run_debug(img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    assert img is not None, f"Nie można wczytać: {img_path}"

    # Krok 1: wyprostuj
    rotated, angle = deskew(img, debug=True)
    cv2.imwrite(f"{out_dir}/01_deskewed.png", rotated)

    # Krok 2: binaryzacja
    mask = binarize(rotated)
    cv2.imwrite(f"{out_dir}/02_mask.png", mask)

    coverage = np.sum(mask > 0) / mask.size
    print(f"Kąt: {angle:.2f}°")
    print(f"Pokrycie maski: {coverage:.4f}  (oczekiwane 0.01–0.05)")
    print(f"Zapisano w: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True)
    parser.add_argument("--out_dir", default="./debug_step2")
    args = parser.parse_args()
    run_debug(args.image, args.out_dir)