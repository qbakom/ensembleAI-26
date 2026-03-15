"""
KROK 3: Podział obrazu na 12 bloków odprowadzeń (+ rhythm strip).

Strategia:
  1. Crop kartki (usuń czarne tło)
  2. Wykryj granice obszaru sygnału (row_sum > 5% max)
  3. Podziel równo na 4 wiersze i 4 kolumny

Test:
  python step3_split.py --image SCIEZKA/ecg_train_XXXX.png --out_dir ./debug3
"""

import os
import argparse
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
import sys
sys.path.insert(0, '.')
from step1_deskew import deskew
from step2_binarize import binarize

LEADS_GRID = [
    ["I",   "AVR", "V1", "V4"],
    ["II",  "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]

# Rhythm strip to 4. wiersz (II full-width)
RHYTHM_LEAD = "II_RHYTHM"


def crop_card(img: np.ndarray) -> np.ndarray:
    """Usuwa czarne tło wokół kartki."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, cw, ch = cv2.boundingRect(coords)
    # Mały margines żeby nie ucinać krawędzi
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    cw = min(img.shape[1] - x, cw + 2 * margin)
    ch = min(img.shape[0] - y, ch + 2 * margin)
    return img[y:y+ch, x:x+cw]


def find_signal_bounds(mask: np.ndarray) -> tuple[int, int]:
    """
    Zwraca (y_top, y_bot) — granice obszaru z sygnałem.
    Używa projekcji wierszy maski binarnej.
    """
    h = mask.shape[0]
    row_sum = mask.sum(axis=1) / 255
    smoothed = uniform_filter1d(row_sum.astype(float), size=max(1, h // 40))

    threshold = smoothed.max() * 0.05
    # Szukaj tylko od 20% wysokości w dół - unikamy czarnego tła na górze
    search_start = h // 5
    signal_rows = np.where(smoothed > threshold)[0]
    signal_rows = signal_rows[signal_rows >= search_start]

    if len(signal_rows) == 0:
        return 0, h

    margin = h // 30
    y_top = max(0, int(signal_rows[0]) - margin)
    y_bot = int(signal_rows[-1])
    return y_top, y_bot


def split_leads(img: np.ndarray,
                mask: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Dzieli obraz i maskę na bloki odprowadzeń.
    Zwraca {lead_name: (img_block, mask_block)}.
    """
    h, w = mask.shape
    y_top, y_bot = find_signal_bounds(mask)
    signal_h = y_bot - y_top

    # 4 wiersze równe w obszarze sygnału
    row_h = signal_h // 4
    col_w = w // 4

    blocks = {}

    for r in range(3):
        for c in range(4):
            lead = LEADS_GRID[r][c]
            y1 = y_top + r * row_h
            y2 = y_top + (r + 1) * row_h
            x1 = c * col_w
            x2 = (c + 1) * col_w
            blocks[lead] = (img[y1:y2, x1:x2], mask[y1:y2, x1:x2])

    # Rhythm strip = 4. wiersz
    y1 = y_top + 3 * row_h
    y2 = y_bot
    if y2 > y1:
        blocks[RHYTHM_LEAD] = (img[y1:y2, :], mask[y1:y2, :])

    return blocks


def run_debug(img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    assert img is not None

    rotated, angle = deskew(img, debug=True)
    cropped = crop_card(rotated)
    mask = binarize(cropped)
    h, w = mask.shape

    y_top, y_bot = find_signal_bounds(mask)
    signal_h = y_bot - y_top
    row_h = signal_h // 4
    col_w = w // 4

    # Wizualizacja podziału
    vis = cropped.copy()
    cv2.line(vis, (0, y_top), (w, y_top), (0, 255, 0), 3)
    cv2.line(vis, (0, y_bot), (w, y_bot), (0, 255, 0), 3)
    for r in range(1, 4):
        y = y_top + r * row_h
        cv2.line(vis, (0, y), (w, y), (0, 0, 255), 3)
    for c in range(1, 4):
        x = c * col_w
        cv2.line(vis, (x, y_top), (x, y_bot), (255, 0, 0), 3)

    # Etykiety odprowadzeń
    for r in range(3):
        for c in range(4):
            lead = LEADS_GRID[r][c]
            x = c * col_w + 10
            y = y_top + r * row_h + 40
            cv2.putText(vis, lead, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 128, 255), 3)

    cv2.imwrite(f"{out_dir}/split_grid.png", vis)

    # Zapisz każdy blok osobno
    blocks = split_leads(cropped, mask)
    for lead, (blk_img, blk_mask) in blocks.items():
        cv2.imwrite(f"{out_dir}/block_{lead}.png", blk_img)
        cv2.imwrite(f"{out_dir}/mask_{lead}.png", blk_mask)

    print(f"Kąt: {angle:.2f}°  y_top={y_top}  y_bot={y_bot}")
    print(f"Zapisano {len(blocks)} bloków w: {out_dir}")
    print("Sprawdź split_grid.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True)
    parser.add_argument("--out_dir", default="./debug_step3")
    args = parser.parse_args()
    run_debug(args.image, args.out_dir)