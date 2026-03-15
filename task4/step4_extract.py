"""
KROK 4: Ekstrakcja sygnału 1D z bloku maski.

Dla każdej kolumny x: znajdź medianę Y białych pikseli → raw_y[x]
Interpoluj braki → odwróć oś Y → odejmij baseline → skaluj px→mV → resample→500Hz

Test:
  python step4_extract.py --image SCIEZKA/ecg_train_XXXX.png --out_dir ./debug4
"""

import os
import argparse
import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
import sys
sys.path.insert(0, '.')
from step1_deskew import deskew
from step2_binarize import binarize
from step3_split import crop_card, split_leads, LEADS_GRID

FS               = 500.0
STRIP_DURATION_S = 2.5
SAMPLES_PER_LEAD = int(STRIP_DURATION_S * FS)   # 1250
GAIN_MM_MV       = 10.0   # 10mm = 1mV
PAPER_SPEED      = 25.0   # mm/s


def detect_pixels_per_mm(img: np.ndarray) -> float:
    """
    Wykrywa pixels_per_mm z siatki EKG przez FFT projekcji wierszy.
    Fallback: 10.0 px/mm.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    roi = gray[h//5: 4*h//5, w//5: 4*w//5]
    rh, rw = roi.shape

    # Odejmij tło
    blur = cv2.GaussianBlur(roi, (0, 0), rw // 10)
    norm = roi - blur

    # FFT na projekcji kolumn (pionowe linie siatki)
    col_proj = norm.mean(axis=0)
    col_proj -= col_proj.mean()
    fft_mag = np.abs(np.fft.rfft(col_proj))
    freqs   = np.fft.rfftfreq(len(col_proj))

    # Szukamy dominującego spacingu między 3 a 50 px
    valid = (freqs >= 1/50) & (freqs <= 1/3)
    if not np.any(valid):
        return 10.0

    fft_valid = fft_mag.copy()
    fft_valid[~valid] = 0
    peak_freq = freqs[np.argmax(fft_valid)]
    if peak_freq <= 0:
        return 10.0

    spacing_px = 1.0 / peak_freq
    if spacing_px < 3 or spacing_px > 50:
        return 10.0

    return float(spacing_px)


def find_signal_start_col(mask_block: np.ndarray) -> int:
    """
    Znajduje pierwszą kolumnę gdzie zaczyna się sygnał.
    Ignoruje pojedyncze piksele szumu.
    """
    col_sum = mask_block.sum(axis=0).astype(float)
    col_sum_smooth = uniform_filter1d(col_sum, size=5)
    threshold = max(col_sum_smooth.max() * 0.01, 1.0)
    candidates = np.where(col_sum_smooth > threshold)[0]
    if len(candidates) == 0:
        return 0
    return int(candidates[0])


def extract_signal(mask_block: np.ndarray,
                   pixels_per_mm: float) -> np.ndarray:
    """
    Ekstrahuje sygnał 1D z bloku maski binarnej.
    Zwraca sygnał w mV, długość = SAMPLES_PER_LEAD.
    """
    height, width = mask_block.shape

    # Odetnij lewy margines przed sygnałem
    start_col = find_signal_start_col(mask_block)
    mask_block = mask_block[:, start_col:]
    width = mask_block.shape[1]

    # ── Kolumna po kolumnie ──
    raw_y = np.full(width, np.nan)
    for x in range(width):
        col = mask_block[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            raw_y[x] = float(np.median(ys))

    # ── Interpolacja brakujących wartości ──
    nan_mask = np.isnan(raw_y)
    if np.all(nan_mask):
        return np.zeros(SAMPLES_PER_LEAD, dtype=np.float16)

    if np.any(nan_mask):
        valid_x = np.where(~nan_mask)[0]
        valid_y = raw_y[~nan_mask]
        f = interpolate.interp1d(valid_x, valid_y,
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value=(valid_y[0], valid_y[-1]))
        raw_y[nan_mask] = f(np.where(nan_mask)[0])

    # ── Odwróć oś Y (obraz: 0=góra, sygnał: 0=dół) ──
    raw_y = height - raw_y

    # ── Usuń baseline (mediana — odporna na piki QRS) ──
    baseline = np.median(raw_y)
    raw_y = raw_y - baseline

    # ── Konwersja px → mV ──
    pixels_per_mv = pixels_per_mm * GAIN_MM_MV
    signal_mv = raw_y / pixels_per_mv

    # ── Resample → dokładnie SAMPLES_PER_LEAD próbek ──
    if len(signal_mv) != SAMPLES_PER_LEAD:
        x_orig   = np.linspace(0, 1, len(signal_mv))
        x_target = np.linspace(0, 1, SAMPLES_PER_LEAD)
        f_t = interpolate.interp1d(x_orig, signal_mv,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=(signal_mv[0], signal_mv[-1]))
        signal_mv = f_t(x_target)

    return signal_mv.astype(np.float16)


def process_image(img_path: str) -> dict[str, np.ndarray]:
    """
    Full pipeline dla jednego obrazu.
    Zwraca {lead_name: signal_float16}.
    """
    img = cv2.imread(img_path)
    if img is None:
        return {}

    record_name = os.path.splitext(os.path.basename(img_path))[0]

    rotated   = deskew(img)[0]
    cropped   = crop_card(rotated)
    mask      = binarize(cropped)
    px_per_mm = detect_pixels_per_mm(cropped)
    blocks    = split_leads(cropped, mask)

    results = {}
    for lead, (_, mask_block) in blocks.items():
        if lead == "II_RHYTHM":
            continue
        sig = extract_signal(mask_block, px_per_mm)
        results[f"{record_name}_{lead}"] = sig

    return results


# ══════════════════════════════════════════════════════
# DIAGNOSTYKA
# ══════════════════════════════════════════════════════

def run_debug(img_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    assert img is not None

    rotated   = deskew(img, debug=True)[0]
    cropped   = crop_card(rotated)
    mask      = binarize(cropped)
    px_per_mm = detect_pixels_per_mm(cropped)
    blocks    = split_leads(cropped, mask)

    print(f"pixels_per_mm = {px_per_mm:.2f}  →  pixels_per_mV = {px_per_mm * GAIN_MM_MV:.2f}")

    # Wykreśl wszystkie 12 sygnałów
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    for r in range(3):
        for c in range(4):
            lead = LEADS_GRID[r][c]
            ax = axes[r][c]
            if lead not in blocks:
                ax.set_title(f"{lead} BRAK")
                continue

            _, mask_block = blocks[lead]
            sig = extract_signal(mask_block, px_per_mm).astype(np.float64)
            t = np.arange(len(sig)) / FS
            ax.plot(t, sig, lw=0.8)
            ax.set_title(f"{lead}  (px/mm={px_per_mm:.1f})")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("mV")
            ax.grid(True, alpha=0.3)

    plt.suptitle(os.path.basename(img_path))
    plt.tight_layout()
    out_path = f"{out_dir}/signals.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Zapisano: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True)
    parser.add_argument("--out_dir", default="./debug_step4")
    args = parser.parse_args()
    run_debug(args.image, args.out_dir)