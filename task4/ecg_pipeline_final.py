"""
ECG Digitization Pipeline - Final
==================================
Uruchomienie:
  python ecg_pipeline_final.py --mode test    # generuje submission.npz
  python ecg_pipeline_final.py --mode eval    # ocenia na zbiorze treningowym
"""

import os, glob, argparse
import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    def progress(it, **kw): return tqdm(it, **kw)
except ImportError:
    def progress(it, **kw): return it

try:
    import wfdb
    WFDB_OK = True
except ImportError:
    WFDB_OK = False

SCRATCH    = os.environ.get("SCRATCH", ".")
DATA_BASE  = os.path.join(SCRATCH, "ensembleAI-26/task4/data/Task 4 public")
DATA_TRAIN = os.path.join(DATA_BASE, "train")
DATA_TEST  = os.path.join(DATA_BASE, "test")
OUTPUT_NPZ = os.path.join(SCRATCH, "data/out/submission.npz")

FS               = 500.0
STRIP_DURATION_S = 2.5
SAMPLES_PER_LEAD = int(STRIP_DURATION_S * FS)  # 1250
GAIN_MM_MV       = 10.0

LEADS_GRID = [
    ["I",   "AVR", "V1", "V4"],
    ["II",  "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]
ALL_LEADS = [l for row in LEADS_GRID for l in row]
WORKERS = max(1, (os.cpu_count() or 4) - 2)


# ═══════════════════════════════════════
# KROK 1: DESKEW
# ═══════════════════════════════════════

def deskew(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    median_s = float(np.median(hsv[h//3:2*h//3, w//3:2*w//3, 1]))

    if median_s > 40:
        angle = _angle_color_grid(img, hsv)
    else:
        angle = _angle_row_projection(img)

    if angle is None or abs(angle) < 0.15:
        return img

    angle = float(np.clip(angle, -45, 45))
    return _rotate(img, angle)


def _angle_color_grid(img, hsv):
    S, V = hsv[:,:,1], hsv[:,:,2]
    h, w = img.shape[:2]
    mask = ((S > 40) & (V > 60)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    coverage = np.sum(mask > 0) / mask.size
    if coverage < 0.05 or coverage > 0.95:
        return _angle_row_projection(img)
    lines = cv2.HoughLinesP(mask, 1, np.pi/720, 80,
                             minLineLength=w//8, maxLineGap=30)
    if lines is None:
        return _angle_row_projection(img)
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2-x1, y2-y1
        if abs(dx) < 1: continue
        a = float(np.degrees(np.arctan2(dy, dx)))
        if abs(a) <= 20:
            angles.append(a)
    if not angles:
        return _angle_row_projection(img)
    angles = np.array(angles)
    median = float(np.median(angles))
    inliers = angles[np.abs(angles - median) < 3.0]
    return float(np.median(inliers))


def _angle_row_projection(img, angle_range=10.0, n_steps=81):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    roi = gray[h//5: 4*h//5, w//5: 4*w//5]
    rh, rw = roi.shape
    blur = cv2.GaussianBlur(roi, (0, 0), rw // 10)
    norm = roi - blur
    cx, cy = rw // 2, rh // 2
    best_angle, best_var = 0.0, -1.0
    for angle in np.linspace(-angle_range, angle_range, n_steps):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(norm, M, (rw, rh),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        var = float(np.var(rotated.mean(axis=1)))
        if var > best_var:
            best_var = var
            best_angle = float(angle)
    return best_angle


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════
# KROK 2: BINARYZACJA
# ═══════════════════════════════════════

def binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray < 50).astype(np.uint8) * 255


# ═══════════════════════════════════════
# KROK 3: CROP + PODZIAŁ NA BLOKI
# ═══════════════════════════════════════

def crop_card(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, cw, ch = cv2.boundingRect(coords)
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    cw = min(img.shape[1] - x, cw + 2*margin)
    ch = min(img.shape[0] - y, ch + 2*margin)
    return img[y:y+ch, x:x+cw]


def find_signal_bounds(mask: np.ndarray) -> tuple[int, int]:
    h = mask.shape[0]
    row_sum = mask.sum(axis=1) / 255
    smoothed = uniform_filter1d(row_sum.astype(float), size=max(1, h//40))
    threshold = smoothed.max() * 0.05
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
                mask: np.ndarray) -> dict:
    h, w = mask.shape
    y_top, y_bot = find_signal_bounds(mask)
    signal_h = y_bot - y_top
    row_h = signal_h // 4
    col_w = w // 4
    blocks = {}
    for r in range(3):
        for c in range(4):
            lead = LEADS_GRID[r][c]
            y1 = y_top + r * row_h
            y2 = y_top + (r + 1) * row_h
            x1, x2 = c * col_w, (c + 1) * col_w
            blocks[lead] = (img[y1:y2, x1:x2], mask[y1:y2, x1:x2])
    return blocks


# ═══════════════════════════════════════
# KROK 4: DETEKCJA SKALI
# ═══════════════════════════════════════

def detect_pixels_per_mm(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    roi = gray[h//5: 4*h//5, w//5: 4*w//5]
    rh, rw = roi.shape
    blur = cv2.GaussianBlur(roi, (0, 0), rw // 10)
    norm = roi - blur
    col_proj = norm.mean(axis=0)
    col_proj -= col_proj.mean()
    fft_mag = np.abs(np.fft.rfft(col_proj))
    freqs   = np.fft.rfftfreq(len(col_proj))
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


# ═══════════════════════════════════════
# KROK 5: EKSTRAKCJA SYGNAŁU
# ═══════════════════════════════════════

def find_calib_end(mask_block: np.ndarray) -> int:
    """
    Wykrywa koniec impulsu kalibracyjnego (prostokąt na początku bloku).
    Zwraca indeks kolumny gdzie kończy się impuls kalibracyjny.
    """
    col_sum = mask_block.sum(axis=0).astype(float)
    smooth = uniform_filter1d(col_sum, size=3)
    threshold = max(smooth.max() * 0.05, 1.0)
    active = smooth > threshold
    w = len(active)

    # Szukaj w pierwszych 30% bloku
    search_end = w // 3
    in_region = False
    for x in range(search_end):
        if active[x]:
            in_region = True
        elif in_region:
            # Sprawdz czy przerwa jest wystarczajaco dluga (>=3px)
            gap = 0
            for xx in range(x, min(x+10, w)):
                if not active[xx]:
                    gap += 1
                else:
                    break
            if gap >= 3:
                return x
            # Jesli przerwa krotka - kontynuuj (to nie koniec impulsu)
            in_region = True

    return 0  # brak impulsu kalibracyjnego


def extract_signal(mask_block: np.ndarray,
                   pixels_per_mm: float) -> np.ndarray:
    height, width = mask_block.shape

    # Wykryj i pomiń impuls kalibracyjny
    calib_end = find_calib_end(mask_block)
    if calib_end > 0:
        mask_block = mask_block[:, calib_end:]
        width = mask_block.shape[1]

    if width == 0:
        return np.zeros(SAMPLES_PER_LEAD, dtype=np.float16)

    # Kolumna po kolumnie
    raw_y = np.full(width, np.nan)
    for x in range(width):
        col = mask_block[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            raw_y[x] = float(np.median(ys))

    # Interpolacja NaN
    nan_mask = np.isnan(raw_y)
    if np.all(nan_mask):
        return np.zeros(SAMPLES_PER_LEAD, dtype=np.float16)
    if np.any(nan_mask):
        valid_x = np.where(~nan_mask)[0]
        valid_y = raw_y[~nan_mask]
        f = interpolate.interp1d(valid_x, valid_y, kind='linear',
                                  bounds_error=False,
                                  fill_value=(valid_y[0], valid_y[-1]))
        raw_y[nan_mask] = f(np.where(nan_mask)[0])

    # Odwróć oś Y
    raw_y = height - raw_y

    # Usuń baseline
    baseline = np.median(raw_y)
    raw_y = raw_y - baseline

    # px → mV
    pixels_per_mv = pixels_per_mm * GAIN_MM_MV
    signal_mv = raw_y / pixels_per_mv

    # Resample → 1250 próbek
    if len(signal_mv) != SAMPLES_PER_LEAD:
        x_orig   = np.linspace(0, 1, len(signal_mv))
        x_target = np.linspace(0, 1, SAMPLES_PER_LEAD)
        f_t = interpolate.interp1d(x_orig, signal_mv, kind='linear',
                                    bounds_error=False,
                                    fill_value=(signal_mv[0], signal_mv[-1]))
        signal_mv = f_t(x_target)

    return signal_mv.astype(np.float16)


# ═══════════════════════════════════════
# GŁÓWNA FUNKCJA
# ═══════════════════════════════════════

def process_image(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return {}
    record_name = Path(img_path).stem

    rotated   = deskew(img)
    cropped   = crop_card(rotated)
    mask      = binarize(cropped)
    px_per_mm = detect_pixels_per_mm(cropped)
    blocks    = split_leads(cropped, mask)

    results = {}
    for lead, (_, mask_block) in blocks.items():
        sig = extract_signal(mask_block, px_per_mm)
        results[f"{record_name}_{lead}"] = sig

    return results


# ═══════════════════════════════════════
# EWALUACJA NA TRAIN
# ═══════════════════════════════════════

def evaluate(data_dir: str, n: int = 50):
    if not WFDB_OK:
        print("Zainstaluj wfdb: pip install wfdb")
        return
    from scipy.stats import pearsonr

    heas = glob.glob(os.path.join(data_dir, "*.hea"))
    np.random.shuffle(heas)
    heas = heas[:n]

    scores_per_lead = {l: [] for l in ALL_LEADS}
    scores_all = []

    for hea in progress(heas, desc="Eval"):
        record = Path(hea).stem
        imgs = glob.glob(os.path.join(data_dir, record + ".*"))
        imgs = [p for p in imgs if p.lower().endswith((".png",".jpg",".jpeg"))]
        if not imgs:
            continue
        try:
            preds = process_image(imgs[0])
            signals, fields = wfdb.rdsamp(os.path.join(data_dir, record))
            gt = {n.upper(): signals[:, i] for i, n in enumerate(fields['sig_name'])}
        except Exception as e:
            continue

        for lead in ALL_LEADS:
            key = f"{record}_{lead}"
            if key not in preds or lead not in gt:
                continue
            pred = preds[key].astype(np.float64)
            true = gt[lead][:SAMPLES_PER_LEAD]
            if len(true) < SAMPLES_PER_LEAD:
                continue
            r, _ = pearsonr(pred, true)
            scores_per_lead[lead].append(r)
            scores_all.append(r)

    print("\n═══ Wyniki (Pearson r) ═══")
    for lead in ALL_LEADS:
        vals = scores_per_lead[lead]
        if vals:
            print(f"  {lead:>5}: {np.mean(vals):.3f}  (n={len(vals)})")
    if scores_all:
        print(f"\n  ŚREDNIA: {np.mean(scores_all):.3f}  (n={len(scores_all)})")


# ═══════════════════════════════════════
# GENEROWANIE SUBMISJI
# ═══════════════════════════════════════

def generate_submission(data_dir: str, output_path: str):
    imgs = (glob.glob(os.path.join(data_dir, "*.jpg")) +
            glob.glob(os.path.join(data_dir, "*.png")) +
            glob.glob(os.path.join(data_dir, "*.jpeg")))
    if not imgs:
        print(f"Brak obrazów w: {data_dir}")
        return

    print(f"Przetwarzam {len(imgs)} obrazów...")
    submission = {}

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_image, p): p for p in imgs}
        for fut in progress(as_completed(futures), total=len(imgs), desc="Test"):
            try:
                submission.update(fut.result())
            except Exception as e:
                print(f"ERR {futures[fut]}: {e}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **submission)
    print(f"\n✅ Zapisano: {output_path}  ({len(submission)} kluczy)")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test","eval","both"], default="both")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--output", default=OUTPUT_NPZ)
    args = parser.parse_args()

    if args.mode in ("eval", "both"):
        evaluate(DATA_TRAIN, n=args.n_eval)

    if args.mode in ("test", "both"):
        generate_submission(DATA_TEST, args.output)