"""
KROK 1 v3: Wykrycie kąta rotacji i wyprostowanie obrazu EKG.

Strategia:
  - S > 40: kolorowa siatka → HSV mask + HoughLinesP
  - S <= 40: szara/biała siatka → projekcja wierszy (float subtract!)

Test:
  python step1_deskew.py --image SCIEZKA/ecg_train_XXXX.png --out_dir ./debug1
"""

import os
import argparse
import numpy as np
import cv2


def deskew(img: np.ndarray, debug: bool = False) -> tuple[np.ndarray, float]:
    """Zwraca (wyprostowany obraz, kąt korekcji w stopniach)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    roi_s = hsv[h//3:2*h//3, w//3:2*w//3, 1]
    median_s = float(np.median(roi_s))

    if debug:
        print(f"  Mediana saturacji ROI: {median_s:.1f}")

    if median_s > 40:
        angle = _angle_color_grid(img, hsv, debug=debug)
        method = "color+hough"
    else:
        angle = _angle_row_projection(img, debug=debug)
        method = "row_projection"

    if angle is None or abs(angle) < 0.15:
        if debug:
            print(f"  [{method}] Brak korekcji")
        return img, 0.0

    angle = float(np.clip(angle, -45, 45))
    if debug:
        print(f"  [{method}] Wykryty kąt: {angle:.2f}°")

    rotated = _rotate(img, angle)
    return rotated, angle


def _angle_color_grid(img, hsv, debug=False):
    S, V = hsv[:,:,1], hsv[:,:,2]
    h, w = img.shape[:2]

    mask = ((S > 40) & (V > 60)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    coverage = np.sum(mask > 0) / mask.size
    if debug:
        print(f"  Pokrycie maski koloru: {coverage:.3f}")

    if coverage < 0.05 or coverage > 0.95:
        return _angle_row_projection(img, debug=debug)

    lines = cv2.HoughLinesP(mask, 1, np.pi/720, 80,
                             minLineLength=w//8, maxLineGap=30)
    if lines is None:
        return _angle_row_projection(img, debug=debug)

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2-x1, y2-y1
        if abs(dx) < 1:
            continue
        a = float(np.degrees(np.arctan2(dy, dx)))
        if abs(a) <= 20:
            angles.append(a)

    if not angles:
        return _angle_row_projection(img, debug=debug)

    angles = np.array(angles)
    median = float(np.median(angles))
    inliers = angles[np.abs(angles - median) < 3.0]
    return float(np.median(inliers))


def _angle_row_projection(img, angle_range=10.0, n_steps=81, debug=False):
    """
    Szuka kąta przy którym wariancja projekcji wierszy jest MAKSYMALNA.
    KLUCZOWE: odejmowanie tła w float32 (nie cv2.subtract które clipuje uint8).
    """
    # Float32 żeby uniknąć clippingu przy odejmowaniu tła
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    roi = gray[h//5: 4*h//5, w//5: 4*w//5]
    rh, rw = roi.shape

    # Odejmij rozmyte tło — uwydatnia linie siatki
    blur = cv2.GaussianBlur(roi, (0, 0), rw // 10)
    norm = roi - blur   # float subtract, bez clippingu

    cx, cy = rw // 2, rh // 2
    best_angle = 0.0
    best_var   = -1.0

    for angle in np.linspace(-angle_range, angle_range, n_steps):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(norm, M, (rw, rh),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        var = float(np.var(rotated.mean(axis=1)))
        if var > best_var:
            best_var   = var
            best_angle = float(angle)

    if debug:
        print(f"  row_projection: best_angle={best_angle:.2f}°  var={best_var:.4f}")

    return best_angle


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def run_debug(img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    assert img is not None, f"Nie można wczytać: {img_path}"
    h, w = img.shape[:2]
    print(f"Obraz: {w}x{h}  ({os.path.basename(img_path)})")

    cv2.imwrite(f"{out_dir}/01_original.png", img)
    rotated, angle = deskew(img, debug=True)
    cv2.imwrite(f"{out_dir}/02_deskewed.png", rotated)

    for fname, image in [("01_original_ref.png", img),
                          ("02_deskewed_ref.png", rotated)]:
        ref = image.copy()
        cv2.line(ref, (w//10, h//2), (9*w//10, h//2), (0, 255, 0), 3)
        cv2.imwrite(f"{out_dir}/{fname}", ref)

    print(f"Wykryty kąt: {angle:.2f}°")
    print(f"Sprawdź: {out_dir}/02_deskewed_ref.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True)
    parser.add_argument("--out_dir", default="./debug_step1")
    args = parser.parse_args()
    run_debug(args.image, args.out_dir)