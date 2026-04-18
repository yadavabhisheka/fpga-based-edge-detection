"""
sobel_opencv.py
---------------
Software implementation of Sobel edge detection using Python + OpenCV.
This is the REFERENCE implementation that will be compared against
the FPGA hardware implementation for latency and output quality.

Project: FPGA-Based Real-Time Edge Detection for Autonomous Vehicles
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ─────────────────────────────────────────────
#  CONFIGURATION — edit these as needed
# ─────────────────────────────────────────────
GAUSSIAN_KERNEL_SIZE = (3, 3)   # Noise reduction kernel. Try (5,5) for noisier images.
GAUSSIAN_SIGMA       = 0        # 0 = OpenCV auto-calculates sigma from kernel size
SOBEL_KERNEL_SIZE    = 3        # Must be 1, 3, 5, or 7. Use 3 to match FPGA design.
THRESHOLD_VALUE      = 80       # 0-255. Pixels above this = edge (white). Tune per image.
APPLY_THRESHOLD      = True     # True = binary edge map. False = grayscale magnitude.

INPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'test_images')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
# ─────────────────────────────────────────────


def sobel_edge_detection(image_path: str, save_output: bool = True) -> dict:
    """
    Run full Sobel edge detection pipeline on a single image.

    Pipeline:
        Load → Grayscale → Gaussian Blur → Sobel Gx → Sobel Gy
        → Magnitude → (Threshold) → Save

    Args:
        image_path : Full path to input image file
        save_output: If True, saves all intermediate and final images to OUTPUT_DIR

    Returns:
        dict with keys:
            'image_name'  : str
            'resolution'  : tuple (H, W)
            'gray'        : np.ndarray — grayscale image
            'blur'        : np.ndarray — blurred grayscale
            'gx_abs'      : np.ndarray — absolute horizontal gradient
            'gy_abs'      : np.ndarray — absolute vertical gradient
            'magnitude'   : np.ndarray — combined gradient magnitude
            'edges'       : np.ndarray — final edge map (thresholded or magnitude)
            'latency_ms'  : float — total processing time in milliseconds
            'error'       : str or None
    """

    result = {
        'image_name': os.path.basename(image_path),
        'resolution': None,
        'gray': None, 'blur': None,
        'gx_abs': None, 'gy_abs': None,
        'magnitude': None, 'edges': None,
        'latency_ms': 0.0,
        'error': None
    }

    # ── STEP 1: Load Image ───────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        result['error'] = f"Could not load image: {image_path}"
        print(f"[ERROR] {result['error']}")
        return result

    result['resolution'] = (img.shape[0], img.shape[1])
    print(f"\n[INFO] Processing: {result['image_name']}  |  Resolution: {img.shape[1]}x{img.shape[0]}")

    # ── START TIMER ──────────────────────────────────────────────────────────
    # Time ONLY the core processing — exclude file I/O to match FPGA comparison
    t_start = time.perf_counter()

    # ── STEP 2: Convert to Grayscale ─────────────────────────────────────────
    # OpenCV loads as BGR. Convert to single-channel intensity.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── STEP 3: Gaussian Blur (Noise Reduction) ───────────────────────────────
    # Smooths high-frequency noise before differentiation.
    # Sobel is a derivative operator — noise gets amplified without pre-blur.
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

    # ── STEP 4: Sobel Gx — Horizontal Gradient ───────────────────────────────
    # Detects vertical edges (transitions in horizontal direction)
    # Kernel: [-1 0 +1 / -2 0 +2 / -1 0 +1]
    # cv2.CV_64F: output as 64-bit float to preserve negative values.
    # If we used CV_8U, negative values would be clipped to 0 (losing half the edges).
    gx = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=SOBEL_KERNEL_SIZE)

    # ── STEP 5: Sobel Gy — Vertical Gradient ─────────────────────────────────
    # Detects horizontal edges (transitions in vertical direction)
    # Kernel: [-1 -2 -1 / 0 0 0 / +1 +2 +1]
    gy = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KERNEL_SIZE)

    # ── STEP 6: Convert to Absolute 8-bit ────────────────────────────────────
    # convertScaleAbs: takes absolute value, then scales to 0-255.
    # This gives us |Gx| and |Gy| as uint8 arrays.
    gx_abs = cv2.convertScaleAbs(gx)
    gy_abs = cv2.convertScaleAbs(gy)

    # ── STEP 7: Gradient Magnitude ────────────────────────────────────────────
    # Exact:  G = sqrt(Gx^2 + Gy^2)          ← what OpenCV uses internally
    # Approx: G = |Gx| + |Gy|                ← what our FPGA uses (L1 norm)
    #
    # We use addWeighted to replicate L1 approximation with 0.5 weights,
    # keeping the output in 0-255 range.
    magnitude = cv2.addWeighted(gx_abs, 0.5, gy_abs, 0.5, 0)

    # ── STEP 8: (Optional) Threshold ─────────────────────────────────────────
    # Converts grayscale gradient map → binary edge map (black/white).
    # This matches the FPGA output which is also binary.
    if APPLY_THRESHOLD:
        _, edges = cv2.threshold(magnitude, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    else:
        edges = magnitude.copy()

    # ── STOP TIMER ───────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    result.update({
        'gray': gray, 'blur': blur,
        'gx_abs': gx_abs, 'gy_abs': gy_abs,
        'magnitude': magnitude, 'edges': edges,
        'latency_ms': latency_ms
    })

    print(f"[INFO] Latency: {latency_ms:.3f} ms  |  Threshold: {THRESHOLD_VALUE if APPLY_THRESHOLD else 'None'}")

    # ── STEP 9: Save Outputs ─────────────────────────────────────────────────
    if save_output:
        _save_results(image_path, img, gray, gx_abs, gy_abs, magnitude, edges)

    return result


def _save_results(image_path, img, gray, gx_abs, gy_abs, magnitude, edges):
    """Save all intermediate + final outputs to results directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_1_gray.png"),      gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_2_blur.png"),
                cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_3_gx.png"),        gx_abs)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_4_gy.png"),        gy_abs)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_5_magnitude.png"), magnitude)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_6_edges.png"),     edges)

    # Save a combined visualization panel
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'Sobel Edge Detection — {os.path.basename(image_path)}', fontsize=14, fontweight='bold')

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Original (RGB)', None),
        (gray,      'Grayscale',       'gray'),
        (gx_abs,    'Sobel Gx (|Horizontal Gradient|)', 'gray'),
        (gy_abs,    'Sobel Gy (|Vertical Gradient|)',   'gray'),
        (magnitude, 'Gradient Magnitude', 'gray'),
        (edges,     f'Final Edge Map (T={THRESHOLD_VALUE})', 'gray'),
    ]

    for ax, (data, title, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base}_panel.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Results → {OUTPUT_DIR}/{base}_*.png")


def process_all_images() -> list:
    """
    Process every image in the test_images directory.
    Returns list of result dicts for all images.
    """
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if f.lower().endswith(supported)
    ]

    if not images:
        print(f"\n[WARNING] No images found in: {INPUT_DIR}")
        print("[INFO] Add .jpg / .png images to software/test_images/ and rerun.")
        return []

    print(f"\n{'='*60}")
    print(f"  Found {len(images)} image(s) to process")
    print(f"{'='*60}")

    all_results = []
    for path in images:
        result = sobel_edge_detection(path, save_output=True)
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Image':<30} {'Resolution':<16} {'Latency (ms)'}")
    print(f"  {'-'*55}")
    for r in all_results:
        if r['error'] is None:
            h, w = r['resolution']
            print(f"  {r['image_name']:<30} {str(w)+'x'+str(h):<16} {r['latency_ms']:.3f} ms")

    if all_results:
        valid = [r for r in all_results if r['error'] is None]
        if valid:
            avg = sum(r['latency_ms'] for r in valid) / len(valid)
            print(f"\n  Average latency: {avg:.3f} ms  over {len(valid)} image(s)")

    return all_results


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    process_all_images()
