"""
============================================================
FILE    : sobel_opencv.py
PROJECT : FPGA-Based Sobel Edge Detection
PURPOSE : Software implementation of Sobel edge detection
          using Python + OpenCV. This is the REFERENCE
          implementation that runs on CPU.

WHAT IT DOES:
  1. Loads every image from software/test_images/
  2. Converts each image to grayscale
  3. Applies Gaussian blur (noise reduction)
  4. Runs Sobel Gx (horizontal gradient)
  5. Runs Sobel Gy (vertical gradient)
  6. Combines into gradient magnitude
  7. Applies threshold → binary edge map
  8. Saves all intermediate + final outputs
  9. Measures and prints latency per image

HOW TO RUN:
  python software/src/sobel_opencv.py

OUTPUT (in software/results/):
  imagename_1_gray.png      - Grayscale image
  imagename_2_blur.png      - After Gaussian blur
  imagename_3_gx.png        - Horizontal gradient
  imagename_4_gy.png        - Vertical gradient
  imagename_5_magnitude.png - Combined magnitude
  imagename_6_edges.png     - Final edge map (MAIN OUTPUT)
  imagename_panel.png       - All stages in one image
============================================================
"""

import cv2           # OpenCV — image processing library
import numpy as np   # NumPy — array math
import matplotlib    # Matplotlib — plotting/saving figures
matplotlib.use('Agg')  # Use non-interactive backend (no display window needed)
import matplotlib.pyplot as plt
import os            # File and directory operations
import time          # For latency measurement

# ============================================================
# CONFIGURATION — change these values to tune the algorithm
# ============================================================
GAUSSIAN_KERNEL = (3, 3)  # Size of blur kernel. (3,3)=light blur, (5,5)=heavier
                           # Must be odd numbers. Larger = smoother but slower
SOBEL_KSIZE     = 3       # Sobel kernel size. Always 3 for our FPGA design
THRESHOLD       = 120      # Edge detection threshold (0-255)
                           # Lower = more edges detected (more noise too)
                           # Higher = fewer edges (only strong boundaries)

# Directory paths (relative to this script location)
INPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'test_images')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
# ============================================================


def sobel_pipeline(gray):
    """
    Core Sobel edge detection pipeline.
    
    This function implements the SAME algorithm as our FPGA hardware.
    The difference is software runs these steps sequentially on CPU,
    while FPGA runs Gx and Gy in PARALLEL in the same clock cycle.
    
    Args:
        gray: Single-channel grayscale image (uint8, values 0-255)
    
    Returns:
        blur   : After Gaussian smoothing
        gx_abs : Absolute horizontal gradient |Gx|
        gy_abs : Absolute vertical gradient |Gy|
        mag    : Gradient magnitude (|Gx| + |Gy|)/2
        edges  : Binary edge map (255=edge, 0=background)
    """
    # STEP 1: Gaussian Blur — reduces noise before differentiation
    # Without this, Sobel amplifies noise (high-frequency variation)
    # Kernel (3,3) matches the smoothing in FPGA hardware design
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

    # STEP 2: Sobel Gx — detects VERTICAL edges (horizontal intensity change)
    # Kernel: [-1  0 +1]    Negative left, positive right
    #         [-2  0 +2]    Center row weighted 2x
    #         [-1  0 +1]
    # cv2.CV_64F: output as 64-bit float — IMPORTANT: preserves negative values
    # If we used CV_8U, all negative values clip to 0 → half the edges are lost
    gx = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=SOBEL_KSIZE)

    # STEP 3: Sobel Gy — detects HORIZONTAL edges (vertical intensity change)
    # Kernel: [-1 -2 -1]    Negative top, positive bottom
    #         [ 0  0  0]    Middle row ignored
    #         [+1 +2 +1]
    gy = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KSIZE)

    # STEP 4: Convert to absolute values (0-255 range)
    # convertScaleAbs does: output = |input| scaled to 0-255
    # We need this because Gx/Gy have negative values
    # FPGA equivalent: abs_gx = gx[11] ? (~gx+1) : gx
    gx_abs = cv2.convertScaleAbs(gx)
    gy_abs = cv2.convertScaleAbs(gy)

    # STEP 5: Combine Gx and Gy into gradient magnitude
    # Exact formula: G = sqrt(Gx² + Gy²)
    # Our approximation: G ≈ 0.5*|Gx| + 0.5*|Gy|
    # FPGA uses: G = |Gx| + |Gy| (L1 norm — no division needed)
    # The 0.5 weights keep the result in 0-255 range
    mag = cv2.addWeighted(gx_abs, 0.5, gy_abs, 0.5, 0)

    # STEP 6: Threshold — convert grayscale magnitude to binary edge map
    # Pixels with magnitude >= THRESHOLD → 255 (white = edge)
    # Pixels with magnitude <  THRESHOLD → 0   (black = background)
    # FPGA equivalent: edge_out = (magnitude >= THRESHOLD) ? 1 : 0
    _, edges = cv2.threshold(mag, THRESHOLD, 255, cv2.THRESH_BINARY)

    return blur, gx_abs, gy_abs, mag, edges


def process_image(image_path):
    """
    Process a single image through the full Sobel pipeline.
    Saves all intermediate results and measures latency.
    
    Args:
        image_path: Full path to input image (.jpg, .png, etc.)
    
    Returns:
        dict with keys: name, width, height, latency_ms, edges
        Returns None if image cannot be loaded
    """
    # Load image — OpenCV loads as BGR (not RGB like most tools)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load: {image_path}")
        return None

    name = os.path.splitext(os.path.basename(image_path))[0]
    h, w = img.shape[:2]
    print(f"\n[INFO] Processing: {name}  ({w}x{h} pixels)")

    # Convert BGR → Grayscale (single channel)
    # Sobel works on intensity only, not color
    # OpenCV formula: Gray = 0.114*B + 0.587*G + 0.299*R
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── LATENCY MEASUREMENT ──────────────────────────────────
    # Timer starts AFTER image load (file I/O not counted)
    # Same way FPGA latency is measured — only computation time
    t_start = time.perf_counter()
    blur, gx_abs, gy_abs, mag, edges = sobel_pipeline(gray)
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0  # Convert to milliseconds
    # ─────────────────────────────────────────────────────────

    print(f"[INFO] Latency: {latency_ms:.3f} ms  (Threshold: {THRESHOLD})")

    # Save all output stages
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_1_gray.png"),      gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_2_blur.png"),      blur)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_3_gx.png"),        gx_abs)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_4_gy.png"),        gy_abs)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_5_magnitude.png"), mag)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_6_edges.png"),     edges)

    # Create panel showing all 6 stages — useful for thesis/presentation
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f'Sobel Edge Detection Pipeline — {name}\n'
        f'Software: Python + OpenCV | Latency: {latency_ms:.2f} ms',
        fontsize=13, fontweight='bold'
    )
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Stage 0: Original Image',           None),
        (gray,   'Stage 1: Grayscale Conversion',                                    'gray'),
        (gx_abs, 'Stage 2: Sobel Gx | Vertical Edges\n(Horizontal gradient)',        'gray'),
        (gy_abs, 'Stage 3: Sobel Gy | Horizontal Edges\n(Vertical gradient)',        'gray'),
        (mag,    'Stage 4: Gradient Magnitude\n(|Gx| + |Gy|) / 2',                  'gray'),
        (edges,  f'Stage 5: Final Edge Map\n(Binary threshold = {THRESHOLD})',       'gray'),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_panel.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Results → {OUTPUT_DIR}/{name}_*.png")

    return {'name': name, 'width': w, 'height': h,
            'latency_ms': latency_ms, 'edges': edges}


def main():
    """
    Main entry point.
    Processes all images in test_images/ folder.
    Prints summary table with latency per image.
    """
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all supported image files
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(exts)
    ])

    if not images:
        print(f"\n[WARNING] No images found in: {INPUT_DIR}")
        print("[INFO] Add .jpg or .png images and rerun.")
        return []

    print(f"\n{'='*55}")
    print(f"  SOFTWARE SOBEL — {len(images)} image(s) to process")
    print(f"{'='*55}")

    results = [process_image(p) for p in images]
    results = [r for r in results if r]  # Remove failed ones

    # Print summary table
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Image':<25} {'Resolution':<14} {'Latency'}")
    print(f"  {'-'*50}")
    for r in results:
        res = f"{r['width']}x{r['height']}"
        print(f"  {r['name']:<25} {res:<14} {r['latency_ms']:.3f} ms")

    if results:
        avg = sum(r['latency_ms'] for r in results) / len(results)
        print(f"\n  Average latency: {avg:.3f} ms")
        print(f"  FPGA target    : 2-10 ms")
        print(f"  Expected speedup: {avg/10:.1f}x - {avg/2:.1f}x")

    return results


if __name__ == '__main__':
    main()
