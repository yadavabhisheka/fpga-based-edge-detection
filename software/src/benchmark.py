"""
============================================================
FILE    : benchmark.py
PROJECT : FPGA-Based Sobel Edge Detection
PURPOSE : Precise latency measurement for software Sobel.
          Runs each image 50 times to get stable numbers.
          Exports CSV and bar chart for thesis.

WHY 50 RUNS?
  Single run latency is unstable due to:
  - CPU cache cold start (first run always slow)
  - OS scheduling interruptions
  - Memory page faults
  50 runs + 5 warmup gives statistically stable mean/median

HOW TO RUN:
  python software/src/benchmark.py

OUTPUT (in software/results/):
  software_benchmark.csv        - Latency stats per image
  software_latency_chart.png    - Bar chart (use in thesis)
============================================================
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import statistics

# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================
N_RUNS    = 50   # Number of timed runs per image
WARMUP    = 5    # Warmup runs (discarded — lets CPU cache settle)
THRESHOLD = 80   # Must match sobel_opencv.py threshold

# Directory paths
INPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'test_images')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
# ============================================================


def run_pipeline(gray):
    """
    Minimal core pipeline for benchmarking.
    No file I/O, no display — pure computation only.
    This isolates exactly what the FPGA replaces.
    
    Args:
        gray: Grayscale image (uint8)
    Returns:
        edges: Binary edge map
    """
    # Gaussian blur — removes noise before differentiation
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel gradients — detect edges in both directions
    # CV_64F preserves negative values (essential for correct results)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # Take absolute values and convert to 0-255
    gx_abs = cv2.convertScaleAbs(gx)
    gy_abs = cv2.convertScaleAbs(gy)

    # Combine gradients — approximates L1 magnitude
    mag = cv2.addWeighted(gx_abs, 0.5, gy_abs, 0.5, 0)

    # Binary threshold — pixels above THRESHOLD become edges
    _, edges = cv2.threshold(mag, THRESHOLD, 255, cv2.THRESH_BINARY)
    return edges


def benchmark_image(path):
    """
    Benchmark one image over N_RUNS iterations.
    Returns dict with mean, median, std, min, max latency.
    
    Args:
        path: Full path to image file
    Returns:
        dict with latency statistics, or None if load fails
    """
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Cannot load: {path}")
        return None

    # Convert to grayscale once — not part of benchmark timing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    name = os.path.basename(path)
    print(f"\n[BENCH] {name}  ({w}x{h} = {w*h:,} pixels)  {N_RUNS} runs")

    # ── WARMUP RUNS (discarded) ──────────────────────────────
    # First few runs are slower because:
    # 1. OpenCV functions load into CPU instruction cache
    # 2. Image data loads into CPU data cache
    # 3. Python JIT effects settle
    # We discard these to get the "steady state" performance
    for _ in range(WARMUP):
        run_pipeline(gray)

    # ── TIMED RUNS ───────────────────────────────────────────
    # time.perf_counter() is the most precise Python timer
    # Resolution: nanoseconds on modern systems
    # Better than time.time() which uses OS wall clock (can jump)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        run_pipeline(gray)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert seconds to milliseconds

    # Calculate statistics
    mean_ms   = statistics.mean(times)
    median_ms = statistics.median(times)
    std_ms    = statistics.stdev(times)
    min_ms    = min(times)
    max_ms    = max(times)

    print(f"  Mean:{mean_ms:.3f}ms  Median:{median_ms:.3f}ms  "
          f"Std:{std_ms:.3f}ms  Min:{min_ms:.3f}ms  Max:{max_ms:.3f}ms")

    # Return results dict (used for CSV export and chart)
    return {
        'image_name': name,
        'width': w, 'height': h,
        'pixels': w * h,
        'n_runs': N_RUNS,
        'mean_ms': round(mean_ms, 4),    # Primary metric for thesis
        'median_ms': round(median_ms, 4),
        'stdev_ms': round(std_ms, 4),    # Shows measurement stability
        'min_ms': round(min_ms, 4),
        'max_ms': round(max_ms, 4),
        'fps': round(1000.0 / mean_ms, 1)  # Frames per second
    }


def save_csv(records):
    """Save benchmark results to CSV file."""
    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, 'software_benchmark.csv')
    df.to_csv(path, index=False)
    print(f"\n[SAVED] CSV → {path}")
    return df


def save_chart(records):
    """
    Create bar chart showing latency per image.
    Includes FPGA target range lines for comparison.
    This chart goes directly into your thesis results section.
    """
    names   = [r['image_name'].rsplit('.', 1)[0] for r in records]
    means   = [r['mean_ms']   for r in records]
    stds    = [r['stdev_ms']  for r in records]
    resols  = [f"{r['width']}x{r['height']}" for r in records]

    fig, ax = plt.subplots(figsize=(max(10, len(records) * 2), 6))

    # Draw bars with error bars showing standard deviation
    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color='#2C5F8A', edgecolor='#1A3A5C', linewidth=1.2,
                  error_kw=dict(ecolor='#C4860A', lw=2),
                  label='SW Mean Latency')

    # Annotate each bar with latency and resolution
    for bar, m, r in zip(bars, means, resols):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.15,
                f'{m:.1f} ms\n{r}',
                ha='center', va='bottom', fontsize=8, color='#1A3A5C')

    # FPGA target range lines — shows what hardware should achieve
    ax.axhline(10, color='#1A6B1A', ls='--', lw=1.5,
               label='FPGA Target Max (10 ms)')
    ax.axhline(2,  color='#C4860A', ls='--', lw=1.5,
               label='FPGA Target Min (2 ms)')
    ax.axhspan(2, 10, alpha=0.08, color='#1A6B1A')  # Green target zone

    ax.set_xlabel('Test Image', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(
        f'Software Sobel — Latency Benchmark\n'
        f'(Python + OpenCV | {N_RUNS} runs each | Gaussian 3×3 | Threshold {THRESHOLD})',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'software_latency_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Chart → {path}")


def main():
    """
    Main entry point.
    Benchmarks all images in test_images/ folder.
    Prints summary and saves CSV + chart.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(exts)
    ])

    if not images:
        print(f"[WARNING] No images in {INPUT_DIR}")
        return

    print(f"\n{'='*55}")
    print(f"  SOFTWARE BENCHMARK — {len(images)} image(s)")
    print(f"  {N_RUNS} runs each | {WARMUP} warmup runs discarded")
    print(f"{'='*55}")

    records = [benchmark_image(p) for p in images]
    records = [r for r in records if r]

    if not records:
        return

    # Print final summary table
    avg = statistics.mean([r['mean_ms'] for r in records])
    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Image':<25} {'Res':<12} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    for r in records:
        res = f"{r['width']}x{r['height']}"
        print(f"  {r['image_name']:<25} {res:<12} "
              f"{r['mean_ms']:>7.2f}ms {r['min_ms']:>7.2f}ms {r['max_ms']:>7.2f}ms")

    print(f"\n  Overall avg latency : {avg:.3f} ms  ← SOFTWARE BASELINE")
    print(f"  FPGA target         : 2-10 ms")
    print(f"  Expected speedup    : {avg/10:.1f}x – {avg/2:.1f}x faster")

    save_csv(records)
    save_chart(records)


if __name__ == '__main__':
    main()
