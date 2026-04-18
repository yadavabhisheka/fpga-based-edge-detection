"""
benchmark.py
------------
Precise latency measurement for the software Sobel implementation.
Runs each image N times, records statistics, and exports a CSV
that will later be used in the hardware vs software comparison.

Project: FPGA-Based Real-Time Edge Detection for Autonomous Vehicles
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import statistics

# ─────────────────────────────────────────────
#  BENCHMARK CONFIGURATION
# ─────────────────────────────────────────────
N_RUNS            = 50      # How many times to run each image (for stable average)
WARMUP_RUNS       = 5       # Discard first N runs (CPU cache warmup)
GAUSSIAN_KSIZE    = (3, 3)
SOBEL_KSIZE       = 3
THRESHOLD_VALUE   = 80

INPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'test_images')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_CSV = os.path.join(OUTPUT_DIR, 'software_benchmark.csv')
# ─────────────────────────────────────────────


def run_pipeline(gray: np.ndarray) -> np.ndarray:
    """
    Core processing pipeline — ONLY this is timed.
    Input must already be grayscale uint8.
    Returns final edge map.
    """
    blur  = cv2.GaussianBlur(gray, GAUSSIAN_KSIZE, 0)
    gx    = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    gy    = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    gx_a  = cv2.convertScaleAbs(gx)
    gy_a  = cv2.convertScaleAbs(gy)
    mag   = cv2.addWeighted(gx_a, 0.5, gy_a, 0.5, 0)
    _, edges = cv2.threshold(mag, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    return edges


def benchmark_image(image_path: str) -> dict:
    """
    Benchmark a single image over N_RUNS iterations.
    Returns statistics dict.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    name = os.path.basename(image_path)
    pixels = h * w

    print(f"\n[BENCH] {name}  ({w}x{h} = {pixels:,} px)  |  {N_RUNS} runs + {WARMUP_RUNS} warmup")

    # Warmup runs — discard (lets CPU cache and JIT settle)
    for _ in range(WARMUP_RUNS):
        run_pipeline(gray)

    # Actual timed runs
    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        run_pipeline(gray)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    mean_ms   = statistics.mean(times)
    median_ms = statistics.median(times)
    stdev_ms  = statistics.stdev(times)
    min_ms    = min(times)
    max_ms    = max(times)

    print(f"  Mean: {mean_ms:.3f} ms  |  Median: {median_ms:.3f} ms  |  "
          f"Std: {stdev_ms:.3f} ms  |  Min: {min_ms:.3f}  |  Max: {max_ms:.3f}")

    return {
        'image_name': name,
        'width': w,
        'height': h,
        'pixels': pixels,
        'n_runs': N_RUNS,
        'mean_ms': round(mean_ms, 4),
        'median_ms': round(median_ms, 4),
        'stdev_ms': round(stdev_ms, 4),
        'min_ms': round(min_ms, 4),
        'max_ms': round(max_ms, 4),
        'fps_estimate': round(1000.0 / mean_ms, 1),
        'raw_times': times
    }


def save_benchmark_csv(records: list):
    """Export benchmark results to CSV (excluding raw_times column)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'raw_times'}
        for r in records
    ])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[SAVED] Benchmark CSV → {RESULTS_CSV}")
    return df


def plot_latency_chart(records: list):
    """Bar chart: mean latency per image with error bars."""
    names    = [r['image_name'].replace('.jpg','').replace('.png','') for r in records]
    means    = [r['mean_ms']  for r in records]
    stdevs   = [r['stdev_ms'] for r in records]
    resols   = [f"{r['width']}x{r['height']}" for r in records]

    fig, ax = plt.subplots(figsize=(max(8, len(records)*2), 6))
    bars = ax.bar(names, means, yerr=stdevs, capsize=5,
                  color='#2C5F8A', edgecolor='#1A3A5C', linewidth=1.2,
                  error_kw=dict(ecolor='#C4860A', lw=2))

    # Annotate bars with ms values and resolution
    for bar, mean, res in zip(bars, means, resols):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stdevs)*0.1,
                f'{mean:.1f} ms\n{res}', ha='center', va='bottom', fontsize=9, color='#1A3A5C')

    # Reference lines for FPGA target
    ax.axhline(y=10, color='#1A6B1A', linestyle='--', linewidth=1.5, label='FPGA Target Max (10 ms)')
    ax.axhline(y=2,  color='#C4860A', linestyle='--', linewidth=1.5, label='FPGA Target Min (2 ms)')

    ax.set_xlabel('Test Image', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(f'Software Sobel — Latency per Image\n(Python + OpenCV, {N_RUNS} runs, Gaussian {GAUSSIAN_KSIZE[0]}×{GAUSSIAN_KSIZE[0]})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, 'software_latency_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Latency chart → {chart_path}")


def run_benchmark():
    """Main entry point — benchmark all images in test_images/."""
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(supported)
    ])

    if not images:
        print(f"\n[WARNING] No images found in {INPUT_DIR}")
        print("[INFO] Add real-world images to software/test_images/ and rerun.")
        return

    print(f"\n{'='*60}")
    print(f"  SOFTWARE BENCHMARK — Sobel Edge Detection")
    print(f"  {len(images)} image(s)  |  {N_RUNS} runs each  |  {WARMUP_RUNS} warmup runs")
    print(f"{'='*60}")

    records = []
    for path in images:
        result = benchmark_image(path)
        if result:
            records.append(result)

    if not records:
        print("[ERROR] No valid benchmark results.")
        return

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Image':<28} {'Res':<12} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*70}")
    for r in records:
        print(f"  {r['image_name']:<28} {str(r['width'])+'x'+str(r['height']):<12} "
              f"{r['mean_ms']:>7.2f}ms {r['median_ms']:>7.2f}ms "
              f"{r['min_ms']:>7.2f}ms {r['max_ms']:>7.2f}ms")

    overall_mean = statistics.mean([r['mean_ms'] for r in records])
    print(f"\n  Overall average latency : {overall_mean:.3f} ms")
    print(f"  This is the baseline that FPGA must beat.")
    print(f"  FPGA target             : 2–10 ms")
    print(f"  Expected speedup        : {overall_mean/10:.1f}x – {overall_mean/2:.1f}x")

    save_benchmark_csv(records)
    plot_latency_chart(records)


if __name__ == '__main__':
    run_benchmark()
