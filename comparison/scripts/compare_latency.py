"""
compare_latency.py
------------------
Final comparison: Software (Python/OpenCV) vs Hardware (FPGA)
Reads software_benchmark.csv + hardware latency values you provide,
then generates the comparison charts and report for your project.

Project: FPGA-Based Real-Time Edge Detection for Autonomous Vehicles

HOW TO USE:
  1. Run software benchmark first:
         python software/src/benchmark.py
  2. Get FPGA latency from Vivado simulation or hardware measurement.
  3. Fill in FPGA_RESULTS below with your actual measured values.
  4. Run this script:
         python comparison/scripts/compare_latency.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ─────────────────────────────────────────────
#  FILL IN YOUR FPGA RESULTS HERE
#  After Vivado simulation / hardware measurement
# ─────────────────────────────────────────────
FPGA_RESULTS = [
    # { 'image_name': 'car.jpg',       'fpga_latency_ms': 3.2,  'clock_mhz': 100 },
    # { 'image_name': 'road.jpg',      'fpga_latency_ms': 4.1,  'clock_mhz': 100 },
    # { 'image_name': 'person.jpg',    'fpga_latency_ms': 2.8,  'clock_mhz': 100 },
    # Add entries matching your test_images/ filenames
]
# ─────────────────────────────────────────────

SW_CSV     = os.path.join(os.path.dirname(__file__), '..', '..', 'software', 'results', 'software_benchmark.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_software_results() -> pd.DataFrame:
    if not os.path.exists(SW_CSV):
        print(f"[ERROR] Run software benchmark first: python software/src/benchmark.py")
        return None
    df = pd.read_csv(SW_CSV)
    print(f"[INFO] Loaded software results: {len(df)} images")
    return df


def generate_comparison_chart(sw_df: pd.DataFrame, hw_results: list):
    """Side-by-side bar chart: Software vs FPGA latency per image."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not hw_results:
        print("[INFO] No FPGA results yet. Chart will show software only with FPGA target range.")
        _chart_software_only(sw_df)
        return

    # Merge on image_name
    hw_df = pd.DataFrame(hw_results)
    merged = sw_df.merge(hw_df, on='image_name', how='inner')

    if merged.empty:
        print("[WARNING] No matching image names between SW and HW results.")
        _chart_software_only(sw_df)
        return

    names = [n.replace('.jpg','').replace('.png','') for n in merged['image_name']]
    sw_ms = merged['mean_ms'].values
    hw_ms = merged['fpga_latency_ms'].values
    speedup = sw_ms / hw_ms

    x = np.arange(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(names)*2.5), 10))
    fig.suptitle('Software vs FPGA Hardware — Sobel Edge Detection\nLatency Comparison',
                 fontsize=15, fontweight='bold')

    # Top: latency bars
    b1 = ax1.bar(x - w/2, sw_ms, w, label='Software (Python/OpenCV)',
                 color='#C4860A', edgecolor='#7A4F00')
    b2 = ax1.bar(x + w/2, hw_ms, w, label='FPGA Hardware',
                 color='#2C5F8A', edgecolor='#1A3A5C')

    for bar, val in zip(b1, sw_ms):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}ms', ha='center', va='bottom', fontsize=9, color='#7A4F00')
    for bar, val in zip(b2, hw_ms):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}ms', ha='center', va='bottom', fontsize=9, color='#1A3A5C')

    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.legend(fontsize=11); ax1.grid(axis='y', alpha=0.3)
    ax1.set_title('Processing Latency per Image', fontsize=12)

    # Bottom: speedup
    bars3 = ax2.bar(names, speedup, color='#1A6B1A', edgecolor='#0D4010')
    for bar, val in zip(bars3, speedup):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f'{val:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#0D4010')

    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax2.set_ylabel('Speedup (SW / HW)', fontsize=12)
    ax2.set_title('FPGA Speedup over Software', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'comparison_latency_chart.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Comparison chart → {out}")

    _print_summary_table(merged, speedup)


def _chart_software_only(sw_df: pd.DataFrame):
    """Show software latency with FPGA target zone while HW results are pending."""
    names = [n.replace('.jpg','').replace('.png','') for n in sw_df['image_name']]
    means = sw_df['mean_ms'].values

    fig, ax = plt.subplots(figsize=(max(9, len(names)*2), 6))
    bars = ax.bar(names, means, color='#C4860A', edgecolor='#7A4F00', label='Software (Python/OpenCV)')

    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f} ms', ha='center', va='bottom', fontsize=10)

    ax.axhspan(2, 10, alpha=0.15, color='#1A6B1A', label='FPGA Target Range (2–10 ms)')
    ax.axhline(y=10, color='#1A6B1A', linestyle='--', linewidth=1.5)
    ax.axhline(y=2,  color='#1A6B1A', linestyle=':',  linewidth=1.5)

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Software Sobel Latency — Baseline\n(FPGA results will be added after hardware implementation)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'software_baseline_chart.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Software baseline chart → {out}")


def _print_summary_table(merged: pd.DataFrame, speedup: np.ndarray):
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Image':<25} {'SW (ms)':>10} {'HW (ms)':>10} {'Speedup':>10}")
    print(f"  {'-'*55}")
    for _, row in merged.iterrows():
        sp = row['mean_ms'] / row['fpga_latency_ms']
        print(f"  {row['image_name']:<25} {row['mean_ms']:>10.2f} {row['fpga_latency_ms']:>10.2f} {sp:>9.1f}×")
    print(f"\n  Average speedup: {speedup.mean():.1f}× faster on FPGA")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sw_df = load_software_results()
    if sw_df is not None:
        generate_comparison_chart(sw_df, FPGA_RESULTS)
