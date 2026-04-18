"""
compare_visual.py
-----------------
Compare visual quality of Software vs FPGA edge maps.
Metrics: MSE, PSNR, Pixel Match Percentage, Edge Overlap.

HOW TO USE:
  1. Put software edge maps in:  software/results/  (named: imagename_6_edges.png)
  2. Put FPGA output images in:  hardware/simulation/  (named: imagename_fpga_edges.png)
  3. Run: python comparison/scripts/compare_visual.py

Project: FPGA-Based Real-Time Edge Detection for Autonomous Vehicles
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SW_RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'software', 'results')
HW_RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'hardware', 'simulation')
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results')


def compute_metrics(sw_img: np.ndarray, hw_img: np.ndarray) -> dict:
    """Compute MSE, PSNR and pixel match % between two edge maps."""
    # Ensure same size and type
    if sw_img.shape != hw_img.shape:
        hw_img = cv2.resize(hw_img, (sw_img.shape[1], sw_img.shape[0]))

    sw = sw_img.astype(np.float64)
    hw = hw_img.astype(np.float64)

    mse = np.mean((sw - hw) ** 2)
    psnr = 10 * np.log10((255**2) / mse) if mse > 0 else float('inf')

    # Pixel match: percentage of pixels that are identical
    match_pct = (np.sum(sw_img == hw_img) / sw_img.size) * 100.0

    # Edge overlap: how many edge pixels (=255) are in both
    sw_edges = (sw_img == 255)
    hw_edges = (hw_img == 255)
    intersection = np.sum(sw_edges & hw_edges)
    union = np.sum(sw_edges | hw_edges)
    iou = (intersection / union * 100.0) if union > 0 else 100.0

    return {
        'mse': round(mse, 4),
        'psnr_db': round(psnr, 2),
        'pixel_match_pct': round(match_pct, 2),
        'edge_iou_pct': round(iou, 2)
    }


def compare_all():
    """Compare all software/FPGA pairs found in results directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sw_files = {
        f.replace('_6_edges.png', ''): os.path.join(SW_RESULTS_DIR, f)
        for f in os.listdir(SW_RESULTS_DIR)
        if f.endswith('_6_edges.png')
    } if os.path.exists(SW_RESULTS_DIR) else {}

    hw_files = {
        f.replace('_fpga_edges.png', ''): os.path.join(HW_RESULTS_DIR, f)
        for f in os.listdir(HW_RESULTS_DIR)
        if f.endswith('_fpga_edges.png')
    } if os.path.exists(HW_RESULTS_DIR) else {}

    common = set(sw_files.keys()) & set(hw_files.keys())

    if not common:
        print("[INFO] No matching SW/HW pairs found yet.")
        print(f"  SW edge maps expected in : {SW_RESULTS_DIR}  (filename: *_6_edges.png)")
        print(f"  HW edge maps expected in : {HW_RESULTS_DIR}  (filename: *_fpga_edges.png)")
        return

    records = []
    for name in sorted(common):
        sw_img = cv2.imread(sw_files[name], cv2.IMREAD_GRAYSCALE)
        hw_img = cv2.imread(hw_files[name], cv2.IMREAD_GRAYSCALE)
        metrics = compute_metrics(sw_img, hw_img)
        metrics['image'] = name
        records.append(metrics)

        print(f"\n[{name}]")
        print(f"  MSE            : {metrics['mse']}")
        print(f"  PSNR           : {metrics['psnr_db']} dB")
        print(f"  Pixel Match    : {metrics['pixel_match_pct']}%")
        print(f"  Edge IoU       : {metrics['edge_iou_pct']}%")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(OUTPUT_DIR, 'visual_comparison.csv'), index=False)
        print(f"\n[SAVED] Visual comparison CSV → {OUTPUT_DIR}/visual_comparison.csv")


if __name__ == '__main__':
    compare_all()
