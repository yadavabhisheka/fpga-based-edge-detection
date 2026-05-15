"""
============================================================
FILE    : img_to_hex.py
PROJECT : FPGA-Based Sobel Edge Detection
PURPOSE : Converts a real JPG/PNG image into hex format
          that the Verilog testbench can read using $readmemh.
          Also runs software Sobel to generate the reference
          edge map for comparison after simulation.

WHY HEX FILES?
  Verilog testbenches cannot load .jpg or .png files directly.
  $readmemh reads a text file where each line is one hex value.
  We convert each grayscale pixel (0-255) to 2-digit hex (00-ff).
  Example: pixel value 200 → "c8" in hex file

WHY RESIZE TO 64x64?
  Vivado simulation is cycle-accurate — it simulates every single
  clock edge. A 612x408 image = 249,696 clock cycles to simulate.
  At 100MHz that's 2.5ms real time but hours of simulation time.
  64x64 = 4096 pixels → simulation finishes in seconds.
  Latency for real images is calculated mathematically.

HOW TO RUN:
  python software/src/img_to_hex.py --image software/test_images/img1.jpg
  python software/src/img_to_hex.py --image software/test_images/img1.jpg --size 32

OUTPUT (in hardware/simulation/):
  input_pixels.hex  - Grayscale pixel values, one per line in hex
  sw_edges.hex      - Software reference edge output in hex
  sw_edges.png      - Software edge map image
  image_info.txt    - Dimensions + latency info (read by hex_to_img.py)
============================================================
"""

import cv2           # Image loading and processing
import numpy as np   # Array operations
import os            # File/directory operations
import time          # Latency measurement
import argparse      # Command line argument parsing

# Output directory for simulation files
SIM_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'hardware', 'simulation')
IN_HEX    = os.path.join(SIM_DIR, 'input_pixels.hex')   # Fed into Vivado testbench
SW_HEX    = os.path.join(SIM_DIR, 'sw_edges.hex')        # Software reference output
SW_PNG    = os.path.join(SIM_DIR, 'sw_edges.png')        # Visual reference
INFO_FILE = os.path.join(SIM_DIR, 'image_info.txt')      # Read by hex_to_img.py


def convert(image_path: str, sim_size: int = 64):
    """
    Full conversion pipeline:
    1. Load image
    2. Resize to sim_size x sim_size
    3. Convert to grayscale
    4. Write pixel hex file (input for FPGA)
    5. Run software Sobel for reference
    6. Write edge hex file (reference output)
    7. Write info file for hex_to_img.py

    Args:
        image_path : Path to input image (.jpg, .png, etc.)
        sim_size   : Target size for simulation (default 64)
                     Larger = more accurate but slower simulation
    """
    os.makedirs(SIM_DIR, exist_ok=True)

    # ── STEP 1: Load image ───────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        print(f"[INFO]  Check the path is correct")
        return
    orig_h, orig_w = img.shape[:2]
    print(f"\n[INFO] Original image: {orig_w}x{orig_h} pixels")

    # ── STEP 2: Resize for simulation ────────────────────────
    # We resize to a square for simplicity
    # In real hardware, any rectangular size works
    img_small = cv2.resize(img, (sim_size, sim_size))
    print(f"[INFO] Resized to: {sim_size}x{sim_size} for simulation")

    # ── STEP 3: Convert to grayscale ─────────────────────────
    # FPGA receives grayscale pixels (1 byte each)
    # Color images need 3 bytes per pixel (RGB) — not handled in HW
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape  # Should be (sim_size, sim_size)

    # ── STEP 4: Write input pixel hex file ───────────────────
    # Format: one pixel per line, 2 hex digits
    # Row-major order: left-to-right, top-to-bottom (raster scan)
    # This matches how Verilog testbench feeds pixels to the pipeline
    with open(IN_HEX, 'w') as f:
        for row in range(h):
            for col in range(w):
                # {:02x} formats as 2-digit lowercase hex with leading zero
                # e.g., 15 → "0f", 200 → "c8", 255 → "ff"
                f.write(f"{gray[row, col]:02x}\n")
    print(f"[SAVED] Input hex: {IN_HEX}  ({w*h} pixels)")

    # ── STEP 5: Run software Sobel (reference) ────────────────
    # Run on the SAME resized grayscale image as FPGA will process
    # This ensures fair comparison — same input, same size
    t0 = time.perf_counter()
    
    gx     = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=3)
    gy     = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=3)
    gx_abs = cv2.convertScaleAbs(gx)
    gy_abs = cv2.convertScaleAbs(gy)
    mag    = cv2.addWeighted(gx_abs, 0.5, gy_abs, 0.5, 0)
    _, sw_edges = cv2.threshold(mag, 120, 255, cv2.THRESH_BINARY)
    t1 = time.perf_counter()
    sw_ms = (t1 - t0) * 1000.0

    # ── STEP 6: Write software edge hex (reference output) ────
    # Same format as input — one pixel per line
    # 00 = no edge (black), ff = edge (white)
    with open(SW_HEX, 'w') as f:
        for row in range(h):
            for col in range(w):
                f.write(f"{sw_edges[row, col]:02x}\n")

    # Save software edge as PNG for visual inspection
    cv2.imwrite(SW_PNG, sw_edges)
    print(f"[SAVED] SW edges hex: {SW_HEX}")
    print(f"[SAVED] SW edges PNG: {SW_PNG}")
    print(f"[INFO]  SW latency:   {sw_ms:.4f} ms")

    # ── STEP 7: Calculate HW latency and write info file ──────
    # Hardware latency = 1 pixel per clock cycle at 100MHz = 10ns/pixel
    hw_ms   = (w * h * 10) / 1_000_000   # 10ns × pixels → ms
    speedup = sw_ms / hw_ms

    with open(INFO_FILE, 'w') as f:
        f.write(f"image_name={os.path.basename(image_path)}\n")
        f.write(f"orig_width={orig_w}\n")
        f.write(f"orig_height={orig_h}\n")
        f.write(f"sim_width={w}\n")
        f.write(f"sim_height={h}\n")
        f.write(f"total_pixels={w*h}\n")
        f.write(f"sw_latency_ms={sw_ms:.4f}\n")
        f.write(f"hw_latency_ms={hw_ms:.4f}\n")
        f.write(f"speedup={speedup:.1f}\n")
    print(f"[SAVED] Image info: {INFO_FILE}")

    # Print what user needs to update in Vivado testbench
    print(f"""
{'='*55}
  NOW UPDATE tb_sobel_top.v WITH THESE VALUES:
  (Open the file in Vivado and change parameters)
{'='*55}
  parameter IMG_WIDTH  = {w};
  parameter IMG_HEIGHT = {h};
  parameter THRESHOLD  = 120;
{'='*55}
  EXPECTED SIMULATION RESULTS:
  SW latency (this image) : {sw_ms:.4f} ms
  HW latency (calculated) : {hw_ms:.4f} ms
  Expected speedup        : {speedup:.1f}x faster

{'='*55}
  NEXT STEPS:
  1. Update tb_sobel_top.v parameters as shown above
  2. Run Vivado simulation: run 100us in Tcl console
  3. hw_edges.hex will be written to hardware/simulation/
  4. Run: python software/src/hex_to_img.py
{'='*55}
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert image to hex for FPGA simulation'
    )
    parser.add_argument(
        '--image', required=True,
        help='Path to input image (e.g., software/test_images/img1.jpg)'
    )
    parser.add_argument(
        '--size', type=int, default=64,
        help='Simulation image size in pixels (default: 64). '
             'Use 32 for faster simulation, 128 for more detail.'
    )
    args = parser.parse_args()
    convert(args.image, sim_size=args.size)
