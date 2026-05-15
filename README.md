# FPGA-Based Real-Time Sobel Edge Detection
### Hardware vs Software Latency Comparison

<div align="center">

![FPGA](https://img.shields.io/badge/FPGA-Artix--7-blue)
![Verilog](https://img.shields.io/badge/HDL-Verilog-orange)
![Python](https://img.shields.io/badge/Python-3.x-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

---

## Team & Institution

| Field | Details |
|-------|---------|
| **College** | Baba Sahab Dr. Bhim Rao Ambedkar College of Agriculture Engineering & Technology, Etawah |
| **University** | Chandra Shekhar Azad University of Agriculture and Technology, Kanpur |
| **Team** | Abhishek Yadav · Shri Ram Sharma · Priyanka Prajapati |
| **Guide** | Prof. DR N.K. Sharma — Head, Electronics Department |
| **Session** | 2025-26 |
| **Branch** | B.Tech Electronics & Communication Engineering (Final Year) |

---

## What This Project Does

This project implements the **Sobel edge detection algorithm** in two ways and compares them:

**1. Software Implementation** — Python + OpenCV running on a CPU
**2. Hardware Implementation** — Verilog RTL pipeline running on an FPGA

The goal is to prove that the FPGA hardware version processes images **10-16x faster** than software while producing visually identical edge maps — which is critical for real-time autonomous vehicle vision systems.

### What is Edge Detection?
Edge detection finds boundaries between objects in an image — where pixel intensity changes sharply. The Sobel operator uses two 3×3 kernels to compute horizontal and vertical gradients, then combines them into a gradient magnitude map.

```
Input Image → Grayscale → Blur → Sobel Gx → |Gx|  ┐
                                → Sobel Gy → |Gy|  ┤→ Magnitude → Threshold → Edge Map
```

---

## Why FPGA is Faster

| | Software (CPU) | Hardware (FPGA) |
|---|---|---|
| Processing | Sequential, pixel by pixel | Pipelined, 1 pixel per clock cycle |
| Gx and Gy | Computed one after another | Computed **simultaneously** in parallel |
| OS interference | Yes — causes random delays | None — fully deterministic |
| Latency (612×408) | ~30-40 ms | ~2.5 ms |
| Power | 50-100 W | 2-5 W |
| Speedup | baseline | **12-16x faster** |

---

## Hardware Architecture — 6 Pipeline Modules

```
                    ┌─────────────────────────────────────────┐
pixel_in ──────────►│ 1. Image Buffer (Dual-port RAM)          │
                    │    Stores entire image in FPGA BRAM      │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │ 2. Window Extractor (Line Buffers)        │
                    │    Generates 3×3 sliding window           │
                    │    Uses 2 line buffers + shift registers  │
                    └──────┬──────────────────┬───────────────┘
                           │                  │
             ┌─────────────▼──┐    ┌──────────▼──────────┐
             │ 3. Conv Engine │    │ 3. Conv Engine       │
             │    Sobel Gx    │    │    Sobel Gy          │
             │ [-1  0 +1]     │    │ [-1 -2 -1]           │
             │ [-2  0 +2]     │    │ [ 0  0  0]           │
             │ [-1  0 +1]     │    │ [+1 +2 +1]           │
             └──────┬─────────┘    └──────────┬───────────┘
                    │   (run in parallel)      │
                    └────────────┬─────────────┘
                    ┌────────────▼────────────────────────────┐
                    │ 4. Edge Detector                         │
                    │    G = |Gx| + |Gy|  (L1 approximation) │
                    │    edge = 1 if G >= THRESHOLD            │
                    └────────────┬────────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────────┐
                    │ 5. Control Unit (FSM)                    │
                    │    IDLE → LOAD → PROCESS → DONE          │
                    │    Coordinates timing of all modules      │
                    └─────────────────────────────────────────┘
```

---

## Repository Structure

```
sobel_project/
│
├── software/                          # Python + OpenCV implementation
│   ├── src/
│   │   ├── sobel_opencv.py            # Step 1: Run Sobel on all images, save edge maps
│   │   ├── benchmark.py               # Step 2: Measure latency (50 runs per image)
│   │   ├── img_to_hex.py              # Step 3: Convert image → hex for FPGA simulation
│   │   └── hex_to_img.py              # Step 5: Convert FPGA output → image + comparison
│   ├── test_images/                   # Drop your test images here (.jpg, .png)
│   ├── results/                       # Auto-generated edge maps + charts saved here
│   └── requirements.txt               # Python dependencies
│
├── hardware/                          # Verilog RTL implementation
│   ├── src/
│   │   ├── image_buffer.v             # Module 1: Dual-port RAM image storage
│   │   ├── grayscale_conv.v           # Module 2: RGB to Grayscale conversion
│   │   ├── window_extractor.v         # Module 3: 3×3 sliding window (line buffers)
│   │   ├── convolution_engine.v       # Module 4: Parallel Sobel Gx + Gy
│   │   ├── edge_detector.v            # Module 5: Gradient magnitude + threshold
│   │   ├── control_unit.v             # Module 6: FSM controller (IDLE→LOAD→PROCESS→DONE)
│   │   └── sobel_top.v                # Top module: connects all 6 modules
│   ├── testbench/
│   │   ├── tb_sobel_top.v             # Full system test: reads hex → processes → writes hex
│   │   └── tb_convolution_engine.v    # Unit test: vertical/horizontal/uniform edge cases
│   ├── constraints/
│   │   └── artix7_constraints.xdc     # Timing + pin constraints for Artix-7
│   └── simulation/
│       ├── input_pixels.hex           # Generated by img_to_hex.py → fed to testbench
│       ├── sw_edges.hex               # Software reference edge output (hex)
│       ├── hw_edges.hex               # Hardware edge output written by testbench
│       └── image_info.txt             # Image dimensions + latency info
│
├── comparison/
│   └── results/
│       ├── comparison_*.png           # Side-by-side SW vs HW edge map panel
│       └── latency_chart_*.png        # Bar chart: SW vs HW latency + speedup
│
├── .gitignore
├── resize_all.py                      # Run as step a to resize all image in 640x480
└── README.md                          # This file

```

---

## Complete Step-by-Step Workflow

### Prerequisites

```bash
# Python dependencies
cd sobel_project
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
pip install -r software/requirements.txt

# Hardware tools
# Xilinx Vivado 2023.x or 2025.x (for FPGA simulation + synthesis)
# OR Icarus Verilog (for quick simulation on Linux)
sudo apt install iverilog gtkwave
```

---

### PHASE 1 — Software Edge Detection

**Purpose:** Establish the software latency baseline that hardware must beat.

```bash
# Step 1: Add images to test_images folder
# Recommended: 50 images with variety (streets, vehicles, buildings, people)
ls software/test_images/

# Step 2: Run Sobel edge detection on all images
python software/src/sobel_opencv.py
```

**Output in `software/results/`:**
```
imagename_1_gray.png        ← Grayscale conversion
imagename_2_blur.png        ← Gaussian blur (noise reduction)
imagename_3_gx.png          ← Horizontal gradient (vertical edges)
imagename_4_gy.png          ← Vertical gradient (horizontal edges)
imagename_5_magnitude.png   ← Combined gradient magnitude
imagename_6_edges.png       ← Final binary edge map (threshold=80)
imagename_panel.png         ← All 6 stages in one image (use in thesis)
```

```bash
# Step 3: Benchmark — runs each image 50 times for stable latency numbers
python software/src/benchmark.py
```

**Output:**
```
software_benchmark.csv       ← Mean/median/min/max per image
software_latency_chart.png   ← Bar chart with FPGA target lines
```

---

### PHASE 2 — Convert Image for Hardware Simulation

**Purpose:** Convert a real JPG image to hex format that the Verilog testbench can read.

```bash
# Convert image to hex (use --size 64 for faster simulation)
python software/src/img_to_hex.py \
    --image software/test_images/img1.jpg \
    --size 64
```

**Output printed to terminal:**
```
NOW UPDATE tb_sobel_top.v WITH THESE VALUES:
  parameter IMG_WIDTH  = 64;
  parameter IMG_HEIGHT = 64;
  parameter THRESHOLD  = 80;

EXPECTED RESULTS:
  SW latency : X.XXX ms
  HW latency : X.XXX ms
  Speedup    : XX.Xx faster
```

**Files generated in `hardware/simulation/`:**
```
input_pixels.hex   ← one pixel per line in hex, feeds into testbench
sw_edges.hex       ← software reference output for comparison
sw_edges.png       ← software edge map image
image_info.txt     ← dimensions + latency info used by hex_to_img.py
```

---

### PHASE 3 — Hardware Simulation in Vivado

#### Create Vivado Project
```
1. Open Vivado 2025.x
2. Create Project → RTL Project
3. Part: xc7a35tcpg236-1 (Artix-7)
4. Add Design Sources → select all hardware/src/*.v files
5. Add Constraints  → hardware/constraints/artix7_constraints.xdc
6. Add Simulation Sources → hardware/testbench/tb_sobel_top.v
7. Right click tb_sobel_top → Set as Top
```

#### Update Testbench Parameters
Open `hardware/testbench/tb_sobel_top.v` and update:
```verilog
parameter IMG_WIDTH  = 64;   // ← from img_to_hex.py output
parameter IMG_HEIGHT = 64;   // ← from img_to_hex.py output
parameter THRESHOLD  = 80;
```

#### Run Simulation
```
Flow Navigator → Run Simulation → Run Behavioral Simulation

In Tcl Console type:
  run 100us

Press Enter and wait.
```

#### Expected Tcl Console Output
```
==============================================
  FPGA SOBEL EDGE DETECTION SIMULATION
  Image: 64x64 | Threshold:80 | 100MHz
==============================================
[SAVED] Hardware edge map: ../simulation/hw_edges.hex

  LATENCY RESULTS:
  Clock cycles : XXXX
  HW Latency   : X.XXXX ms
  Edge pixels  : XX / 4096

  REAL IMAGE EXTRAPOLATION (100MHz):
  img1 (612x306=187272px): 1.873 ms
  img2 (612x408=249696px): 2.497 ms

  SOFTWARE baseline: ~30.741 ms
  HW speedup img1 : 16.4x
  HW speedup img2 : 12.3x
==============================================
```

#### Run Synthesis (Resource Report for Thesis)
```
Flow Navigator → Run Synthesis → Open Synthesized Design
Reports → Report Utilization   → screenshot LUT/FF/BRAM usage
Reports → Report Timing        → verify WNS >= 0 (meets 100MHz)
```

---

### PHASE 4 — Generate Comparison Results

**Purpose:** Convert FPGA hex output to image and compare with software output.

```bash
# After Vivado simulation completes and hw_edges.hex is generated
python software/src/hex_to_img.py
```

**Output in `comparison/results/`:**
```
comparison_img1.jpg.png      ← 3-panel: Input | SW edges | HW edges
latency_chart_img1.jpg.png   ← Bar chart: SW latency vs HW latency + speedup
```

**Metrics printed:**
```
QUALITY METRICS:
  MSE         : X.XX       ← lower is better (0 = identical)
  PSNR        : XX dB      ← higher is better
  Pixel Match : XX.X%      ← percentage of identical pixels
  Edge IoU    : XX.X%      ← edge overlap between SW and HW

LATENCY COMPARISON:
  Software : XX.XX ms
  Hardware : X.XX ms
  Speedup  : XX.Xx faster
```

---

### Alternative: Quick Test with Icarus Verilog (No Vivado needed)

```bash
# Unit test — convolution engine only
iverilog -o conv_test \
  hardware/testbench/tb_convolution_engine.v \
  hardware/src/convolution_engine.v
vvp conv_test

# Expected output:
# [Test1 Vertical Edge]   Gx=400 Gy=0   RESULT: PASS
# [Test2 Horizontal Edge] Gx=0   Gy=400 RESULT: PASS
# [Test3 Uniform Region]  Gx=0   Gy=0   RESULT: PASS

# Full system simulation
iverilog -o sobel_sim \
  hardware/testbench/tb_sobel_top.v \
  hardware/src/sobel_top.v \
  hardware/src/image_buffer.v \
  hardware/src/window_extractor.v \
  hardware/src/convolution_engine.v \
  hardware/src/edge_detector.v \
  hardware/src/control_unit.v
vvp sobel_sim

# View waveforms
gtkwave sobel_sim.vcd
```

---

## Latency Results

### Software Baseline (Python + OpenCV)

| Image | Resolution | Mean Latency | Std Dev |
|-------|-----------|-------------|---------|
| img1  | 612×306   | ~21 ms      | ~15 ms  |
| img2  | 612×408   | ~40 ms      | ~4 ms   |
| **Average** | — | **~30.7 ms** | — |

### Hardware Performance (FPGA @ 100MHz)

| Image | Resolution | Pixels | HW Latency | SW Latency | Speedup |
|-------|-----------|--------|-----------|-----------|---------|
| img1  | 612×306   | 187,272 | **1.87 ms** | 21.3 ms | **11.4×** |
| img2  | 612×408   | 249,696 | **2.50 ms** | 40.2 ms | **16.1×** |

> Hardware latency = Total pixels × 10ns (1 pixel/clock at 100MHz)
> This is deterministic — same every time, no OS interference.

---

## How to Verify the Hardware is Correct

**Test 1 — Convolution Unit Test** (run tb_convolution_engine):
```
Vertical edge window  → Gx should be 400, Gy should be 0
Horizontal edge window → Gx should be 0,  Gy should be 400
Uniform window        → Gx should be 0,  Gy should be 0
```

**Test 2 — Visual Comparison** (run hex_to_img.py):
```
Compare sw_edges.png vs hw_edges.png visually
Both should show same edge locations
Minor differences only at image borders (padding difference)
```

**Test 3 — Synthesis Timing** (Vivado Report Timing):
```
Worst Negative Slack (WNS) >= 0 means design runs at 100MHz ✓
```

---

## Sobel Algorithm — Quick Reference

```
For each pixel at position (i,j), extract 3×3 neighbourhood:

P = [p00  p01  p02]
    [p10  p11  p12]
    [p20  p21  p22]

Gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22
   = (p02-p00) + 2*(p12-p10) + (p22-p20)    ← detects vertical edges

Gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22
   = (p20-p00) + 2*(p21-p01) + (p22-p02)    ← detects horizontal edges

G  = |Gx| + |Gy|          ← L1 approximation (used in FPGA)
   (exact: sqrt(Gx²+Gy²)  ← expensive, not used in hardware)

edge = 1  if G >= THRESHOLD (80)
edge = 0  if G <  THRESHOLD
```

**FPGA advantage:** Gx and Gy are computed in parallel in the same clock cycle.
**Software:** Gx computed first, then Gy sequentially.

---

## Tools & Technologies

| Category | Tool | Version |
|----------|------|---------|
| HDL | Verilog HDL | IEEE 1364-2001 |
| FPGA Synthesis | Xilinx Vivado | 2025.x |
| Target FPGA | Xilinx Artix-7 | xc7a35tcpg236-1 |
| Simulation | Vivado Simulator / Icarus Verilog | — |
| Waveform | GTKWave | — |
| Image Processing | Python + OpenCV | 3.x + 4.8.x |
| Numerical | NumPy | 1.24+ |
| Visualization | Matplotlib | 3.7+ |
| Data | Pandas | 2.0+ |
| Version Control | Git + GitHub | — |

---

## References

1. Canny, J. (1986). A computational approach to edge detection. *IEEE PAMI*, 8(6), 679-698.
2. Vasamsetti et al. (2017). FPGA implementation of high performance image edge detection. *WiSPNET*, Chennai.
3. Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing*, 4th Ed. Pearson.
4. Anh et al. (2016). FPGA implementation of real-time edge detection. *ATC*, Hanoi.
5. Kumar & Saini (2020). Performance comparison of edge detection on FPGA. *ComPE*, Shillong.
6. Monmasson & Cirstea (2007). FPGA design methodology for industrial control. *IEEE Trans. Ind. Electron.*, 54(4).
7. AngeloJacobo (2021). FPGA Real-Time Sobel Edge Detection. GitHub: AngeloJacobo/FPGA_RealTime_and_Static_Sobel_Edge_Detection
