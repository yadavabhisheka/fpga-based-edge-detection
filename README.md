# FPGA-Based Real-Time Edge Detection
### Sobel Algorithm — Hardware vs Software Latency Comparison

**College:** Baba Sahab Dr. Bhim Rao Ambedkar College of Agriculture Engineering & Technology, Etawah  
**Team:** Abhishek Yadav, Shri Ram Sharma, Priyanka Prajapati  
**Guide:** Prof. DR N.K. Sharma, Head — Electronics Department  
**Session:** 2025–26

---

## Project Goal

Demonstrate that an FPGA hardware implementation of the Sobel edge detection algorithm achieves **10–50x lower latency** compared to a CPU-based Python/OpenCV software implementation, while producing visually equivalent edge maps.

| Metric | Software (OpenCV) | FPGA Hardware (Target) |
|--------|-------------------|------------------------|
| Latency per frame | ~100 ms | 2–10 ms |
| Power | ~50–100 W | ~2–5 W |
| Determinism | Non-deterministic | Fully deterministic |
| Throughput | ~30–60 FPS | 200–500+ FPS |

---

## Repository Structure

```
fpga-sobel-edge-detection/
│
├── software/                        # Python + OpenCV implementation
│   ├── src/
│   │   ├── sobel_opencv.py          # Core Sobel implementation
│   │   └── benchmark.py             # Latency measurement for software
│   ├── test_images/                 # Input images (real-world objects)
│   ├── results/                     # Output edge maps (software)
│   └── requirements.txt
│
├── hardware/                        # Verilog RTL implementation
│   ├── src/
│   │   ├── image_buffer.v           # Module 1: Dual-port RAM image storage
│   │   ├── grayscale_conv.v         # Module 2: RGB to grayscale
│   │   ├── window_extractor.v       # Module 3: 3x3 sliding window
│   │   ├── convolution_engine.v     # Module 4: Sobel Gx + Gy parallel
│   │   ├── edge_detector.v          # Module 5: Gradient magnitude + threshold
│   │   ├── control_unit.v           # Module 6: FSM controller
│   │   └── sobel_top.v              # Top-level integration
│   ├── testbench/
│   │   ├── tb_image_buffer.v
│   │   ├── tb_window_extractor.v
│   │   ├── tb_convolution_engine.v
│   │   ├── tb_edge_detector.v
│   │   └── tb_sobel_top.v
│   ├── constraints/
│   │   └── artix7_constraints.xdc
│   └── simulation/                  # Vivado simulation outputs
│
├── comparison/                      # Head-to-head benchmarking
│   ├── scripts/
│   │   ├── compare_latency.py       # Software vs hardware latency analysis
│   │   └── compare_visual.py        # Edge map quality comparison (MSE, PSNR)
│   ├── results/                     # CSVs, plots, logs
│   └── reports/                     # Final comparison report
│
└── docs/
    ├── synopsis.pdf
    ├── sobel_algorithm_reference.docx
    └── software_flowchart.html
```

---

## Setup & Run

### Software Implementation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fpga-sobel-edge-detection.git
cd fpga-sobel-edge-detection

# Install Python dependencies
pip install -r software/requirements.txt

# Run Sobel edge detection on test images
python software/src/sobel_opencv.py

# Run latency benchmark
python software/src/benchmark.py
```

### Hardware Implementation
- Open `hardware/src/sobel_top.v` in Xilinx Vivado 2023.x
- Target board: Artix-7 / Spartan-6 FPGA
- Run simulation with testbenches in `hardware/testbench/`

### Comparison
```bash
# After FPGA results are exported, run comparison
python comparison/scripts/compare_latency.py
python comparison/scripts/compare_visual.py
```

---

## Results (To Be Updated)

| Image | Resolution | SW Latency | HW Latency | Speedup | MSE |
|-------|-----------|------------|------------|---------|-----|
| TBD   | TBD       | TBD ms     | TBD ms     | TBDx    | TBD |

---

## Tools & Technologies

- **Hardware:** Verilog HDL, Xilinx Vivado 2023.x, Artix-7 FPGA
- **Software:** Python 3.x, OpenCV 4.x, NumPy, Matplotlib
- **Version Control:** Git / GitHub
