// ============================================================
// FILE    : convolution_engine.v
// PROJECT : FPGA-Based Sobel Edge Detection
// MODULE  : convolution_engine
// PURPOSE : Computes Sobel Gx and Gy convolutions in PARALLEL
//           on a 3×3 pixel window in a single clock cycle.
//
// SOBEL KERNELS:
//   Gx (detects vertical edges):    Gy (detects horizontal edges):
//   [-1  0  +1]                     [-1  -2  -1]
//   [-2  0  +2]                     [ 0   0   0]
//   [-1  0  +1]                     [+1  +2  +1]
//
// KEY ADVANTAGE OVER SOFTWARE:
//   Software computes Gx THEN Gy (sequential)
//   FPGA computes Gx AND Gy SIMULTANEOUSLY (parallel)
//   Both results ready in 1 clock cycle after input
//
// NO MULTIPLIERS NEEDED:
//   All kernel weights are -1, 0, +1, -2, +2
//   -1 = negate, 0 = ignore, +1 = pass through
//   ×2 = left shift by 1 bit (<<< 1) — free in hardware
//   This saves DSP blocks which can be used for other logic
//
// SIGNED ARITHMETIC:
//   Gx and Gy can be negative (dark-to-bright or bright-to-dark)
//   Using 16-bit signed intermediates to prevent overflow
//   Max possible value: 4 × 255 = 1020 → needs 10 bits minimum
//   We use 16-bit for safety margin
//
// PIPELINE LATENCY: 1 clock cycle
//   valid_in  → valid_out delayed by exactly 1 clock
//   gx, gy output valid 1 cycle after valid_in goes high
//
// PORTS:
//   clk/rst       : Clock and active-high reset
//   valid_in      : 1 when 3×3 window inputs are valid
//   p00..p22      : 3×3 pixel window (from window_extractor)
//   gx            : Horizontal gradient output (signed 12-bit)
//   gy            : Vertical gradient output (signed 12-bit)
//   valid_out     : 1 when gx/gy are valid
// ============================================================

module convolution_engine (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,

    // 3×3 pixel window inputs (all unsigned 8-bit)
    input  wire [7:0]  p00, p01, p02,   // Top row
    input  wire [7:0]  p10, p11, p12,   // Middle row
    input  wire [7:0]  p20, p21, p22,   // Bottom row

    // Signed gradient outputs (12-bit handles full range ±1020)
    output reg  signed [11:0] gx,
    output reg  signed [11:0] gy,
    output reg                valid_out
);

    // ── CONVERT UNSIGNED PIXELS TO SIGNED ────────────────────
    // Extend 8-bit unsigned pixels to 16-bit signed
    // {8'd0, p00} = zero-extend to 16 bits (always positive)
    // We need signed context for subtraction to work correctly
    wire signed [15:0] s00 = {8'd0, p00};
    wire signed [15:0] s01 = {8'd0, p01};
    wire signed [15:0] s02 = {8'd0, p02};
    wire signed [15:0] s10 = {8'd0, p10};
    wire signed [15:0] s11 = {8'd0, p11};  // Centre pixel (not used in Sobel)
    wire signed [15:0] s12 = {8'd0, p12};
    wire signed [15:0] s20 = {8'd0, p20};
    wire signed [15:0] s21 = {8'd0, p21};
    wire signed [15:0] s22 = {8'd0, p22};

    // ── GX COMPUTATION (combinational, no clock) ──────────────
    // Gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22
    //    = (p02-p00) + 2*(p12-p10) + (p22-p20)
    // <<<1 is arithmetic left shift (multiply by 2, preserves sign)
    wire signed [15:0] gx_w = -s00 + s02
                               - (s10 <<< 1) + (s12 <<< 1)
                               - s20 + s22;

    // ── GY COMPUTATION (combinational, in parallel with Gx) ───
    // Gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22
    //    = (p20-p00) + 2*(p21-p01) + (p22-p02)
    wire signed [15:0] gy_w = -s00 - (s01 <<< 1) - s02
                               + s20 + (s21 <<< 1) + s22;

    // ── REGISTERED OUTPUTS ───────────────────────────────────
    // Register Gx, Gy, and valid to ensure clean timing
    // Truncate 16-bit result to 12 bits [11:0]
    // Max value ±1020 fits in 12 bits (range ±2047)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            gx        <= 12'sd0;
            gy        <= 12'sd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;   // Pass valid through with 1 cycle delay
            if (valid_in) begin
                gx <= gx_w[11:0];   // Take lower 12 bits of 16-bit result
                gy <= gy_w[11:0];
            end
        end
    end

endmodule
