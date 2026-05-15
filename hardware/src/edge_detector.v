// ============================================================
// FILE    : edge_detector.v
// PROJECT : FPGA-Based Sobel Edge Detection
// MODULE  : edge_detector
// PURPOSE : Computes gradient magnitude from Gx and Gy,
//           then applies a threshold to produce binary edge map.
//
// GRADIENT MAGNITUDE:
//   Exact:  G = sqrt(Gx² + Gy²)    ← expensive (needs multiplier + sqrt)
//   L1 approx: G = |Gx| + |Gy|    ← cheap (just abs + add)
//
//   WHY L1 APPROXIMATION?
//   sqrt requires a DSP block or many clock cycles.
//   |Gx| + |Gy| overestimates by max 41% in worst case
//   (when Gx = Gy, L1 = 2|G| vs exact = sqrt(2)|G| ≈ 1.41|G|)
//   Visually the edge maps are nearly identical.
//   This is also what the Python software uses (addWeighted 0.5+0.5)
//
// ABSOLUTE VALUE IMPLEMENTATION:
//   For a signed number in 2's complement:
//   If bit[11] = 1 (negative) → abs = ~value + 1 (negate)
//   If bit[11] = 0 (positive) → abs = value (unchanged)
//   gx[11] ? (~gx + 1) : gx
//
// CLAMPING:
//   |Gx| + |Gy| can exceed 255 for strong edges
//   We clamp to 255 so output fits in 8-bit pixel format
//
// THRESHOLD:
//   G >= THRESHOLD → edge_out = 1 (white pixel in edge map)
//   G <  THRESHOLD → edge_out = 0 (black pixel, no edge)
//   THRESHOLD = 80 (default) — tune higher to suppress weak edges
//
// PIPELINE LATENCY: 1 clock cycle
//
// PORTS:
//   clk/rst       : Clock and reset
//   valid_in      : 1 when gx/gy are valid
//   gx, gy        : Signed 12-bit gradient inputs (from convolution_engine)
//   magnitude     : Unsigned 8-bit gradient magnitude output
//   edge_out      : Binary edge flag (1=edge, 0=no edge)
//   valid_out     : 1 when magnitude and edge_out are valid
// ============================================================

module edge_detector #(
    parameter THRESHOLD = 80    // Edge detection threshold (0-255)
                                // Increase to detect only strong edges
                                // Decrease to detect more edges (more noise)
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,

    input  wire signed [11:0] gx,   // Horizontal gradient from conv engine
    input  wire signed [11:0] gy,   // Vertical gradient from conv engine

    output reg  [7:0]  magnitude,   // Gradient magnitude |Gx|+|Gy| clamped to 0-255
    output reg         edge_out,    // 1 = edge pixel, 0 = no edge
    output reg         valid_out    // 1 = outputs are valid
);

    // ── ABSOLUTE VALUES (combinational) ──────────────────────
    // For 12-bit 2's complement signed numbers:
    // gx[11] is the sign bit (1=negative, 0=positive)
    // ~gx+1 is the 2's complement negation
    wire [11:0] abs_gx = gx[11] ? (~gx + 1) : gx;
    wire [11:0] abs_gy = gy[11] ? (~gy + 1) : gy;

    // ── L1 MAGNITUDE (combinational) ──────────────────────────
    // Use 13 bits to hold sum (max: 1020 + 1020 = 2040 > 2^11 = 2048)
    // 13 bits handles values up to 8191 — more than enough
    wire [12:0] mag_sum = abs_gx + abs_gy;

    // ── CLAMP TO 8-BIT (combinational) ───────────────────────
    // If sum > 255, clamp to 255 (maximum white = strong edge)
    // Otherwise take lower 8 bits
    wire [7:0] mag_clamped = (mag_sum > 13'd255) ? 8'd255 : mag_sum[7:0];

    // ── REGISTERED OUTPUTS ───────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            magnitude <= 8'd0;
            edge_out  <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;   // 1 cycle pipeline delay
            if (valid_in) begin
                magnitude <= mag_clamped;
                // Threshold comparison: edge if magnitude strong enough
                // Matches FPGA behaviour to Python: threshold(mag, THRESHOLD, 255, BINARY)
                edge_out  <= (mag_clamped >= THRESHOLD) ? 1'b1 : 1'b0;
            end
        end
    end

endmodule
