// ============================================================
// FILE    : window_extractor.v
// PROJECT : FPGA-Based Sobel Edge Detection
// MODULE  : window_extractor
// PURPOSE : Generates a 3×3 sliding window of pixels for the
//           convolution engine. Uses line buffers and shift
//           registers to maintain a moving window as pixels
//           stream in raster order (left-to-right, top-to-bottom)
//
// THE SLIDING WINDOW PROBLEM:
//   Sobel needs pixels from 3 consecutive rows simultaneously.
//   But pixels arrive one at a time in raster order.
//   Solution: Store previous rows in line buffers (shift registers)
//
// LINE BUFFER MECHANISM:
//   lb0 [IMG_WIDTH] : holds row (N-2) — oldest row
//   lb1 [IMG_WIDTH] : holds row (N-1) — previous row
//   current pixel   : row N           — newest pixel
//
//   As each new pixel arrives at column C:
//   - lb0[C] gets old lb1[C] (push down)
//   - lb1[C] gets new pixel  (push down)
//   - sr0/sr1/sr2 shift left, new column added on right
//   - 3×3 window = left 3 elements of all 3 rows
//
// VALID OUTPUT:
//   Window is only valid after filling 2 full rows (row >= 2)
//   AND processing at least 2 pixels in current row (col >= 2)
//   Before that, output is zeros — edge padding
//
// WINDOW OUTPUT NAMING:
//   p00 p01 p02   ← top row    (row N-2, cols C-2, C-1, C)
//   p10 p11 p12   ← middle row (row N-1, cols C-2, C-1, C)
//   p20 p21 p22   ← bottom row (row N,   cols C-2, C-1, C)
//   p11 is the centre pixel being processed
//
// PORTS:
//   clk/rst    : Clock and reset
//   valid_in   : 1 when pixel_in is valid
//   pixel_in   : Current grayscale pixel value (8-bit)
//   p00..p22   : 3×3 window outputs (9 × 8-bit)
//   valid_out  : 1 when window outputs are valid
// ============================================================

module window_extractor #(
    parameter IMG_WIDTH  = 64,    // Image width (must match image_buffer)
    parameter IMG_HEIGHT = 64,    // Image height
    parameter DATA_WIDTH = 8      // Bits per pixel
)(
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  valid_in,
    input  wire [DATA_WIDTH-1:0] pixel_in,

    // 3×3 window outputs (all registered, same latency)
    output reg  [DATA_WIDTH-1:0] p00, p01, p02,  // Top row
    output reg  [DATA_WIDTH-1:0] p10, p11, p12,  // Middle row
    output reg  [DATA_WIDTH-1:0] p20, p21, p22,  // Bottom row
    output reg                   valid_out
);

    // ── LINE BUFFERS ─────────────────────────────────────────
    // lb0: stores row N-2 (two rows ago)
    // lb1: stores row N-1 (previous row)
    // Each is an array of IMG_WIDTH pixels
    reg [DATA_WIDTH-1:0] lb0 [0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] lb1 [0:IMG_WIDTH-1];

    // ── SHIFT REGISTERS (3 per row = hold 3 columns) ─────────
    // sr0[0..2] holds 3 pixels from row N-2 (oldest 3 in window)
    // sr1[0..2] holds 3 pixels from row N-1
    // sr2[0..2] holds 3 pixels from row N (current)
    // Index 0=leftmost, 2=rightmost (newest)
    reg [DATA_WIDTH-1:0] s0 [0:2];
    reg [DATA_WIDTH-1:0] s1 [0:2];
    reg [DATA_WIDTH-1:0] s2 [0:2];

    // Column and row counters to track position in image
    reg [7:0] col;   // Current column (0 to IMG_WIDTH-1)
    reg [7:0] row;   // Current row (0 to IMG_HEIGHT-1)

    integer k;  // Loop variable for reset initialization

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col       <= 0;
            row       <= 0;
            valid_out <= 0;
            // Initialize all line buffers and shift regs to 0
            // This is zero-padding for border pixels
            for (k = 0; k < IMG_WIDTH; k = k + 1) begin
                lb0[k] <= 0;
                lb1[k] <= 0;
            end
            s0[0]<=0; s0[1]<=0; s0[2]<=0;
            s1[0]<=0; s1[1]<=0; s1[2]<=0;
            s2[0]<=0; s2[1]<=0; s2[2]<=0;

        end else if (valid_in) begin

            // ── PUSH DOWN LINE BUFFERS ────────────────────────
            // At column C: shift current values down through buffers
            // lb0[C] ← lb1[C]  (row N-2 gets old N-1 value)
            // lb1[C] ← pixel_in (row N-1 gets current pixel)
            lb0[col] <= lb1[col];
            lb1[col] <= pixel_in;

            // ── SHIFT WINDOW REGISTERS LEFT ───────────────────
            // New column value enters from right ([2] side)
            // Oldest column falls off the left ([0] side)
            s0[0] <= s0[1]; s0[1] <= s0[2]; s0[2] <= lb0[col];
            s1[0] <= s1[1]; s1[1] <= s1[2]; s1[2] <= lb1[col];
            s2[0] <= s2[1]; s2[1] <= s2[2]; s2[2] <= pixel_in;

            // ── UPDATE WINDOW OUTPUTS ─────────────────────────
            // Register outputs for clean timing (no glitches)
            p00 <= s0[0]; p01 <= s0[1]; p02 <= s0[2];
            p10 <= s1[0]; p11 <= s1[1]; p12 <= s1[2];
            p20 <= s2[0]; p21 <= s2[1]; p22 <= s2[2];

            // ── VALID SIGNAL ──────────────────────────────────
            // Window only valid after:
            // - 2 complete rows have passed (row >= 2): lb0 and lb1 filled
            // - 2 pixels in current row (col >= 2): shift regs filled
            valid_out <= (row >= 2) && (col >= 2);

            // ── UPDATE COUNTERS ───────────────────────────────
            if (col == IMG_WIDTH - 1) begin
                col <= 0;           // End of row: reset column
                row <= row + 1;     // Move to next row
            end else begin
                col <= col + 1;     // Move to next column
            end

        end else begin
            // No valid input: deassert valid output
            valid_out <= 0;
        end
    end

endmodule
