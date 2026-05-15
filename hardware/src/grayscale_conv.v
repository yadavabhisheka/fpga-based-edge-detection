// ============================================================
// FILE    : grayscale_conv.v
// PROJECT : FPGA-Based Sobel Edge Detection
// MODULE  : grayscale_conv
// PURPOSE : Converts RGB colour pixel to single grayscale value.
//           Used as an optional preprocessing stage when camera
//           input is RGB (e.g., OV7670 camera module).
//
// NOTE: In our current simulation, input is already grayscale.
//       This module is included for completeness and future use
//       when connecting a real camera to the FPGA board.
//
// FORMULA (hardware approximation):
//   Gray = (R + 2*G + B) / 4
//        = (R + 2*G + B) >> 2   (right shift by 2 = divide by 4)
//
//   WHY NOT standard 0.299R + 0.587G + 0.114B?
//   That requires floating-point multiplications — expensive in HW.
//   (R + 2G + B) >> 2 uses only additions and a shift — zero LUTs
//   for the divide. Accuracy is slightly lower but acceptable.
//
// TIMING:
//   1 clock cycle latency (registered output)
//   valid_out follows valid_in by exactly 1 cycle
//
// PORTS:
//   clk       : 100MHz clock
//   rst       : Synchronous reset (active high)
//   valid_in  : 1 when RGB inputs are valid
//   r,g,b     : 8-bit colour channel inputs (0-255 each)
//   gray      : 8-bit grayscale output
//   valid_out : 1 when gray output is valid (1 cycle after valid_in)
// ============================================================

module grayscale_conv (
    input  wire       clk,        // System clock
    input  wire       rst,        // Reset (active high)
    input  wire       valid_in,   // Input valid flag
    input  wire [7:0] r,          // Red channel (0-255)
    input  wire [7:0] g,          // Green channel (0-255)
    input  wire [7:0] b,          // Blue channel (0-255)
    output reg  [7:0] gray,       // Grayscale output (0-255)
    output reg        valid_out   // Output valid flag
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // On reset: clear outputs, deassert valid
            gray      <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            // Pass valid signal through with 1 cycle delay
            valid_out <= valid_in;

            if (valid_in) begin
                // Gray = (R + 2G + B) >> 2
                // g << 1 = multiply G by 2 (left shift = free in hardware)
                // >> 2   = divide by 4    (right shift = free in hardware)
                // Total: 2 additions + 2 shifts = very low resource usage
                gray <= (r + (g << 1) + b) >> 2;
            end
        end
    end

endmodule
