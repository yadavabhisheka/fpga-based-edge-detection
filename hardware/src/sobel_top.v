// ============================================================
// FILE    : sobel_top.v
// PROJECT : FPGA-Based Sobel Edge Detection
// MODULE  : sobel_top
// PURPOSE : Top-level module that connects all 6 pipeline
//           submodules into a complete edge detection system.
//
// COMPLETE PIPELINE:
//
//  pixel_in → [Control Unit FSM]
//                  |
//                  ├──(load_en)──→ [Image Buffer] ──(buf_out)──→
//                  |                                              |
//                  └──(proc_en)────────────────────────────────  |
//                                                                 ↓
//                                                    [Window Extractor]
//                                                    (3×3 sliding window)
//                                                          |
//                                              ┌───────────┴───────────┐
//                                              ↓                       ↓
//                                     [Conv Engine Gx]      [Conv Engine Gy]
//                                              └───────────┬───────────┘
//                                                          ↓
//                                                 [Edge Detector]
//                                                 (magnitude + threshold)
//                                                          |
//                                              ┌───────────┴───────────┐
//                                              ↓                       ↓
//                                          edge_out              magnitude_out
//
// TOTAL PIPELINE LATENCY:
//   image_buffer    : 1 cycle (registered read)
//   window_extractor: 1 cycle (registered output)
//   convolution     : 1 cycle (registered gx/gy)
//   edge_detector   : 1 cycle (registered magnitude/edge)
//   Total           : 4 clock cycles from pixel_in to edge_out
//
// TARGET BOARD: Xilinx Artix-7 xc7a35tcpg236-1
//   LUT Elements : 20800 available
//   Flip-Flops   : 41600 available
//   Block RAMs   : 50 × 36Kb available
//
// CLOCK: 100 MHz (10ns period)
//   1 pixel processed per clock = 10ns per pixel
//   249,696 pixel image (612×408) = 2.497ms total latency
//
// PORTS:
//   clk          : 100MHz system clock
//   rst          : Active-high synchronous reset
//   start        : Pulse to begin processing
//   pixel_valid  : High when pixel_in is valid (each clock during load)
//   pixel_in     : Input grayscale pixel (8-bit)
//   edge_valid   : High when edge_out is valid
//   edge_out     : 1=edge detected, 0=no edge
//   magnitude_out: Gradient magnitude (0-255)
//   done         : Pulses high when full image processed
// ============================================================

module sobel_top #(
    parameter IMG_WIDTH  = 64,   // Image width in pixels
    parameter IMG_HEIGHT = 64,   // Image height in pixels
    parameter THRESHOLD  = 80    // Edge detection threshold (0-255)
)(
    input  wire       clk,          // 100MHz system clock
    input  wire       rst,          // Reset (active high)
    input  wire       start,        // Start signal (pulse)
    input  wire       pixel_valid,  // Input pixel valid flag
    input  wire [7:0] pixel_in,     // Input grayscale pixel

    output wire       edge_valid,      // Edge output valid flag
    output wire       edge_out,        // Edge detected (1=yes, 0=no)
    output wire [7:0] magnitude_out,   // Gradient magnitude
    output wire       done             // Processing complete
);

    // ── INTERNAL SIGNAL DECLARATIONS ─────────────────────────

    // Control signals from FSM to each module
    wire ctrl_load;    // → image_buffer write enable
    wire ctrl_proc;    // → window_extractor valid_in
    wire ctrl_out;     // → output enable (same as proc_en)

    // Image buffer address registers
    reg  [13:0] wr_addr;   // Write address (auto-increments with each pixel)
    reg  [13:0] rd_addr;   // Read address  (auto-increments during processing)
    wire [7:0]  buf_out;   // Pixel read from buffer → window extractor

    // 3×3 window pixels from window_extractor
    wire [7:0] p00, p01, p02;   // Top row
    wire [7:0] p10, p11, p12;   // Middle row
    wire [7:0] p20, p21, p22;   // Bottom row
    wire win_valid;              // Window valid flag

    // Convolution outputs (signed 12-bit gradients)
    wire signed [11:0] gx, gy;
    wire conv_valid;             // Convolution valid flag

    // ── MODULE INSTANTIATIONS ─────────────────────────────────

    // ── 1. CONTROL UNIT FSM ───────────────────────────────────
    // Coordinates all other modules via enable signals
    control_unit #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_ctrl (
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .pixel_valid(pixel_valid),
        .load_en    (ctrl_load),    // Enables buffer writes
        .proc_en    (ctrl_proc),    // Enables pipeline
        .out_en     (ctrl_out),     // Enables output capture
        .done       (done)          // Completion signal
    );

    // ── 2. IMAGE BUFFER (Dual-port RAM) ───────────────────────
    // Stores the input image while pipeline processes it
    image_buffer #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_buf (
        .clk    (clk),
        .we     (pixel_valid & ctrl_load),  // Write when valid AND load enabled
        .wr_addr(wr_addr),
        .wr_data(pixel_in),
        .rd_addr(rd_addr),
        .rd_data(buf_out)           // Goes to window_extractor
    );

    // Address counter logic for image buffer
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_addr <= 14'd0;
            rd_addr <= 14'd0;
        end else begin
            // Increment write address each time a valid pixel is written
            if (pixel_valid && ctrl_load)
                wr_addr <= wr_addr + 14'd1;
            // Increment read address each processing clock
            if (ctrl_proc)
                rd_addr <= rd_addr + 14'd1;
        end
    end

    // ── 3. WINDOW EXTRACTOR ───────────────────────────────────
    // Generates 3×3 sliding window from streaming pixel data
    // Uses two line buffers to hold previous two rows
    window_extractor #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_win (
        .clk      (clk),
        .rst      (rst),
        .valid_in (ctrl_proc),  // Enable when FSM says process
        .pixel_in (buf_out),    // Pixels from image buffer
        // 3×3 window outputs to convolution engine
        .p00(p00), .p01(p01), .p02(p02),
        .p10(p10), .p11(p11), .p12(p12),
        .p20(p20), .p21(p21), .p22(p22),
        .valid_out(win_valid)
    );

    // ── 4. CONVOLUTION ENGINE ─────────────────────────────────
    // Computes Gx and Gy simultaneously in one clock cycle
    // No multipliers — only additions, subtractions, shifts
    convolution_engine u_conv (
        .clk      (clk),
        .rst      (rst),
        .valid_in (win_valid),  // Process when window is valid
        // 3×3 window inputs
        .p00(p00), .p01(p01), .p02(p02),
        .p10(p10), .p11(p11), .p12(p12),
        .p20(p20), .p21(p21), .p22(p22),
        // Gradient outputs
        .gx       (gx),
        .gy       (gy),
        .valid_out(conv_valid)
    );

    // ── 5. EDGE DETECTOR ─────────────────────────────────────
    // Computes |Gx|+|Gy| magnitude, applies threshold
    // Produces binary edge map: 1=edge, 0=background
    edge_detector #(
        .THRESHOLD(THRESHOLD)
    ) u_edge (
        .clk      (clk),
        .rst      (rst),
        .valid_in (conv_valid),   // Process when gradients are valid
        .gx       (gx),
        .gy       (gy),
        .magnitude(magnitude_out),
        .edge_out (edge_out),
        .valid_out(edge_valid)
    );

endmodule
