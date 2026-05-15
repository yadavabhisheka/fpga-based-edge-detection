// ============================================================
// FILE    : image_buffer.v
// PROJECT : FPGA-Based Sobel Edge Detection
// MODULE  : image_buffer
// PURPOSE : Dual-port Block RAM to store the input grayscale
//           image before pipeline processing begins.
//
// WHY DUAL-PORT?
//   Port A (write): Receives pixels from testbench/camera
//   Port B (read) : Feeds pixels to window_extractor
//   Both can operate simultaneously — write new image while
//   reading previous image for processing (ping-pong possible)
//
// MEMORY SIZE:
//   128x128 pixels = 16,384 locations (configurable via params)
//   Each location stores 8 bits (one grayscale pixel, 0-255)
//   Total = 16,384 bytes = 16 KB
//   Artix-7 has 50 BRAMs of 36Kb each — this uses < 1 BRAM
//
// TIMING:
//   Write: data appears in memory on next posedge after we=1
//   Read : data appears on rd_data one cycle after rd_addr set
//   (registered output — 1 cycle read latency)
//
// PORTS:
//   clk      : System clock (100MHz)
//   we       : Write enable (1=write, 0=no write)
//   wr_addr  : Write address (14-bit → 0 to 16383)
//   wr_data  : Pixel data to write (8-bit, 0-255)
//   rd_addr  : Read address (14-bit)
//   rd_data  : Pixel data output (8-bit, registered)
// ============================================================

module image_buffer #(
    parameter IMG_WIDTH  = 64,    // Image width in pixels
    parameter IMG_HEIGHT = 64,    // Image height in pixels
    parameter DATA_WIDTH = 8      // Bits per pixel (8 = grayscale)
)(
    input  wire                  clk,      // 100MHz clock
    input  wire                  we,       // Write enable
    input  wire [13:0]           wr_addr,  // Write address (14-bit = up to 16383)
    input  wire [DATA_WIDTH-1:0] wr_data,  // Pixel to write
    input  wire [13:0]           rd_addr,  // Read address
    output reg  [DATA_WIDTH-1:0] rd_data   // Pixel output (1 cycle latency)
);

    // Memory array: total pixels × 8 bits each
    // Vivado infers this as Block RAM automatically
    reg [DATA_WIDTH-1:0] mem [0:(IMG_WIDTH * IMG_HEIGHT) - 1];

    // ── WRITE PORT ───────────────────────────────────────────
    // Synchronous write: on every rising clock edge,
    // if write-enable is high, store wr_data at wr_addr
    always @(posedge clk) begin
        if (we)
            mem[wr_addr] <= wr_data;
    end

    // ── READ PORT ────────────────────────────────────────────
    // Synchronous read: rd_data appears ONE CLOCK after rd_addr
    // This 1-cycle latency is accounted for in sobel_top.v timing
    always @(posedge clk) begin
        rd_data <= mem[rd_addr];
    end

endmodule
