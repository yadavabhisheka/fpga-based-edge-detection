// ============================================================
// Testbench : tb_sobel_top.v
// Purpose   : Full system simulation
//             Reads input_pixels.hex (from img_to_hex.py)
//             Writes hw_edges.hex (read by hex_to_img.py)
//             Prints hardware latency only
//             Software latency + speedup handled by Python
//
// Things to update:
//   1. INPUT_HEX  path - point to your simulation folder
//   2. OUTPUT_HEX path - same simulation folder
//   Never change IMG_WIDTH/HEIGHT (always 64 from img_to_hex.py)
// ============================================================
`timescale 1ns/1ps

module tb_sobel_top;

    // ── NEVER CHANGE THESE (img_to_hex.py always uses 64x64) ─
    parameter IMG_WIDTH  = 64;
    parameter IMG_HEIGHT = 64;
    parameter THRESHOLD  = 120;
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz
    // ─────────────────────────────────────────────────────────

    // ── UPDATE THESE PATHS ONCE TO MATCH YOUR SYSTEM ─────────
    parameter INPUT_HEX  = "/home/yadavabhishekaa/Downloads/sobel_project/hardware/simulation/input_pixels.hex";
    parameter OUTPUT_HEX = "/home/yadavabhishekaa/Downloads/sobel_project/hardware/simulation/hw_edges.hex";
    // ─────────────────────────────────────────────────────────

    // DUT signals
    reg        clk, rst, start, pixel_valid;
    reg  [7:0] pixel_in;
    wire       edge_valid, edge_out, done;
    wire [7:0] magnitude_out;

    // DUT instantiation
    sobel_top #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .THRESHOLD(THRESHOLD)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .pixel_valid(pixel_valid), .pixel_in(pixel_in),
        .edge_valid(edge_valid), .edge_out(edge_out),
        .magnitude_out(magnitude_out), .done(done)
    );

    // 100MHz clock
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Memory arrays
    reg [7:0] input_pixels [0:IMG_WIDTH*IMG_HEIGHT-1];
    reg [7:0] hw_output    [0:IMG_WIDTH*IMG_HEIGHT-1];

    integer i, start_time, end_time;
    integer edge_count, out_idx, fd;

    initial begin
        // Load real image pixels from hex file
        $readmemh(INPUT_HEX, input_pixels);

        $display("==============================================");
        $display("  FPGA SOBEL EDGE DETECTION SIMULATION");
        $display("  Image: %0dx%0d | Threshold:%0d | 100MHz",
                 IMG_WIDTH, IMG_HEIGHT, THRESHOLD);
        $display("==============================================");

        // Initialize everything
        rst=1; start=0; pixel_valid=0;
        pixel_in=0; edge_count=0; out_idx=0;

        for (i=0; i<IMG_WIDTH*IMG_HEIGHT; i=i+1)
            hw_output[i] = 8'd0;

        // Reset pulse - 5 clock cycles
        repeat(5) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        // Pulse start for exactly 1 clock cycle
        @(negedge clk); start = 1;
        @(posedge clk);
        @(negedge clk); start = 0;

        // Record start time for latency measurement
        start_time = $time;

        // Wait for FSM to enter LOAD state
        repeat(3) @(posedge clk);

        // ── PHASE 1: Feed all pixels into image buffer ────────
        // pixel_valid stays HIGH for all IMG_WIDTH*IMG_HEIGHT pixels
        // FSM is in LOAD state - proc_en=0, pipeline is OFF
        pixel_valid = 1;
        for (i=0; i<IMG_WIDTH*IMG_HEIGHT; i=i+1) begin
            @(negedge clk);              // Set input at negedge
            pixel_in = input_pixels[i]; // One pixel per clock
            @(posedge clk);              // DUT samples here
        end
        @(negedge clk);
        pixel_valid = 0;   // All pixels sent

        // ── PHASE 2: Wait for PROCESS + FLUSH states ─────────
        // FSM auto-transitions: LOAD → PROCESS → FLUSH → DONE
        // PROCESS: IMG_WIDTH*IMG_HEIGHT cycles (pipeline runs)
        // FLUSH  : IMG_WIDTH*3+10 extra cycles (pipeline drains)
        repeat(IMG_WIDTH*IMG_HEIGHT + IMG_WIDTH*3 + 50) @(posedge clk);

        end_time = $time;

        // ── Write hardware output to hex file ─────────────────
        // hex_to_img.py reads this to generate comparison image
        fd = $fopen(OUTPUT_HEX, "w");
        if (fd == 0) begin
            $display("[ERROR] Cannot open output file: %s", OUTPUT_HEX);
            $display("[INFO]  Check folder exists and path is correct");
        end else begin
            for (i=0; i<IMG_WIDTH*IMG_HEIGHT; i=i+1)
                $fwrite(fd, "%02x\n", hw_output[i]);
            $fclose(fd);
            $display("[SAVED] hw_edges.hex written successfully");
        end

        // ── Print hardware results only ───────────────────────
        // Speedup calculation done in Python by hex_to_img.py
        // which reads both sw_latency and hw_latency from image_info.txt
        $display("");
        $display("  HARDWARE RESULTS:");
        $display("  Clock cycles : %0d",
                 (end_time-start_time)/CLK_PERIOD);
        $display("  Sim time     : %0.4f ms",
                 (end_time-start_time)/1000000.0);
        $display("  Edge pixels  : %0d / %0d",
                 edge_count, IMG_WIDTH*IMG_HEIGHT);
        $display("");
        $display("  ACTUAL FPGA LATENCY (1px/clock @ 100MHz):");
        $display("  64x64   (%0d px) : %0.4f ms",
                 IMG_WIDTH*IMG_HEIGHT,
                 IMG_WIDTH*IMG_HEIGHT*CLK_PERIOD/1000000.0);
        $display("  640x480 (307200 px) : 3.072 ms");
        $display("");
        $display("  Run hex_to_img.py for speedup comparison");
        $display("==============================================");
        $finish;
    end

    // ── Capture edge output pixels ────────────────────────────
    // Runs every clock - stores each valid edge pixel
    // edge=1 → 0xFF (white, edge detected)
    // edge=0 → 0x00 (black, no edge)
    always @(posedge clk) begin
        if (edge_valid) begin
            if (out_idx < IMG_WIDTH*IMG_HEIGHT) begin
                hw_output[out_idx] <= edge_out ? 8'hFF : 8'h00;
                out_idx <= out_idx + 1;
            end
            if (edge_out)
                edge_count <= edge_count + 1;
        end
    end

    // Waveform dump for Vivado waveform viewer
    initial begin
        $dumpfile("sobel_sim.vcd");
        $dumpvars(0, tb_sobel_top);
    end

endmodule