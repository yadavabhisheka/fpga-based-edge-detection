// ============================================================
// FILE    : tb_convolution_engine.v
// PROJECT : FPGA-Based Sobel Edge Detection
// TYPE    : Unit Testbench (not synthesized — simulation only)
// PURPOSE : Verifies that convolution_engine.v correctly computes
//           Sobel Gx and Gy for known 3×3 pixel windows.
//
// TEST CASES:
//   Test 1 — Vertical Edge:
//     Left half dark (100), right half bright (200)
//     Expected Gx = 400 (strong horizontal change)
//     Expected Gy = 0   (no vertical change)
//
//   Test 2 — Horizontal Edge:
//     Top half dark (100), bottom half bright (200)
//     Expected Gx = 0   (no horizontal change)
//     Expected Gy = 400 (strong vertical change)
//
//   Test 3 — Uniform Region:
//     All pixels same (128)
//     Expected Gx = 0, Gy = 0 (no edges)
//
// MANUAL VERIFICATION (Test 1):
//   Gx = (p02-p00) + 2*(p12-p10) + (p22-p20)
//      = (200-100) + 2*(200-100) + (200-100)
//      = 100 + 200 + 100 = 400 ✓
//
// HOW TO RUN IN VIVADO:
//   Sources → right click tb_convolution_engine → Set as Top
//   Flow Navigator → Run Simulation → Run Behavioral Simulation
//   Tcl console: run 1us
//   Check output for PASS/FAIL
//
// HOW TO RUN WITH ICARUS:
//   iverilog -o conv_test tb_convolution_engine.v convolution_engine.v
//   vvp conv_test
// ============================================================

`timescale 1ns / 1ps   // Time unit = 1ns, precision = 1ps

module tb_convolution_engine;

    // ── DUT SIGNAL DECLARATIONS ───────────────────────────────
    reg        clk;        // Test clock
    reg        rst;        // Reset
    reg        valid_in;   // Input valid flag

    // 3×3 pixel window inputs (unsigned 8-bit)
    reg [7:0]  p00, p01, p02;
    reg [7:0]  p10, p11, p12;
    reg [7:0]  p20, p21, p22;

    // DUT outputs
    wire signed [11:0] gx;    // Horizontal gradient
    wire signed [11:0] gy;    // Vertical gradient
    wire               valid_out;

    // ── DUT INSTANTIATION ─────────────────────────────────────
    convolution_engine dut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .p00(p00), .p01(p01), .p02(p02),
        .p10(p10), .p11(p11), .p12(p12),
        .p20(p20), .p21(p21), .p22(p22),
        .gx(gx), .gy(gy), .valid_out(valid_out)
    );

    // ── CLOCK GENERATION ─────────────────────────────────────
    // 10ns period = 100MHz (matches hardware target)
    initial clk = 0;
    always #5 clk = ~clk;

    // ── INITIALIZE INPUTS ─────────────────────────────────────
    initial begin
        rst=1; valid_in=0;
        p00=0; p01=0; p02=0;
        p10=0; p11=0; p12=0;
        p20=0; p21=0; p22=0;
    end

    // ── TASK: Apply window and check result ───────────────────
    // Sets a 3×3 window, waits for registered output,
    // then checks against expected values and prints PASS/FAIL
    task apply_and_check;
        // 9 pixel inputs
        input [7:0] i00, i01, i02;
        input [7:0] i10, i11, i12;
        input [7:0] i20, i21, i22;
        input [200:0] test_name;          // Test description string
        input signed [11:0] exp_gx;       // Expected Gx value
        input signed [11:0] exp_gy;       // Expected Gy value
        begin
            // Set inputs at negedge to avoid race conditions with posedge clock
            // This ensures inputs are stable before the next rising edge samples them
            @(negedge clk);
            valid_in = 1;
            p00=i00; p01=i01; p02=i02;
            p10=i10; p11=i11; p12=i12;
            p20=i20; p21=i21; p22=i22;

            @(posedge clk);  // DUT samples inputs on this edge
            @(negedge clk);
            valid_in = 0;    // Deassert valid

            @(posedge clk);  // Wait 1 cycle for registered output
            #1;              // Small delay to let output stabilize

            // Print test results
            $display("[%s]", test_name);
            $display("  Gx = %0d  (expected %0d)", $signed(gx), exp_gx);
            $display("  Gy = %0d  (expected %0d)", $signed(gy), exp_gy);

            if ($signed(gx) == exp_gx && $signed(gy) == exp_gy)
                $display("  RESULT: PASS");
            else
                $display("  RESULT: FAIL ← check convolution_engine.v");
            $display("");
        end
    endtask

    // ── MAIN TEST SEQUENCE ────────────────────────────────────
    initial begin
        // Enable waveform dump for GTKWave / Vivado waveform viewer
        $dumpfile("conv_sim.vcd");
        $dumpvars(0, tb_convolution_engine);

        // Hold reset for 4 cycles then release
        repeat(4) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        $display("==============================================");
        $display("  CONVOLUTION ENGINE UNIT TESTS");
        $display("  Project: FPGA-Based Sobel Edge Detection");
        $display("==============================================");
        $display("");

        // ── TEST 1: Vertical Edge ─────────────────────────────
        // Window: dark left half (100), bright right half (200)
        // Gx should be large (strong horizontal change = vertical edge)
        // Gy should be 0 (uniform top-to-bottom)
        apply_and_check(
            100, 100, 200,
            100, 100, 200,
            100, 100, 200,
            "Test 1 - Vertical Edge (Gx=400, Gy=0)",
            12'sd400, 12'sd0
        );

        // ── TEST 2: Horizontal Edge ───────────────────────────
        // Window: dark top half (100), bright bottom half (200)
        // Gx should be 0 (uniform left-to-right)
        // Gy should be large (strong vertical change = horizontal edge)
        apply_and_check(
            100, 100, 100,
            100, 100, 100,
            200, 200, 200,
            "Test 2 - Horizontal Edge (Gx=0, Gy=400)",
            12'sd0, 12'sd400
        );

        // ── TEST 3: Uniform Region ────────────────────────────
        // All pixels same intensity — no edges anywhere
        // Both Gx and Gy should be exactly 0
        apply_and_check(
            128, 128, 128,
            128, 128, 128,
            128, 128, 128,
            "Test 3 - Uniform (No Edge, Gx=0, Gy=0)",
            12'sd0, 12'sd0
        );

        $display("==============================================");
        $display("  All unit tests complete.");
        $display("  If all PASS: convolution_engine.v is correct.");
        $display("  Proceed to run tb_sobel_top.v next.");
        $display("==============================================");
        $finish;
    end

endmodule
