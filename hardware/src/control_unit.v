// ============================================================
// Module   : control_unit.v
// Fix      : LOAD and PROCESS are separate states
//            Load ALL pixels into buffer first (proc_en=0)
//            Then process from buffer (proc_en=1)
//            This fixes edge pixels = 0 issue
// FSM      : IDLE → LOAD → PROCESS → FLUSH → DONE_ST
// ============================================================
module control_unit #(
    parameter IMG_WIDTH  = 64,
    parameter IMG_HEIGHT = 64
)(
    input  wire clk, rst, start, pixel_valid,
    output reg  load_en, proc_en, out_en, done
);
    // State encoding
    localparam IDLE    = 3'd0;
    localparam LOAD    = 3'd1;  // Write pixels to buffer only
    localparam PROCESS = 3'd2;  // Read buffer, run pipeline
    localparam FLUSH   = 3'd3;  // Wait for pipeline to drain
    localparam DONE_ST = 3'd4;  // Pulse done, return to idle

    // Total pixels in image
    localparam TOTAL = IMG_WIDTH * IMG_HEIGHT;

    reg [2:0]  state;
    reg [16:0] pixel_cnt;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state     <= IDLE;
            pixel_cnt <= 0;
            load_en   <= 0;
            proc_en   <= 0;
            out_en    <= 0;
            done      <= 0;
        end else begin
            case (state)

                // Wait for start pulse, all outputs off
                IDLE: begin
                    done    <= 0;
                    load_en <= 0;
                    proc_en <= 0;
                    out_en  <= 0;
                    if (start) begin
                        state     <= LOAD;
                        pixel_cnt <= 0;
                    end
                end

                // Load pixels into buffer one by one
                // proc_en = 0 - pipeline is OFF during load
                // Ensures buffer is full before pipeline reads it
                LOAD: begin
                    load_en <= 1;
                    proc_en <= 0;   // KEY: pipeline OFF during load
                    out_en  <= 0;
                    if (pixel_valid) begin
                        if (pixel_cnt == TOTAL - 1) begin
                            state     <= PROCESS;
                            pixel_cnt <= 0;
                        end else
                            pixel_cnt <= pixel_cnt + 1;
                    end
                end

                // All pixels loaded - now run pipeline
                // load_en = 0 - no more buffer writes
                // proc_en = 1 - pipeline reads buffer
                PROCESS: begin
                    load_en <= 0;
                    proc_en <= 1;   // Pipeline ON
                    out_en  <= 1;
                    if (pixel_cnt == TOTAL - 1) begin
                        state     <= FLUSH;
                        pixel_cnt <= 0;
                    end else
                        pixel_cnt <= pixel_cnt + 1;
                end

                // Wait for pipeline to fully drain
                // Window extractor needs 2 extra rows to finish
                // Extra cycles = IMG_WIDTH * 3 + 10
                FLUSH: begin
                    load_en <= 0;
                    proc_en <= 1;
                    out_en  <= 1;
                    if (pixel_cnt == IMG_WIDTH * 3 + 10) begin
                        state   <= DONE_ST;
                        proc_en <= 0;
                        out_en  <= 0;
                    end else
                        pixel_cnt <= pixel_cnt + 1;
                end

                // Pulse done for exactly 1 clock cycle
                // Then return to IDLE ready for next image
                DONE_ST: begin
                    done    <= 1;
                    load_en <= 0;
                    proc_en <= 0;
                    out_en  <= 0;
                    state   <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end
endmodule