# ============================================================
# File    : artix7_constraints.xdc
# Target  : Xilinx Artix-7 xc7a35tcpg236-1
# Project : FPGA-Based Sobel Edge Detection
# ============================================================

# 100MHz clock
create_clock -period 10.000 -name sys_clk [get_ports clk]
set_clock_uncertainty 0.5 [get_clocks sys_clk]
set_input_delay  -clock sys_clk 2.0 [all_inputs]
set_output_delay -clock sys_clk 2.0 [all_outputs]

# Pin assignments (Basys3 / Nexys A7)
set_property PACKAGE_PIN W5      [get_ports clk]
set_property IOSTANDARD  LVCMOS33 [get_ports clk]
set_property PACKAGE_PIN U18     [get_ports rst]
set_property IOSTANDARD  LVCMOS33 [get_ports rst]
set_property PACKAGE_PIN T18     [get_ports start]
set_property IOSTANDARD  LVCMOS33 [get_ports start]
set_property PACKAGE_PIN U16     [get_ports done]
set_property IOSTANDARD  LVCMOS33 [get_ports done]
set_property PACKAGE_PIN E19     [get_ports edge_out]
set_property IOSTANDARD  LVCMOS33 [get_ports edge_out]

set_false_path -from [get_ports rst]
set_property CFGBVS        VCCO  [current_design]
set_property CONFIG_VOLTAGE 3.3  [current_design]
