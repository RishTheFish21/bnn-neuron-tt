/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * One 8-input binary neuron for Tiny Tapeout.
 * Behaviour is defined by SPEC.md. Before submitting to a shuttle, rename the module to
 * tt_um_<yourgithubname>_bnn and update test/tb.v and info.yaml to match.
 */
`default_nettype none

module tt_um_bnn_neuron (
    input  wire [7:0] ui_in,    // data byte
    output wire [7:0] uo_out,   // {3'b000, pc[3:0], y}
    input  wire [7:0] uio_in,   // [1:0] = mode
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,      // high when this design is selected; unused
    input  wire       clk,
    input  wire       rst_n     // active-low reset
);

  // Modes on uio_in[1:0] (spec.md section 4)
  localparam [1:0] MODE_RUN    = 2'b00;
  localparam [1:0] MODE_LOAD_W = 2'b01;
  localparam [1:0] MODE_LOAD_T = 2'b10;
  localparam [1:0] MODE_HOLD   = 2'b11;

  wire [1:0] mode = uio_in[1:0];

  // ---- Storage (flip-flops) ----
  reg [7:0] w_reg;   // weights: bit i = weight i, 1 means +1
  reg [3:0] t_reg;   // threshold, 0..9
  reg [3:0] pc_reg;  // last popcount
  reg       y_reg;   // last output

  // ---- Compute (pure combinational logic) ----
  // XNOR: bit i is 1 when input i agrees with weight i
  wire [7:0] match = ~(ui_in ^ w_reg);

  // Popcount: count the 1 bits. Synthesis turns this loop into an adder tree.
  function [3:0] popcount8(input [7:0] v);
    integer i;
    begin
      popcount8 = 4'd0;
      for (i = 0; i < 8; i = i + 1)
        popcount8 = popcount8 + {3'b000, v[i]};
    end
  endfunction

  wire [3:0] pc     = popcount8(match);
  wire       fires  = (pc >= t_reg);

  // ---- Sequential logic: everything updates on the rising clock edge ----
  always @(posedge clk) begin
    if (!rst_n) begin
      w_reg  <= 8'd0;
      t_reg  <= 4'd0;
      pc_reg <= 4'd0;
      y_reg  <= 1'b0;
    end else begin
      case (mode)
        MODE_RUN: begin
          pc_reg <= pc;
          y_reg  <= fires;
        end
        MODE_LOAD_W: w_reg <= ui_in;
        MODE_LOAD_T: t_reg <= ui_in[3:0];
        MODE_HOLD:   ;  // nothing changes
      endcase
    end
  end

  // ---- Outputs ----
  assign uo_out  = {3'b000, pc_reg, y_reg};
  assign uio_out = 8'd0;
  assign uio_oe  = 8'd0;  // all uio pins are inputs

  // Tell the linter these inputs are intentionally unused
  wire _unused = &{ena, uio_in[7:2], 1'b0};

endmodule
