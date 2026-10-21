# Vertical slice spec: one 8-input binary neuron

This file is the contract. `model/reference.py`, `model/train.py`,
`model/export.py`, `src/project.v` and `firmware/host_example.c` all implement
exactly what is written here. If they ever disagree, this file decides who
is wrong.

## 1. Numbers

| Thing          | Math value   | Stored as                         |
|----------------|--------------|-----------------------------------|
| input x[i]     | -1 or +1     | 1 bit: 1 means +1, 0 means -1     |
| weight w[i]    | -1 or +1     | 1 bit: 1 means +1, 0 means -1     |
| threshold T    | 0 .. 9       | 4-bit unsigned integer            |
| popcount pc    | 0 .. 8       | 4-bit unsigned integer            |
| output y       | 0 or 1       | 1 bit                             |

There are 8 inputs and 8 weights, i = 0..7.

## 2. Bit order

Input i and weight i always live in **bit i** of their byte.
Bit 0 is the least significant bit (LSB). So the byte `0b0000_0001`
means x[0] = +1 and x[1..7] = -1.

## 3. The computation

    match = XNOR(x_byte, w_byte)        # bit i is 1 when x[i] == w[i]
    pc    = number of 1 bits in match   # popcount, 0..8
    y     = 1 if pc >= T else 0

Why this equals a neuron: for +/-1 values, sum(x[i]*w[i]) = 2*pc - 8.
A neuron with bias b fires when 2*pc - 8 + b >= 0, which is the same as
pc >= (8 - b) / 2. Because pc is a whole number, that is pc >= T with

    T = ceil((8 - b) / 2), clipped to 0..9

(T = 0 means "always fire", T = 9 means "never fire".)

No overflow is possible: pc never exceeds 8 and fits in 4 bits.

## 4. Pins (Tiny Tapeout interface)

| Pin          | Direction | Meaning                                  |
|--------------|-----------|------------------------------------------|
| ui_in[7:0]   | in        | data byte (inputs, weights or threshold) |
| uio_in[1:0]  | in        | mode (see below)                         |
| uio_in[7:2]  | in        | unused                                   |
| uo_out[0]    | out       | y, the neuron's output                   |
| uo_out[4:1]  | out       | pc, the popcount (for debugging)         |
| uo_out[7:5]  | out       | always 0                                 |
| uio_out/oe   | out       | always 0 (all uio pins are inputs)       |

Modes, sampled on each rising clock edge:

| uio_in[1:0] | Name   | What happens on the rising edge                    |
|-------------|--------|----------------------------------------------------|
| 00          | RUN    | compute with x = ui_in; store pc and y in registers |
| 01          | LOAD_W | weight register <= ui_in                            |
| 10          | LOAD_T | threshold register <= ui_in[3:0]                    |
| 11          | HOLD   | nothing changes                                     |

Timing: results are registered, so y and pc appear on uo_out after the
rising edge that sampled the input (one clock of latency).

Reset (rst_n low, synchronous): weights = 0, threshold = 0, pc = 0, y = 0.

## 5. Configuration "bitstream"

For this slice the whole model is 2 bytes, sent in this order:

1. weight byte, with mode LOAD_W
2. threshold byte (value in the low 4 bits), with mode LOAD_T
