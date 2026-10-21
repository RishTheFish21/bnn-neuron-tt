<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This chip is one neuron from a binary neural network, trained in PyTorch and then loaded into the chip.

The neuron has 8 inputs and 8 weights, and each one is either +1 or -1. A 1 bit means +1 and a 0 bit means -1. Input
bit i is paired with weight bit i.

On each run, the chip:

1. XNORs the input byte with the weight byte. Each result bit is 1 where the input agrees with the weight.
2. Counts the 1 bits (the popcount, 0 to 8).
3. Outputs Y = 1 if the popcount is at least the threshold T, otherwise Y = 0.

This is the same as a normal neuron with +1/-1 values: the sum of input × weight equals 2 × popcount - 8, and the bias
is folded into the threshold T. The weights and threshold come from quantization-aware training (see `model/` in the
repository). The exact behaviour is defined in `SPEC.md`.

The two mode pins choose what the byte on `ui_in` means:

| uio[1:0] | Mode   | On the rising clock edge                          |
|----------|--------|---------------------------------------------------|
| 00       | RUN    | Compute with input = `ui_in`, store POPCOUNT and Y |
| 01       | LOAD_W | Weights = `ui_in`                                  |
| 10       | LOAD_T | Threshold = `ui_in[3:0]` (0 to 9)                  |
| 11       | HOLD   | Nothing changes                                    |

Outputs are registered, so they appear one clock after the input is sampled. Reset clears the weights, threshold and
outputs to 0.

## How to test

1. Hold `rst_n` low for a few clocks, then release it.
2. Set `uio[1:0]` = 01, put the weight byte on `ui_in`, and give one clock.
3. Set `uio[1:0]` = 10, put the threshold on `ui_in`, and give one clock.
4. Set `uio[1:0]` = 00, put an input byte on `ui_in`, and give one clock. Read Y on `uo[0]` and the popcount on
   `uo[4:1]`.

Example: weights 0xFF, threshold 8, input 0xFF gives popcount 8 and Y = 1. Input 0xFE gives popcount 7 and Y = 0.

`firmware/host_example.c` in the repository shows the same sequence in C for a host microcontroller.

## External hardware

None. The pins can be driven by the demo board's microcontroller.
