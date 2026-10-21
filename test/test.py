# SPDX-License-Identifier: Apache-2.0
"""Co-verification: the simulated chip vs the Python reference model.

Run from the test/ folder:  make
Uses test/test_vectors.json, which model/export.py writes.
"""
import json
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR.parent / "model"))
from reference import neuron  # noqa: E402  (the answer key)

MODE_RUN, MODE_LOAD_W, MODE_LOAD_T, MODE_HOLD = 0, 1, 2, 3


def as_int(value):
    """Read a signal as an unsigned integer (works on cocotb 2.x and 1.x)."""
    return value.to_unsigned() if hasattr(value, "to_unsigned") else int(value)


async def start(dut):
    """Start a 10 MHz clock and hold reset for a few cycles."""
    cocotb.start_soon(Clock(dut.clk, 100, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


async def step(dut, mode, data):
    """Drive the pins on a falling edge, let one rising edge happen, then return.

    Changing inputs on the falling edge keeps them steady around the rising
    edge, which is when the chip reads them. When this returns, the chip's
    registers have updated and uo_out is safe to read.
    """
    await FallingEdge(dut.clk)
    dut.uio_in.value = mode
    dut.ui_in.value = data
    await FallingEdge(dut.clk)


def read_outputs(dut):
    out = as_int(dut.uo_out.value)
    return out & 1, (out >> 1) & 0xF, out >> 5  # y, pc, unused bits


async def check_all_inputs(dut, w_byte, threshold, expected=None):
    """Load a model into the chip, then compare all 256 inputs with the reference."""
    await step(dut, MODE_LOAD_W, w_byte)
    await step(dut, MODE_LOAD_T, threshold)
    for x in range(256):
        await step(dut, MODE_RUN, x)
        y, pc, top = read_outputs(dut)
        ref_y, ref_pc = neuron(x, w_byte, threshold)
        assert top == 0, f"uo_out[7:5] should be 0, got {top}"
        assert (y, pc) == (ref_y, ref_pc), (
            f"MISMATCH W={w_byte:08b} T={threshold} x={x:08b}: "
            f"chip y={y} pc={pc}, reference y={ref_y} pc={ref_pc}"
        )
        if expected is not None:
            assert (y, pc) == (expected[x]["y"], expected[x]["pc"]), f"test_vectors.json disagrees at x={x}"


@cocotb.test()
async def test_trained_model(dut):
    """Load the exported (trained) model and compare on all 256 inputs."""
    cfg = json.loads((TEST_DIR / "test_vectors.json").read_text())
    await start(dut)
    await check_all_inputs(dut, cfg["w_byte"], cfg["threshold"], cfg["vectors"])
    dut._log.info("trained model: chip matches reference on all 256 inputs")


@cocotb.test()
async def test_random_configs(dut):
    """The chip must be correct for ANY weights, not just the trained ones."""
    await start(dut)
    rng = random.Random(1234)
    configs = [(0x00, 0), (0xFF, 9), (0xAA, 4), (0x55, 8)]  # edge cases first
    configs += [(rng.randrange(256), rng.randrange(10)) for _ in range(20)]
    for w, t in configs:
        await check_all_inputs(dut, w, t)
    dut._log.info(f"{len(configs)} configs x 256 inputs all match")


@cocotb.test()
async def test_hold_and_reset(dut):
    """HOLD mode must not change anything; reset must clear everything."""
    await start(dut)
    await step(dut, MODE_LOAD_W, 0xFF)
    await step(dut, MODE_LOAD_T, 8)
    await step(dut, MODE_RUN, 0xFF)  # every bit matches: pc 8, y 1
    assert read_outputs(dut)[:2] == (1, 8), "RUN with W=0xFF, T=8, x=0xFF should give y=1, pc=8"
    await step(dut, MODE_HOLD, 0x00)  # HOLD with different data on the pins
    assert read_outputs(dut)[:2] == (1, 8), "HOLD changed the outputs"
    await step(dut, MODE_RUN, 0xFF)  # the weights must not have been overwritten
    assert read_outputs(dut)[:2] == (1, 8), "HOLD changed the weights"

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)
    assert read_outputs(dut) == (0, 0, 0), "reset did not clear the outputs"
    await step(dut, MODE_RUN, 0x00)  # after reset W=0x00 and T=0, so x=0x00 matches all 8
    assert read_outputs(dut)[:2] == (1, 8), "reset did not clear W and T to 0"
