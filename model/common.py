"""Shared helpers: bit conversions and the toy dataset.

The toy task: a hidden "teacher" neuron with secret weights decides the
label of every 8-bit input. We add a little label noise, then ask training
to rediscover the teacher. Because there are only 256 possible inputs, we
can later test the chip on every single one of them.
"""
import numpy as np

N_INPUTS = 8

# The secret rule. Training never sees these directly, only the labels.
TEACHER_W = 0b1011_0010
TEACHER_T = 5
LABEL_NOISE = 0.05  # fraction of training labels flipped on purpose


def byte_to_pm1(b):
    """0b...1 -> [+1, ...]. Bit i (LSB = 0) becomes element i."""
    return np.array([1 if (b >> i) & 1 else -1 for i in range(N_INPUTS)], dtype=np.int64)


def pm1_to_byte(v):
    """Inverse of byte_to_pm1. Element i becomes bit i."""
    out = 0
    for i, val in enumerate(v):
        if val > 0:
            out |= 1 << i
    return out


def teacher_label(x_byte):
    match = ~(x_byte ^ TEACHER_W) & 0xFF
    return int(bin(match).count("1") >= TEACHER_T)


def make_train_set(n=4000, seed=0):
    """Random inputs with noisy teacher labels. Returns (bytes, labels)."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 256, size=n)
    ys = np.array([teacher_label(int(x)) for x in xs])
    flip = rng.random(n) < LABEL_NOISE
    ys = np.where(flip, 1 - ys, ys)
    return xs, ys


def make_test_set():
    """All 256 inputs with clean teacher labels."""
    xs = np.arange(256)
    ys = np.array([teacher_label(int(x)) for x in xs])
    return xs, ys
