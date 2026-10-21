"""Bit-exact integer reference model. This is the answer key.

It uses only integers and bit operations, exactly as spec.md describes.
The PyTorch model must agree with it, and the Verilog must agree with it.
"""

MASK8 = 0xFF


def popcount(v):
    return bin(v & MASK8).count("1")


def neuron(x_byte, w_byte, threshold):
    """Returns (y, pc) for one input byte, per spec.md section 3."""
    match = ~(x_byte ^ w_byte) & MASK8   # XNOR, kept to 8 bits
    pc = popcount(match)
    y = 1 if pc >= threshold else 0
    return y, pc


def bias_to_threshold(b_int, n=8):
    """T = ceil((n - b) / 2), clipped to 0..n+1 (spec.md section 3)."""
    num = n - b_int
    t = -((-num) // 2)  # integer ceil division, safe for negatives
    return max(0, min(n + 1, t))


if __name__ == "__main__":
    # Hand-checkable examples. Run: python reference.py
    # x = all +1, w = all +1 -> every bit matches -> pc = 8
    assert neuron(0xFF, 0xFF, 5) == (1, 8)
    # x = all +1, w = all -1 -> nothing matches -> pc = 0
    assert neuron(0xFF, 0x00, 5) == (0, 0)
    # 4 of 8 match -> pc = 4, below T=5
    assert neuron(0b0000_1111, 0xFF, 5) == (0, 4)
    # T=0 always fires, T=9 never fires
    assert neuron(0x00, 0xFF, 0)[0] == 1 and neuron(0xFF, 0xFF, 9)[0] == 0
    # Threshold folding: bias -2 means 2*pc - 8 - 2 >= 0, i.e. pc >= 5
    assert bias_to_threshold(-2) == 5
    assert bias_to_threshold(-1) == 5    # 2*pc >= 9 -> pc >= 4.5 -> 5
    assert bias_to_threshold(0) == 4
    assert bias_to_threshold(20) == 0 and bias_to_threshold(-20) == 9
    # Cross-check folding against the "real" neuron for every case
    for b in range(-12, 13):
        t = bias_to_threshold(b)
        for pc in range(9):
            assert (2 * pc - 8 + b >= 0) == (pc >= t), (b, pc)
    print("reference.py: all self-checks passed")
