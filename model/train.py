"""Quantization-aware training of one 8-input binary neuron.

Run from the repo's top folder:  python model/train.py
Writes model/build/model.json with the quantized weights, the integer bias
and the model's predictions on all 256 inputs (export.py checks those).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import N_INPUTS, TEACHER_W, TEACHER_T, byte_to_pm1, make_test_set, make_train_set

torch.manual_seed(0)

BUILD_DIR = Path(__file__).resolve().parent / "build"


class SignSTE(torch.autograd.Function):
    """Forward: round each weight to +1/-1. Backward: pretend it was identity."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # "Clipped" STE: let the gradient through only where |x| <= 1.
        # Outside that range, nudging the weight can't flip its sign anytime soon.
        return grad_out * (x.abs() <= 1).to(grad_out.dtype)


class RoundSTE(torch.autograd.Function):
    """Forward: round the bias to a whole number. Backward: identity."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class BinaryNeuron(nn.Module):
    def __init__(self, n=N_INPUTS):
        super().__init__()
        # "Latent" float weights: training adjusts these smoothly.
        # Only their signs are ever used in the forward pass.
        self.w = nn.Parameter(torch.randn(n) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def quantized(self):
        return SignSTE.apply(self.w), RoundSTE.apply(self.b)

    def forward(self, x_pm1):
        w_q, b_q = self.quantized()
        # This is exactly 2*popcount - 8 + b, an integer, just like the chip.
        return (x_pm1 * w_q).sum(dim=1) + b_q


def to_tensor(xs):
    return torch.tensor(np.stack([byte_to_pm1(int(x)) for x in xs]), dtype=torch.float32)


def main():
    x_train_bytes, y_train = make_train_set()
    x_test_bytes, y_test = make_test_set()
    x_train, x_test = to_tensor(x_train_bytes), to_tensor(x_test_bytes)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    model = BinaryNeuron()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(300):
        opt.zero_grad()
        z = model(x_train)
        # Scale z down so the sigmoid isn't instantly saturated by values like +/-8.
        loss = loss_fn(0.5 * z, y_train_t)
        loss.backward()
        opt.step()
        with torch.no_grad():
            # Keep latents inside [-1, 1], where the clipped STE passes gradient.
            # Otherwise a weight could drift past 1 and get stuck forever.
            model.w.clamp_(-1, 1)

        if epoch % 50 == 0 or epoch == 299:
            with torch.no_grad():
                pred = (model(x_test) >= 0).long().numpy()
            acc = (pred == y_test).mean()
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  test acc (all 256 inputs) {acc:.3f}")

    with torch.no_grad():
        w_q, b_q = model.quantized()
        preds = (model(x_test) >= 0).long().tolist()

    w_byte = sum(1 << i for i, v in enumerate(w_q.tolist()) if v > 0)
    b_int = int(b_q.item())
    acc = float(np.mean(np.array(preds) == y_test))

    print(f"\nlearned weights  {w_byte:08b}  (teacher {TEACHER_W:08b}; printed MSB first, bit 7 .. bit 0)")
    print(f"learned bias     {b_int}  (teacher threshold {TEACHER_T} corresponds to bias -2 or -1)")
    print(f"final test acc   {acc:.3f}")

    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(BUILD_DIR / "model.json", "w") as f:
        json.dump({"w_byte": w_byte, "b_int": b_int, "torch_preds": preds, "test_acc": acc}, f, indent=2)
    print("wrote model/build/model.json")


if __name__ == "__main__":
    main()
