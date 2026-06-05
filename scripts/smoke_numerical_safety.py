#!/usr/bin/env python3
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from quantizers import SequenceQuantizerSoftEMA  # noqa: E402


def check_safe_quantizer_zero_input():
    quantizer = SequenceQuantizerSoftEMA(4, 8, safe_normalize=True)
    inputs = torch.zeros(2, 8)
    reconstruction, loss = quantizer(inputs)
    if not torch.isfinite(reconstruction).all():
        raise SystemExit("safe quantizer reconstruction is non-finite")
    if not torch.isfinite(loss):
        raise SystemExit("safe quantizer loss is non-finite")


def check_unsafe_quantizer_documents_risk():
    quantizer = SequenceQuantizerSoftEMA(4, 8, safe_normalize=False)
    inputs = torch.zeros(2, 8)
    reconstruction, loss = quantizer(inputs)
    if torch.isfinite(reconstruction).all() and torch.isfinite(loss):
        raise SystemExit("unsafe quantizer unexpectedly stayed finite on zero input")


def main():
    check_safe_quantizer_zero_input()
    check_unsafe_quantizer_documents_risk()
    print("numerical safety smoke tests passed")


if __name__ == "__main__":
    main()
