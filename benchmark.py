"""Benchmark CLI: so sanh thoi gian loc theo kernel size.

Vi du:
    python benchmark.py --image assets/sample.png --kernels 3 5 7 9 --repeat 3
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.filters import benchmark_filters
from utils.image_loader import load_image


def format_benchmark_lines(kernels: list[int], modes: list[str],
                           results: dict[str, list[float]]) -> list[str]:
    """Dong ket qua theo dung thu tu + chi so (kernel lap van dung gia tri).

    Tach rieng de unit-test duoc (bug cu dung kernels.index(k) nen kernel
    lap luon lay gia tri phan tu dau tien).
    """
    lines: list[str] = []
    for i, k in enumerate(kernels):
        for m in modes:
            ms = results[m][i]
            lines.append(f"  kernel={k}x{k} mode={m:12s} -> {ms:8.2f} ms")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Benchmark Median Filter")
    ap.add_argument("--image", default="assets/sample.png")
    ap.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7, 9])
    ap.add_argument("--modes", nargs="+",
                    default=["optimized", "quickselect"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="benchmark_result.png")
    args = ap.parse_args()

    img = load_image(args.image)
    print(f"Image: {args.image} shape={img.shape}")

    results = benchmark_filters(img, args.kernels, args.modes,
                                repeat=args.repeat)
    for line in format_benchmark_lines(args.kernels, args.modes, results):
        print(line)

    plt.figure(figsize=(7, 4.5))
    for m in args.modes:
        plt.plot(args.kernels, results[m], marker="o", label=m)
    plt.xticks(args.kernels)
    plt.xlabel("Kernel size")
    plt.ylabel("Thoi gian (ms, best of %d)" % args.repeat)
    plt.title(f"Median Filter benchmark ({img.shape[1]}x{img.shape[0]})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Da luu bieu do: {args.out}")


if __name__ == "__main__":
    main()
