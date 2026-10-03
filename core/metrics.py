"""Chi so danh gia chat luong anh: PSNR, SSIM + do thoi gian."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def calculate_psnr(
    original: np.ndarray, compared: np.ndarray, max_val: float = 255.0
) -> float:
    """Peak Signal-to-Noise Ratio (dB). Anh giong het nhau -> inf."""
    if original.shape != compared.shape:
        raise ValueError("original and compared must have same shape")
    mse = float(np.mean(
        (original.astype(np.float64) - compared.astype(np.float64)) ** 2
    ))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_val**2) / mse)


def calculate_ssim(original: np.ndarray, compared: np.ndarray,
                   win_size: int | None = None) -> float:
    """Structural Similarity Index (0-1). Ho tro gray + RGB.

    scikit-image mac dinh dung cua so 7x7 nen doi hoi moi chieu >= 7.
    De khong vo GUI voi anh nho: tu chon cua so le lon nhat vua voi anh
    (7/5/3). Anh nho hon 3x3 thi raise ValueError ro rang — phia goi phai
    bat loi nay va van hien thi ket qua loc.
    """
    if original.shape != compared.shape:
        raise ValueError("original and compared must have same shape")
    min_side = min(original.shape[0], original.shape[1])
    if win_size is None:
        if min_side < 3:
            raise ValueError(
                f"SSIM can anh toi thieu 3x3 pixel (anh hien tai "
                f"{original.shape[1]}x{original.shape[0]}). "
                f"Ket qua loc van hop le — chi chi so SSIM khong tinh duoc."
            )
        win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1
                                            else min_side - 1)
    else:
        if win_size % 2 == 0 or win_size < 3:
            raise ValueError("win_size phai la so le >= 3")
        if win_size > min_side:
            raise ValueError(
                f"win_size={win_size} vuot kich thuoc anh ({min_side}px). "
                f"Chon win_size <= {min_side}."
            )
    if original.ndim == 2:
        return float(_ssim(original, compared, data_range=255,
                           win_size=win_size))
    # skimage >= 0.19: channel_axis=-1 thay cho multichannel=True
    try:
        return float(_ssim(original, compared, data_range=255,
                           channel_axis=-1, win_size=win_size))
    except TypeError:  # fallback skimage cu
        return float(_ssim(original, compared, data_range=255,
                           multichannel=True, win_size=win_size))


def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
    """Chay func(*args, **kwargs), tra ve (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0
