"""Median Filter implementations: naive, quickselect, OpenCV-optimized.

All functions accept uint8 images:
  - grayscale: (H, W)
  - color RGB: (H, W, 3)
and return uint8 images of the same shape.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

try:
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # Numba optional (may not support Python 3.14 yet)
    _HAS_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f

        if a and callable(a[0]):
            return a[0]
        return deco


_PADDING_MAP = {
    "zero": "constant",
    "reflect": "reflect",
    "replicate": "edge",
}


def _validate(image: np.ndarray, ksize: int) -> int:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8 (0-255)")
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale (H,W) or RGB (H,W,3)")
    if not isinstance(ksize, int) or ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer >= 3 (3,5,7,9,...)")
    return ksize


def _pad(image: np.ndarray, ksize: int, padding: str) -> np.ndarray:
    if padding not in _PADDING_MAP:
        raise ValueError(f"padding must be one of {list(_PADDING_MAP)}")
    mode = _PADDING_MAP[padding]
    p = ksize // 2
    if image.ndim == 2:
        return np.pad(image, p, mode=mode)
    return np.pad(image, ((p, p), (p, p), (0, 0)), mode=mode)


def median_filter_naive(
    image: np.ndarray, ksize: int = 3, padding: str = "reflect"
) -> np.ndarray:
    """Cai dat thu cong bang NumPy (phuc vu hoc tap).

    Duyet tung pixel, trich cua so ksize x ksize, lay np.median (full sort).
    Do phuc tap O(H*W*K^2 log K^2) nen cham voi anh lon / kernel lon.
    """
    _validate(image, ksize)
    padded = _pad(image, ksize, padding)
    p = ksize // 2

    if image.ndim == 2:
        h, w = image.shape
        out = np.empty_like(image)
        for i in range(h):
            for j in range(w):
                window = padded[i : i + ksize, j : j + ksize]
                out[i, j] = np.median(window)
        return out

    h, w, c = image.shape
    out = np.empty_like(image)
    for i in range(h):
        for j in range(w):
            for ch in range(c):
                window = padded[i : i + ksize, j : j + ksize, ch]
                out[i, j, ch] = np.median(window)
    return out


def median_filter_quickselect(
    image: np.ndarray, ksize: int = 3, padding: str = "reflect"
) -> np.ndarray:
    """Dung np.partition (Quickselect) de lay median ma khong sort full.

    Nhanh hon naive ~2-3x, ket qua tuong duong.
    """
    _validate(image, ksize)
    padded = _pad(image, ksize, padding)
    p = ksize // 2
    mid = (ksize * ksize) // 2

    def _median_of_window(win: np.ndarray) -> int:
        flat = win.reshape(-1)
        return int(np.partition(flat, mid)[mid])

    if image.ndim == 2:
        h, w = image.shape
        out = np.empty_like(image)
        for i in range(h):
            for j in range(w):
                out[i, j] = _median_of_window(padded[i : i + ksize, j : j + ksize])
        return out

    h, w, c = image.shape
    out = np.empty_like(image)
    for i in range(h):
        for j in range(w):
            for ch in range(c):
                out[i, j, ch] = _median_of_window(
                    padded[i : i + ksize, j : j + ksize, ch]
                )
    return out


if _HAS_NUMBA:

    @njit
    def _numba_median_gray(
        padded: np.ndarray, out: np.ndarray, ksize: int, h: int, w: int
    ) -> None:
        p = ksize // 2
        n = ksize * ksize
        mid = n // 2
        buf = np.empty(n, dtype=np.int64)
        for i in range(h):
            for j in range(w):
                k = 0
                for di in range(ksize):
                    for dj in range(ksize):
                        buf[k] = padded[i + di, j + dj]
                        k += 1
                # insertion sort on small buffer (K^2 <= 81) is fast
                for a in range(1, n):
                    v = buf[a]
                    b = a - 1
                    while b >= 0 and buf[b] > v:
                        buf[b + 1] = buf[b]
                        b -= 1
                    buf[b + 1] = v
                out[i, j] = np.uint8(buf[mid])

    _NUMBA_AVAILABLE = True
else:
    _NUMBA_AVAILABLE = False


def median_filter_numba(
    image: np.ndarray, ksize: int = 3, padding: str = "reflect"
) -> np.ndarray:
    """Tang toc bang Numba (neu co). Fallback ve quickselect neu thieu numba."""
    _validate(image, ksize)
    if not _NUMBA_AVAILABLE:
        return median_filter_quickselect(image, ksize, padding)
    padded = _pad(image, ksize, padding)
    if image.ndim == 2:
        h, w = image.shape
        out = np.empty_like(image)
        _numba_median_gray(padded, out, ksize, h, w)
        return out
    # color: process each channel with numba kernel
    h, w, c = image.shape
    out = np.empty_like(image)
    for ch in range(c):
        ch_out = np.empty((h, w), dtype=np.uint8)
        _numba_median_gray(padded[:, :, ch], ch_out, ksize, h, w)
        out[:, :, ch] = ch_out
    return out


def median_filter_opencv(image: np.ndarray, ksize: int = 3,
                         padding: str = "replicate") -> np.ndarray:
    """Dung cv2.medianBlur (C++ toi uu). Nhanh nhat, dung cho anh lon/FullHD.

    `cv2.medianBlur` mac dinh dung vien REPLICATE. De ton trong lua chon
    `padding` cua nguoi dung (reflect/replicate/zero), ham tu pad thu cong
    roi blur tren anh da pad va cat lay vung giua — ket qua vien khop voi
    `median_filter_naive` cung padding. Duong `replicate` giu nguyen goi
    truc tiep (nhanh, giu hanh vi cu).
    """
    _validate(image, ksize)
    if padding not in _PADDING_MAP:
        raise ValueError(f"padding must be one of {list(_PADDING_MAP)}")
    if padding == "replicate":
        # cv2 requires ksize odd > 1; medianBlur works on both gray and BGR.
        # Our convention is RGB, but median is channel-wise so BGR==RGB result
        # up to channel order (identical operation per channel).
        return cv2.medianBlur(image, ksize)
    p = ksize // 2
    padded = _pad(image, ksize, padding)
    blurred = cv2.medianBlur(padded, ksize)
    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w = image.shape[:2]
    if image.ndim == 2:
        return blurred[p:p + h, p:p + w].copy()
    return blurred[p:p + h, p:p + w, :].copy()


_BORDER_MAP = {
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "zero": cv2.BORDER_CONSTANT,
}


def mean_filter(image: np.ndarray, ksize: int = 3,
                padding: str = "reflect") -> np.ndarray:
    """Loc trung binh (box blur) — tot cho nhieu Gaussian.

    Median manh voi nhieu xung (muoi tieu) nhung yeu voi nhieu Gaussian;
    mean/gaussian blur bu dap thieu sot nay.
    """
    _validate(image, ksize)
    if padding not in _BORDER_MAP:
        raise ValueError(f"padding must be one of {list(_BORDER_MAP)}")
    return cv2.blur(image, (ksize, ksize),
                    borderType=_BORDER_MAP[padding])


def gaussian_filter(image: np.ndarray, ksize: int = 3,
                    padding: str = "reflect") -> np.ndarray:
    """Loc Gaussian blur — tot cho nhieu Gaussian, giu cau truc mem hon mean."""
    _validate(image, ksize)
    if padding not in _BORDER_MAP:
        raise ValueError(f"padding must be one of {list(_BORDER_MAP)}")
    # sigma tu dong theo OpenCV (0 = tu tinh tu ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), 0,
                            borderType=_BORDER_MAP[padding])


def median_filter(
    image: np.ndarray,
    ksize: int = 3,
    mode: str = "optimized",
    padding: str = "reflect",
) -> np.ndarray:
    """Dispatcher chinh.

    Median (naive/quickselect/numba/optimized): manh voi nhieu MUOI TIEU.
    Mean/gaussian: danh cho nhieu GAUSSIAN (median yeu voi loai nay).
    """
    mode = mode.lower()
    if mode in ("optimized", "opencv"):
        return median_filter_opencv(image, ksize, padding)
    if mode == "naive":
        return median_filter_naive(image, ksize, padding)
    if mode == "quickselect":
        return median_filter_quickselect(image, ksize, padding)
    if mode == "numba":
        return median_filter_numba(image, ksize, padding)
    if mode == "mean":
        return mean_filter(image, ksize, padding)
    if mode in ("gaussian", "gauss"):
        return gaussian_filter(image, ksize, padding)
    raise ValueError(
        f"Unknown mode '{mode}'. "
        "Choose naive/quickselect/numba/optimized (median) "
        "or mean/gaussian (cho nhieu Gaussian).")


def timed_filter(
    image: np.ndarray, ksize: int = 3, mode: str = "optimized",
    padding: str = "reflect",
) -> tuple[np.ndarray, float]:
    """Chay filter va do thoi gian (ms). Tien cho GUI/benchmark."""
    t0 = time.perf_counter()
    out = median_filter(image, ksize, mode, padding)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out, dt_ms


def benchmark_filters(
    image: np.ndarray,
    ksizes: tuple[int, ...] | list[int] = (3, 5, 7, 9),
    modes: tuple[str, ...] | list[str] = ("optimized", "quickselect"),
    repeat: int = 1,
    padding: str = "reflect",
) -> dict[str, list[float]]:
    """Do thoi gian loc (ms) theo kernel/mode. Tra ve {mode: [ms per ksize]}.

    Lay best-of-repeat cho moi o. Mode loi cho NaN thay vi raise, de GUI/CLI
    van ve duoc bieu do va khong ket chuong trinh.
    """
    results: dict[str, list[float]] = {m: [] for m in modes}
    for k in ksizes:
        for m in modes:
            best = float("inf")
            ok = False
            for _ in range(max(1, repeat)):
                t0 = time.perf_counter()
                try:
                    median_filter(image, k, mode=m, padding=padding)
                except Exception:
                    best = float("nan")
                    ok = True
                    break
                dt = (time.perf_counter() - t0) * 1000.0
                best = min(best, dt)
                ok = True
            results[m].append(best if ok else float("nan"))
    return results
