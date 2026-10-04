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
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(
            "image phai co chieu cao va chieu rong duong "
            f"(nhan duoc {image.shape}).")
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


def _channel_median_fast(padded_ch: np.ndarray, ksize: int,
                         method: str) -> np.ndarray:
    """Median 1 kenh dung sliding_window_view, chia strip de nhe RAM.

    Nhanh hon vong lap Python ~100x, ket qua BYTE-IDENTICAL giua 2 method
    (cua so le). Strip ~16MB nen anh 12MP van chay.
    method="sort": full-sort kieu naive (phuc vu hoc tap, cham hon).
    method="partition": quickselect, chi dua median ve dung vi tri.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    hp, wp = padded_ch.shape
    h, w = hp - ksize + 1, wp - ksize + 1
    out = np.empty((h, w), dtype=np.uint8)
    mid = (ksize * ksize) // 2
    bytes_per_row = w * ksize * ksize
    rows = max(1, min(h, (16 * 1024 * 1024) // max(1, bytes_per_row)))
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        strip = padded_ch[r0:r1 + ksize - 1, :]
        win = sliding_window_view(strip, (ksize, ksize))  # (rows,W,k,k)
        flat = win.reshape(r1 - r0, w, -1)
        if method == "sort":
            out[r0:r1] = np.sort(flat, axis=2)[:, :, mid].astype(np.uint8)
        elif method == "partition":
            out[r0:r1] = np.partition(flat, mid, axis=2)[:, :, mid].astype(
                np.uint8)
        else:
            raise ValueError(f"Unknown median method '{method}'")
    return out


def _median_fast(image: np.ndarray, ksize: int, padding: str,
                 method: str) -> np.ndarray:
    _validate(image, ksize)
    padded = _pad(image, ksize, padding)
    if image.ndim == 2:
        return _channel_median_fast(padded, ksize, method)
    return np.stack([_channel_median_fast(padded[:, :, c], ksize, method)
                     for c in range(image.shape[2])], axis=-1)


def median_filter_naive(
    image: np.ndarray, ksize: int = 3, padding: str = "reflect"
) -> np.ndarray:
    """Loc trung vi kieu 'thu cong' (phuc vu hoc tap).

    Full-sort vector hoa: van dung y tuong naive (sap xep toan bo cua so roi
    lay giua) nhung chay tren NumPy stride tricks + chia strip nen anh lon
    khong treo giao dien. Ket qua BYTE-IDENTICAL voi quickselect (cua so le).
    """
    return _median_fast(image, ksize, padding, method="sort")


def median_filter_quickselect(
    image: np.ndarray, ksize: int = 3, padding: str = "reflect"
) -> np.ndarray:
    """Dung partition (Quickselect) de lay median ma khong sort full.

    Nhanh hon naive, ket qua BYTE-IDENTICAL voi naive (cua so le);
    ban vector hoa, anh lon van chay.
    """
    return _median_fast(image, ksize, padding, method="partition")


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
    # "reflect" phai la REFLECT_101 de khop np.pad(mode="reflect") ma median
    # dung (khong lap pixel bien) — truoc day map nham sang BORDER_REFLECT.
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "zero": cv2.BORDER_CONSTANT,
}


def mean_filter(image: np.ndarray, ksize: int = 3,
                padding: str = "reflect") -> np.ndarray:
    """Loc trung binh (box blur) — tot cho nhieu Gaussian.

    Median manh voi nhieu xung (muoi tieu) nhung yeu voi nhieu Gaussian;
    mean blur bu dap thieu sot nay.
    """
    _validate(image, ksize)
    if padding not in _BORDER_MAP:
        raise ValueError(f"padding must be one of {list(_BORDER_MAP)}")
    return cv2.blur(image, (ksize, ksize),
                    borderType=_BORDER_MAP[padding])


def median_filter(
    image: np.ndarray,
    ksize: int = 3,
    mode: str = "optimized",
    padding: str = "reflect",
) -> np.ndarray:
    """Dispatcher chinh.

    Median (naive/quickselect/numba/optimized): manh voi nhieu MUOI TIEU.
    Mean: danh cho nhieu GAUSSIAN (median yeu voi loai nay).
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
    raise ValueError(
        f"Unknown mode '{mode}'. "
        "Choose naive/quickselect/numba/optimized (median) "
        "or mean (cho nhieu Gaussian).")


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

    Lay best-of-repeat cho moi o. Cau hinh sai (mode la, kernel chan,
    repeat < 1, padding la) duoc bao ValueError NGAY, khong giau thanh NaN.
    Loi runtime cua tung o do van tiep tuc benchmark (khong mat ket qua cac
    o khac) nhung duoc giu lai qua warnings.warn thay vi NaN cam.
    """
    import warnings

    valid_modes = {"naive", "quickselect", "numba", "optimized", "opencv",
                   "mean"}
    if not modes:
        raise ValueError("modes khong duoc rong")
    for m in modes:
        if not isinstance(m, str) or m.lower() not in valid_modes:
            raise ValueError(
                f"Unknown benchmark mode '{m}'. Chon trong {sorted(valid_modes)}.")
    ksizes = list(ksizes)
    if not ksizes:
        raise ValueError("ksizes khong duoc rong")
    for k in ksizes:
        if (not isinstance(k, (int, np.integer)) or isinstance(k, bool)
                or int(k) < 3 or int(k) % 2 == 0):
            raise ValueError(
                f"ksize phai la so le >= 3 (nhan duoc {k!r}).")
    repeat_ok = (isinstance(repeat, (int, np.integer))
                 and not isinstance(repeat, bool) and int(repeat) >= 1)
    if not repeat_ok:
        raise ValueError(f"repeat phai la so nguyen >= 1 (nhan duoc {repeat!r}).")
    if padding not in _PADDING_MAP:
        raise ValueError(f"padding phai la mot trong {list(_PADDING_MAP)}.")
    results: dict[str, list[float]] = {m: [] for m in modes}
    for k in ksizes:
        for m in modes:
            best = float("inf")
            ok = False
            for _ in range(max(1, repeat)):
                t0 = time.perf_counter()
                try:
                    median_filter(image, k, mode=m, padding=padding)
                except Exception as exc:
                    best = float("nan")
                    ok = True
                    warnings.warn(
                        f"benchmark {m} k={k} that bai: "
                        f"{type(exc).__name__}: {exc}")
                    break
                dt = (time.perf_counter() - t0) * 1000.0
                best = min(best, dt)
                ok = True
            results[m].append(best if ok else float("nan"))
    return results


_SLOW_MODES = {"naive", "quickselect", "numba"}


def estimate_filter_ms(h: int, w: int, channels: int, ksize: int,
                       mode: str) -> float:
    """Uoc tinh thoi gian loc (ms) de GUI canh bao truoc khi chay.

    Hieu chuan tu do thuc te: mode NumPy ton ~5ns moi pixel-cua-so
    (12MP RGB k5 naive ~13s); mode OpenCV (optimized/mean) vai ms.
    """
    if mode.lower() not in _SLOW_MODES:
        return 5.0
    return h * w * channels * ksize * ksize * 5e-6
