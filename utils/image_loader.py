"""Ham phu tro doc/luu file anh."""

from __future__ import annotations

import os

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Doc anh tu dia -> RGB uint8 (H,W,3) hoac gray (H,W).

    Raises FileNotFoundError / ValueError neu file khong hop le.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    if not is_supported(path):
        raise ValueError(
            f"Dinh dang khong ho tro: {os.path.splitext(path)[1]}. "
            f"Ho tro: {sorted(SUPPORTED_EXTS)}"
        )
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Khong doc duoc anh (file loi?): {path}")
    if grayscale:
        return img  # (H,W) uint8
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str, image_rgb: np.ndarray) -> None:
    """Luu anh RGB/gray uint8 ra dia (tu dong doi RGB->BGR khi can)."""
    if image_rgb.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image_rgb.ndim == 2:
        ok = cv2.imwrite(path, image_rgb)
    else:
        ok = cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Khong luu duoc anh: {path}")


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def image_info(image: np.ndarray) -> str:
    if image.ndim == 2:
        h, w = image.shape
        return f"{w}×{h} | Ảnh xám"
    h, w, c = image.shape
    return f"{w}×{h} | Ảnh màu ({c} kênh)"
