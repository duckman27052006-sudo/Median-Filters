"""Bo gia lap nhieu: Salt & Pepper + Gaussian."""

from __future__ import annotations

import numpy as np


def add_salt_pepper_noise(
    image: np.ndarray,
    density: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Them nhieu muoi tieu.

    Args:
        image: uint8 (H,W) hoac (H,W,3).
        density: TI LE pixel muc tieu bi nhieu, 0.01 - 1.0. So pixel nhieu
            = round(density * H * W), chon DUY NHAT khong lap (khong dung
            lay mau co hoan lai nhu ban cu).
        salt_vs_pepper: ti le muoi (trang) trong tong nhieu, mac dinh 0.5.
            Tap pixel nhieu duoc chia doi thanh 2 nhom roi rac (khong giao).
        seed: de lap lai ket qua neu can (cung seed -> cung output).

    Returns:
        Anh uint8 da them nhieu (salt=255, pepper=0).
    """
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if not 0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if not 0 <= salt_vs_pepper <= 1.0:
        raise ValueError("salt_vs_pepper must be in [0, 1]")

    rng = np.random.default_rng(seed)
    noisy = image.copy()
    h, w = image.shape[:2]
    n_total = int(round(density * h * w))
    n_total = max(1, min(h * w, n_total))
    n_salt = int(round(n_total * salt_vs_pepper))
    n_salt = max(0, min(n_total, n_salt))
    n_pepper = n_total - n_salt

    flat = rng.choice(h * w, size=n_total, replace=False)
    salt_flat = flat[:n_salt]
    pep_flat = flat[n_salt:]
    salt_rows, salt_cols = np.unravel_index(salt_flat, (h, w))
    pep_rows, pep_cols = np.unravel_index(pep_flat, (h, w))

    if noisy.ndim == 2:
        noisy[salt_rows, salt_cols] = 255
        noisy[pep_rows, pep_cols] = 0
    else:
        noisy[salt_rows, salt_cols, :] = 255
        noisy[pep_rows, pep_cols, :] = 0
    return noisy


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: int | None = None,
) -> np.ndarray:
    """Them nhieu Gaussian N(mean, sigma^2), clip ve [0,255] uint8."""
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    rng = np.random.default_rng(seed)
    # float32 ngay tu RNG (nhe 1/2 RAM va nhanh hon vs float64 mac dinh)
    noise = rng.standard_normal(image.shape, dtype=np.float32)
    noise *= np.float32(sigma)
    if mean != 0.0:
        noise += np.float32(mean)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
