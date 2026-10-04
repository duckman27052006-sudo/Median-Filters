"""Unit test cho Core Engine: filters, noise, metrics."""

import math

import numpy as np
import pytest

from core.filters import (
    mean_filter,
    median_filter,
    median_filter_naive,
    median_filter_opencv,
    median_filter_quickselect,
)
from core.metrics import calculate_psnr, calculate_ssim
from core.noise import add_gaussian_noise, add_salt_pepper_noise


def _gradient(h=32, w=32):
    x = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    return x


def test_naive_uniform_image_unchanged():
    img = np.full((16, 16), 128, dtype=np.uint8)
    out = median_filter_naive(img, 3)
    np.testing.assert_array_equal(out, img)


def test_naive_vs_opencv_grayscale():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
    # OpenCV medianBlur dung BORDER_REPLICATE o vien -> so sanh voi replicate
    a = median_filter_naive(img, 3, padding="replicate")
    b = median_filter_opencv(img, 3)
    np.testing.assert_array_equal(a, b)
    # Vung giua (bo vien) thi moi padding deu giong nhau
    a2 = median_filter_naive(img, 3, padding="reflect")
    np.testing.assert_array_equal(a2[1:-1, 1:-1], b[1:-1, 1:-1])


def test_quickselect_matches_naive():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(20, 20), dtype=np.uint8)
    a = median_filter_naive(img, 5)
    b = median_filter_quickselect(img, 5)
    np.testing.assert_array_equal(a, b)


def test_color_image_shape_preserved():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    for fn in (median_filter_naive, median_filter_quickselect,
               median_filter_opencv):
        out = fn(img, 3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_invalid_ksize_raises():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        median_filter_naive(img, 4)  # chan -> loi
    with pytest.raises(ValueError):
        median_filter(img, 2, mode="optimized")


def test_salt_pepper_density():
    img = np.full((100, 100), 128, dtype=np.uint8)
    noisy = add_salt_pepper_noise(img, density=0.1, seed=42)
    changed = float(np.mean((noisy == 0) | (noisy == 255)))
    # lay mau duy nhat khong lap -> dung chinh xac 10% tren anh 128 deu
    assert changed == pytest.approx(0.1, abs=1e-9)
    assert int((noisy == 255).sum()) == 500  # muoi
    assert int((noisy == 0).sum()) == 500    # tieu (khong giao nhau)


def test_salt_pepper_full_density_and_reproducible():
    img = np.full((10, 10), 128, dtype=np.uint8)
    a = add_salt_pepper_noise(img, density=1.0, salt_vs_pepper=0.25, seed=0)
    assert set(np.unique(a)).issubset({0, 255})
    assert int((a == 255).sum()) == 25
    assert int((a == 0).sum()) == 75
    b = add_salt_pepper_noise(img, density=1.0, salt_vs_pepper=0.25, seed=0)
    np.testing.assert_array_equal(a, b)  # cung seed -> cung output


def test_salt_pepper_removed_by_median():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    noisy = add_salt_pepper_noise(img, density=0.1, seed=7)
    denoised = median_filter_opencv(noisy, 3)
    # Sau loc phai gan goc hon nhieu
    mse_noisy = np.mean((img.astype(float) - noisy.astype(float)) ** 2)
    mse_clean = np.mean((img.astype(float) - denoised.astype(float)) ** 2)
    assert mse_clean < mse_noisy


def test_gaussian_shape_and_range():
    rng_img = np.full((32, 32, 3), 128, dtype=np.uint8)
    noisy = add_gaussian_noise(rng_img, mean=0, sigma=20, seed=0)
    assert noisy.shape == rng_img.shape
    assert noisy.dtype == np.uint8


def test_psnr_identical_is_inf():
    img = np.zeros((8, 8), dtype=np.uint8)
    assert calculate_psnr(img, img) == float("inf")


def test_psnr_noisy_lower_than_denoised():
    img = np.full((64, 64), 128, dtype=np.uint8)
    noisy = add_salt_pepper_noise(img, density=0.15, seed=3)
    denoised = median_filter_opencv(noisy, 3)
    assert calculate_psnr(img, denoised) > calculate_psnr(img, noisy)


def test_ssim_range():
    img = np.full((32, 32), 128, dtype=np.uint8)
    noisy = add_salt_pepper_noise(img, density=0.1, seed=5)
    s = calculate_ssim(img, noisy)
    assert 0.0 <= s <= 1.0
    assert calculate_ssim(img, img) == pytest.approx(1.0)


def test_mean_shape():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    out = mean_filter(img, 3)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_mean_better_than_median_for_gaussian():
    img = np.full((64, 64), 128, dtype=np.uint8)
    noisy = add_gaussian_noise(img, mean=0, sigma=25, seed=0)
    psnr_median = calculate_psnr(img, median_filter(noisy, 3, mode="optimized"))
    psnr_mean = calculate_psnr(img, median_filter(noisy, 3, mode="mean"))
    assert psnr_mean > psnr_median


# ---- Fix 1: optimized ton trong padding ----

def test_optimized_padding_matches_naive():
    rng = np.random.default_rng(10)
    for shape in [(16, 16), (12, 10, 3)]:
        img = rng.integers(0, 256, size=shape, dtype=np.uint8)
        for pad in ["reflect", "replicate", "zero"]:
            for k in [3, 5]:
                a = median_filter(img, k, mode="optimized", padding=pad)
                b = median_filter_naive(img, k, padding=pad)
                np.testing.assert_array_equal(a, b)


def test_optimized_default_backward_compatible():
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    np.testing.assert_array_equal(
        median_filter_opencv(img, 3),
        median_filter_opencv(img, 3, padding="replicate"),
    )


def test_mean_padding_border_differs_interior_equal():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 255
    z = mean_filter(img, 3, padding="zero")
    r = mean_filter(img, 3, padding="replicate")
    assert not np.array_equal(z, r)  # vien khac nhau theo padding
    np.testing.assert_array_equal(z[2:-2, 2:-2], r[2:-2, 2:-2])


def test_reflect_matches_numpy_convention():
    """reflect phai la REFLECT_101 (khop np.pad), khong phai BORDER_REFLECT."""
    import cv2

    from core.filters import _BORDER_MAP

    assert _BORDER_MAP["reflect"] == cv2.BORDER_REFLECT_101
    rng = np.random.default_rng(20)
    img = rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
    got = mean_filter(img, 3, padding="reflect")
    padded = np.pad(img, 1, mode="reflect")  # REFLECT_101
    ref = cv2.blur(padded, (3, 3))[1:-1, 1:-1]
    np.testing.assert_array_equal(got, ref)


def test_naive_quickselect_distinct_paths_same_result():
    """Hai mode di qua implementation rieng (sort vs partition)."""
    from core.filters import _median_fast

    rng = np.random.default_rng(21)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    a = _median_fast(img, 5, "reflect", method="sort")
    b = _median_fast(img, 5, "reflect", method="partition")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(median_filter_naive(img, 5), a)
    np.testing.assert_array_equal(median_filter_quickselect(img, 5), b)
    with pytest.raises(ValueError):
        _median_fast(img, 3, "reflect", method="bogus")


# ---- Fix 2: SSIM cho anh nho ----

@pytest.mark.parametrize("h,w", [(3, 3), (5, 5), (6, 6), (7, 7), (8, 10)])
def test_ssim_small_images_ok(h, w):
    for shape in [(h, w), (h, w, 3)]:
        img = np.full(shape, 128, dtype=np.uint8)
        assert calculate_ssim(img, img) == pytest.approx(1.0)
        noisy = add_salt_pepper_noise(img, density=0.1, seed=1)
        assert 0.0 <= calculate_ssim(img, noisy) <= 1.0


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (1, 1, 3), (2, 5)])
def test_ssim_too_small_raises_clear_error(shape):
    img = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="3x3"):
        calculate_ssim(img, img)


def test_ssim_win_size_validation():
    img = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_ssim(img, img, win_size=4)  # chan
    with pytest.raises(ValueError):
        calculate_ssim(img, img, win_size=9)  # vuot kich thuoc


# ---- Fix 4: benchmark helper ----

def test_benchmark_filters_result_shape():
    from core.filters import benchmark_filters

    rng = np.random.default_rng(12)
    img = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    res = benchmark_filters(img, [3, 5], ["optimized", "mean"], repeat=2)
    assert set(res) == {"optimized", "mean"}
    for v in res.values():
        assert len(v) == 2
        assert all(float(x) >= 0 for x in v)


def test_estimate_filter_ms_flags_slow_combo():
    from core.filters import estimate_filter_ms

    fast = estimate_filter_ms(4000, 3000, 3, 5, "optimized")
    slow = estimate_filter_ms(4000, 3000, 3, 5, "naive")
    assert fast < 3000
    assert slow > 3000  # 12MP + k5 naive: thuc te ~13s
    assert estimate_filter_ms(256, 256, 3, 3, "naive") < 3000


def test_benchmark_lines_duplicate_kernels():
    """Kernel lap phai lay dung gia tri theo chi so, khong phai index()."""
    from benchmark import format_benchmark_lines

    lines = format_benchmark_lines([3, 5, 3], ["optimized"],
                                   {"optimized": [10.0, 20.0, 30.0]})
    assert len(lines) == 3
    assert "10.00" in lines[0] and "kernel=3x3" in lines[0]
    assert "20.00" in lines[1] and "kernel=5x5" in lines[1]
    assert "30.00" in lines[2] and "kernel=3x3" in lines[2]  # bug cu ra 10.00


def test_benchmark_rejects_bad_config():
    from core.filters import benchmark_filters

    img = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown benchmark mode"):
        benchmark_filters(img, [3], ["misspelled"])
    with pytest.raises(ValueError, match="rong"):
        benchmark_filters(img, [3], [])
    with pytest.raises(ValueError, match="rong"):
        benchmark_filters(img, [], ["optimized"])
    with pytest.raises(ValueError, match="so le"):
        benchmark_filters(img, [4], ["optimized"])
    with pytest.raises(ValueError, match="repeat"):
        benchmark_filters(img, [3], ["optimized"], repeat=0)
    with pytest.raises(ValueError, match="padding"):
        benchmark_filters(img, [3], ["optimized"], padding="bogus")


def test_benchmark_keeps_runtime_error_and_others(monkeypatch):
    """Loi runtime 1 o do van bao ro + khong mat ket qua o khac."""
    import core.filters as F

    real = F.median_filter

    def flaky(image, ksize=3, mode="optimized", padding="reflect"):
        if mode == "mean":
            raise RuntimeError("boom-test")
        return real(image, ksize, mode=mode, padding=padding)

    monkeypatch.setattr(F, "median_filter", flaky)
    img = np.zeros((8, 8), dtype=np.uint8)
    with pytest.warns(UserWarning, match="boom-test"):
        res = F.benchmark_filters(img, [3], ["optimized", "mean"])
    assert res["optimized"] and all(float(x) >= 0 for x in res["optimized"])
    assert len(res["mean"]) == 1 and math.isnan(res["mean"][0])


def test_empty_images_rejected_at_validation():
    from core.filters import mean_filter as _mf

    for shape in [(0, 8), (8, 0), (0, 0), (0, 8, 3), (8, 0, 3)]:
        img = np.zeros(shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="duong"):
            median_filter(img, 3, mode="optimized")
        with pytest.raises(ValueError, match="duong"):
            _mf(img, 3)
    ok = np.zeros((8, 8), dtype=np.uint8)
    assert median_filter(ok, 3, mode="optimized").shape == (8, 8)


def test_psnr_rejects_bad_max_val():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.full((8, 8), 10, dtype=np.uint8)
    for bad in (0, -1, 0.0, float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError, match="max_val"):
            calculate_psnr(a, b, max_val=bad)
    assert calculate_psnr(a, b, max_val=255.0) > 0
    assert calculate_psnr(a, a) == float("inf")  # hanh vi cu giu nguyen
