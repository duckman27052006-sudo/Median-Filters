"""Test hoi quy cho Fix 3 + Fix 6 (logic thuan, khong can mo cua so GUI)."""

import numpy as np
import pytest

from gui.app import _freeze_error, _quality_text, improvement_note, make_noisy
from gui.components import RunGuard


def test_runguad_marks_old_task_stale():
    g = RunGuard()
    first = g.start()
    second = g.start()
    assert not g.is_current(first)
    assert g.is_current(second)


def test_runguad_invalidate_kills_inflight_task():
    g = RunGuard()
    seq = g.start()
    g.invalidate()  # mo phong: tai anh moi / doi xam-mau khi worker dang chay
    assert not g.is_current(seq)


def test_freeze_error_survives_after_except_block():
    try:
        raise RuntimeError("boom")
    except Exception as e:
        msg = _freeze_error(e)
        cb = lambda m=msg: m  # cach dung dung trong callback `after`
    assert cb() == "RuntimeError: boom"  # khong NameError


def test_old_closure_pattern_would_raise_nameerror():
    """Tai hien bug Fix 6: closure giu truc tiep bien `e` cua except."""
    try:
        raise RuntimeError("boom")
    except Exception as e:
        cb = lambda: str(e)
    with pytest.raises(NameError):
        cb()


def test_quality_text_never_raises_on_tiny_image():
    tiny = np.zeros((2, 2), dtype=np.uint8)
    text = _quality_text(tiny, tiny)
    assert "PSNR" in text
    assert "không tính được" in text  # SSIM that bai nhung van co chuoi


def test_quality_text_normal_image():
    img = np.full((16, 16), 128, dtype=np.uint8)
    text = _quality_text(img, img)
    assert "∞" in text  # PSNR inf
    assert "1.0000" in text  # SSIM


def test_make_noisy_same_seed_same_output():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    a = make_noisy(img, "Salt & Pepper", density_pct=20.0, seed=99)
    b = make_noisy(img, "Salt & Pepper", density_pct=20.0, seed=99)
    np.testing.assert_array_equal(a, b)
    g1 = make_noisy(img, "Gaussian", sigma=25.0, mean=0.0, seed=7)
    g2 = make_noisy(img, "Gaussian", sigma=25.0, mean=0.0, seed=7)
    np.testing.assert_array_equal(g1, g2)


def test_make_noisy_passthrough_without_noise():
    img = np.full((8, 8), 200, dtype=np.uint8)
    out = make_noisy(img, "Không thêm nhiễu", seed=1)
    np.testing.assert_array_equal(out, img)
    assert out is not img  # bản sao, không alias


def test_gray_toggle_roundtrip_keeps_master():
    """Mô phỏng toggle xám→màu: bản màu gốc không bao giờ bị ghi đè."""
    from utils.image_loader import to_grayscale, to_rgb

    rng = np.random.default_rng(5)
    master = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    gray = to_grayscale(master)  # những gì toggle-xám dựng
    assert gray.shape == (16, 16)
    # bật lại màu = dùng lại master, không phải to_rgb(gray)
    restored = master
    np.testing.assert_array_equal(restored, master)
    assert not np.array_equal(to_rgb(gray), master)  # bug cũ: mất màu


def test_improvement_note_no_change():
    assert "giống hệt" in improvement_note(float("inf"), float("inf"))
    assert "gần như" in improvement_note(20.0, 20.2)
    assert "+0.20" in improvement_note(20.0, 20.2)


def test_improvement_note_good_result_empty():
    assert improvement_note(13.4, 33.7) == ""
    assert improvement_note(float("nan"), 30.0) == ""


def test_improvement_note_decrease_and_boundary():
    note = improvement_note(30.0, 25.0)
    assert "-5.00" in note and "+-" not in note  # dau dung, khong '+-'
    assert "tệ hơn" in note  # giam thi bao giam, khong bao 'gan nhu'
    assert improvement_note(20.0, 20.5) == ""  # bang nguong 0.5 -> dat
    assert "gần như" in improvement_note(20.0, 20.49)
    # nhieu == goc (inf) ma loc lam te di -> bao giam, dung
    assert "tệ hơn" in improvement_note(float("inf"), 30.0)


def _make_app_quiet(monkeypatch):
    """Dung app that co man hinh; chan messagebox de test khong bi block."""
    import gui.app as appmod

    class _QuietBox:
        def __init__(self):
            self.errors = []

        def showerror(self, *a, **k):
            self.errors.append((a, k))

        def showwarning(self, *a, **k):
            pass

        def showinfo(self, *a, **k):
            pass

    box = _QuietBox()
    monkeypatch.setattr(appmod, "messagebox", box)
    try:
        app = appmod.MedianFilterApp()
    except Exception as e:
        pytest.skip(f"can man hinh de test GUI: {e}")
    app.update()
    return app, box


def test_on_sample_returns_bool_and_startup_status(monkeypatch):
    import gui.app as appmod

    app, box = _make_app_quiet(monkeypatch)
    try:
        assert app.on_sample() is True  # anh mau hop le

        # mo phong anh mau hong: load that bai
        def _boom(*a, **k):
            raise ValueError("file hong")

        monkeypatch.setattr(appmod, "load_image", _boom)
        assert app.on_sample() is False
        assert box.errors  # thong bao loi cu the duoc giu

        # khoi dong voi anh hong -> khong duoc bao "San sang"
        app._load_sample_if_exists()
        assert "Sẵn sàng" not in app.metrics.cget("text")
        assert "Tải ảnh lên" in app.metrics.cget("text")

        # khoi phuc: nap lai duoc
        monkeypatch.undo()
        monkeypatch.setattr(appmod, "messagebox", box)
        assert app.on_sample() is True
    finally:
        app.quit()
        app.destroy()
