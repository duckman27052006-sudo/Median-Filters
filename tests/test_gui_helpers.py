"""Test hoi quy cho Fix 3 + Fix 6 (logic thuan, khong can mo cua so GUI)."""

import numpy as np
import pytest

from gui.app import _freeze_error, _quality_text
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
