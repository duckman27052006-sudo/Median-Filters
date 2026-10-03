"""Widget tái sử dụng: ImagePanel, LabeledSlider."""

from __future__ import annotations

import customtkinter as ctk
import numpy as np
from PIL import Image

FONT = "Segoe UI"


def numpy_to_tk(image: np.ndarray, max_size: tuple[int, int] = (380, 320)):
    """Đổi numpy RGB/gray -> CTkImage giữ tỉ lệ, vừa khung hiển thị."""
    from customtkinter import CTkImage

    if image.ndim == 2:
        pil = Image.fromarray(image, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(image, mode="RGB")
    w, h = pil.size
    mw, mh = max_size
    scale = min(mw / max(w, 1), mh / max(h, 1), 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if (nw, nh) != (w, h):
        pil = pil.resize((nw, nh), Image.BILINEAR)
    return CTkImage(light_image=pil, dark_image=pil, size=(nw, nh))


class ImagePanel(ctk.CTkFrame):
    """Khung hiển thị 1 ảnh + tiêu đề + thông tin."""

    def __init__(self, master, title: str = "Ảnh", **kwargs):
        super().__init__(master, corner_radius=12, **kwargs)
        self.title_label = ctk.CTkLabel(self, text=title,
                                        font=(FONT, 14, "bold"))
        self.title_label.pack(pady=(10, 2))
        self.image_label = ctk.CTkLabel(self, text="Chưa có ảnh",
                                        font=(FONT, 12),
                                        width=380, height=320)
        self.image_label.pack(padx=10, pady=4, expand=True, fill="both")
        self.info_label = ctk.CTkLabel(self, text="—",
                                       font=(FONT, 11), text_color="gray")
        self.info_label.pack(pady=(2, 10))
        self._image: np.ndarray | None = None

    def set_image(self, image: np.ndarray | None, info: str = ""):
        self._image = image
        if image is None:
            self.image_label.configure(image=None, text="Chưa có ảnh")
            self.info_label.configure(text=info or "—")
            return
        tk_img = numpy_to_tk(image)
        # giữ tham chiếu chống GC
        self.image_label._tk_img = tk_img  # type: ignore[attr-defined]
        self.image_label.configure(image=tk_img, text="")
        if not info:
            if image.ndim == 2:
                h, w = image.shape
                info = f"{w}×{h} | Ảnh xám"
            else:
                h, w, _ = image.shape
                info = f"{w}×{h} | Ảnh màu"
        self.info_label.configure(text=info)

    def get_image(self):
        return self._image


class LabeledSlider(ctk.CTkFrame):
    """Slider + nhãn hiển thị giá trị (dùng lại nhiều lần)."""

    def __init__(self, master, label: str, from_: float, to: float,
                 default: float, fmt: str = "{:.0f}", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.fmt = fmt
        self.value = default
        self._label = label
        self.header = ctk.CTkLabel(self, text="", font=(FONT, 12))
        self.header.pack(anchor="w")
        self.slider = ctk.CTkSlider(self, from_=from_, to=to,
                                    command=self._on_change)
        self.slider.pack(fill="x", pady=(2, 4))
        self.slider.set(default)
        self._refresh()

    def _on_change(self, v):
        self.value = float(v)
        self._refresh()

    def _refresh(self):
        self.header.configure(text=f"{self._label}: {self.fmt.format(self.value)}")

    def get(self) -> float:
        return float(self.slider.get())

    def set(self, v: float):
        self.slider.set(v)
        self.value = float(v)
        self._refresh()


class SectionCard(ctk.CTkFrame):
    """Thẻ nhóm điều khiển: tiêu đề + nội dung, gọn mắt hơn."""

    def __init__(self, master, title: str, **kwargs):
        super().__init__(master, corner_radius=12, **kwargs)
        ctk.CTkLabel(self, text=title,
                     font=(FONT, 13, "bold")).pack(anchor="w",
                                                   padx=12, pady=(10, 6))
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="x", padx=12, pady=(0, 12))


class RunGuard:
    """Bộ đếm phiên tác vụ chạy nền (không cần Tk, dễ unit-test).

    Mỗi lần khởi chạy worker gọi `start()` và chụp số phiên; callback chỉ
    được cập nhật UI khi `is_current(seq)` còn đúng. Mọi thao tác làm đổi
    dữ liệu nguồn (tải ảnh, đổi xám/màu) gọi `invalidate()` để callback cũ
    thành stale và bị bỏ qua — tránh ảnh cũ ghi đè ảnh mới.
    """

    def __init__(self) -> None:
        self._seq = 0

    def start(self) -> int:
        self._seq += 1
        return self._seq

    def invalidate(self) -> int:
        self._seq += 1
        return self._seq

    def is_current(self, seq: int) -> bool:
        return seq == self._seq
