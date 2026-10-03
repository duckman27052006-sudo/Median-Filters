"""Màn hình chính Median Filter App (CustomTkinter)."""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time

import customtkinter as ctk
import numpy as np
from tkinter import filedialog, messagebox

# Cho phép chạy cả từ root (main.py) và trực tiếp trong gui/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filters import benchmark_filters, estimate_filter_ms, median_filter
from core.metrics import calculate_psnr, calculate_ssim
from core.noise import add_gaussian_noise, add_salt_pepper_noise
from gui.components import ImagePanel, LabeledSlider, RunGuard, SectionCard
from utils.image_loader import SUPPORTED_EXTS, image_info, load_image, save_image

FONT = "Segoe UI"

APP_VERSION = "1.5.2"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def _freeze_error(exc: BaseException) -> str:
    """Chốt thông điệp ngoại lệ thành chuỗi bền vững.

    Tránh closure tham chiếu trực tiếp biến `e` của khối `except` (Python
    xóa biến đó khi thoát khối, callback chạy sau sẽ gặp NameError).
    """
    return f"{type(exc).__name__}: {exc}"


def make_noisy(base: np.ndarray, kind: str, density_pct: float = 10.0,
               sigma: float = 25.0, mean: float = 0.0,
               seed: int | None = None) -> np.ndarray:
    """Tạo ảnh nhiễu từ ảnh gốc (hàm thuần, dễ unit-test).

    Cùng `seed` → cùng ảnh nhiễu, dùng để dựng lại đúng ảnh nhiễu khi
    chuyển qua lại xám/màu mà không mất dữ liệu gốc.
    """
    if kind == "Không thêm nhiễu":
        return base.copy()
    if kind == "Salt & Pepper":
        return add_salt_pepper_noise(base, density=density_pct / 100.0,
                                     seed=seed)
    return add_gaussian_noise(base, mean=mean, sigma=sigma, seed=seed)


def _quality_text(original: np.ndarray, compared: np.ndarray) -> str:
    """Chuỗi 'PSNR: … | SSIM: …' không bao giờ raise.

    SSIM không tính được với ảnh < 3×3 (ValueError) thì vẫn trả chuỗi với
    lời nhắn rõ ràng để phía gọi luôn hiển thị được kết quả lọc.
    """
    try:
        psnr = calculate_psnr(original, compared)
        ps = "∞" if psnr == float("inf") else f"{psnr:.2f} dB"
    except Exception:
        ps = "không tính được"
    try:
        ss = f"{calculate_ssim(original, compared):.4f}"
    except ValueError:
        ss = "không tính được (ảnh quá nhỏ)"
    return f"PSNR: {ps}   |   SSIM: {ss}"


def improvement_note(psnr_noisy: float, psnr_out: float) -> str:
    """Gợi ý khi kết quả lọc không cải thiện (hàm thuần).

    Phân biệt 3 trường hợp: giống hệt (cùng inf), chất lượng GIẢM (diff âm)
    và cải thiện nhỏ (0 <= diff < 0.5). Dấu chênh lệch định dạng đúng
    (+/-), không bao giờ ra '+-5.00 dB'.
    """
    if math.isnan(psnr_noisy) or math.isnan(psnr_out):
        return ""
    if math.isinf(psnr_noisy) and math.isinf(psnr_out):
        return ("   ⚠️ Kết quả giống hệt ảnh gốc — có thể ảnh chưa có nhiễu "
                "(kiểm tra lại σ/mật độ nhiễu).")
    diff = psnr_out - psnr_noisy
    if diff < 0:
        return ("   ⚠️ Kết quả tệ hơn ảnh nhiễu "
                f"({diff:+.2f} dB) — thử giảm kích thước cửa sổ hoặc đổi "
                "chế độ lọc.")
    if diff < 0.5:
        return ("   ⚠️ Kết quả gần như giống ảnh nhiễu "
                f"({diff:+.2f} dB) — thử tăng kích thước "
                "cửa sổ, tăng độ nhiễu, hoặc đổi chế độ lọc.")
    return ""


class MedianFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"Median Filters v{APP_VERSION} — Lọc trung vị khử nhiễu ảnh")
        self.geometry("1320x840")
        self.minsize(1150, 750)

        self.original: np.ndarray | None = None   # ảnh đang hiển thị/lọc
        self.noisy: np.ndarray | None = None      # ảnh nhiễu đang hiển thị
        self.result: np.ndarray | None = None     # ảnh sau lọc
        self.original_color: np.ndarray | None = None  # bản màu gốc (không mất)
        self._noise_seed: int | None = None  # seed lần tạo nhiễu gần nhất
        self.gray_mode = ctk.BooleanVar(value=False)
        self.busy = False
        self._guard = RunGuard()  # chống worker cũ ghi đè trạng thái mới
        self._inflight = 0  # số worker đang chạy (để không kẹt busy)
        self._busy_lock: list = []  # widget bị khóa khi busy

        self._build_layout()
        self._load_sample_if_exists()

    # ---------------- layout ----------------
    def _build_layout(self):
        # ===== Sidebar =====
        self.sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=(0, 0), pady=0)

        ctk.CTkLabel(self.sidebar, text="MEDIAN FILTER",
                     font=(FONT, 20, "bold")).pack(pady=(16, 2))
        ctk.CTkLabel(self.sidebar, text=f"bản v{APP_VERSION}  •  Khử nhiễu muối tiêu & Gaussian",
                     font=(FONT, 12), text_color="gray").pack(pady=(0, 12))

        # --- Ảnh đầu vào ---
        card_in = SectionCard(self.sidebar, "ẢNH ĐẦU VÀO")
        card_in.pack(fill="x", padx=12, pady=6)
        self.btn_upload = ctk.CTkButton(
            card_in.body, text="📂  Tải ảnh lên…",
            height=34, font=(FONT, 13, "bold"),
            command=self.on_upload)
        self.btn_upload.pack(fill="x", pady=3)
        self.btn_sample = ctk.CTkButton(
            card_in.body, text="🖼️  Dùng ảnh mẫu",
            height=32, font=(FONT, 12),
            fg_color="gray", hover_color="#5a5a5a",
            command=self.on_sample)
        self.btn_sample.pack(fill="x", pady=3)
        self.gray_switch = ctk.CTkSwitch(
            card_in.body, text="Chuyển sang ảnh xám",
            font=(FONT, 12),
            variable=self.gray_mode,
            command=self.on_gray_toggle)
        self.gray_switch.pack(anchor="w", pady=(8, 2))

        # --- Nhiễu ---
        card_noise = SectionCard(self.sidebar, "①  GIẢ LẬP NHIỄU")
        card_noise.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(card_noise.body, text="Loại nhiễu:",
                      font=(FONT, 12)).pack(anchor="w", pady=(2, 0))
        self.noise_type = ctk.CTkOptionMenu(
            card_noise.body,
            values=["Salt & Pepper", "Gaussian", "Không thêm nhiễu"],
            font=(FONT, 12),
            command=lambda _: self.on_noise_kind_changed())
        self.noise_type.set("Salt & Pepper")
        self.noise_type.pack(fill="x", pady=4)

        self.density_slider = LabeledSlider(
            card_noise.body, "Mật độ nhiễu", 1, 50, 10, fmt="{:.0f} %",
            command=lambda _: self._on_noise_settings_changed())
        self.density_slider.pack(fill="x", pady=2)
        self.sigma_slider = LabeledSlider(
            card_noise.body, "Độ lệch chuẩn σ", 0, 100, 25, fmt="{:.0f}",
            command=lambda _: self._on_noise_settings_changed())
        self.sigma_slider.pack(fill="x", pady=2)
        self.mean_slider = LabeledSlider(
            card_noise.body, "Giá trị trung bình μ", -50, 50, 0, fmt="{:.0f}",
            command=lambda _: self._on_noise_settings_changed())
        self.mean_slider.pack(fill="x", pady=2)
        self.btn_noise = ctk.CTkButton(
            card_noise.body, text="Tạo nhiễu",
            height=32, font=(FONT, 12),
            command=self.on_add_noise)
        self.btn_noise.pack(fill="x", pady=(6, 2))

        # --- Lọc ---
        card_filter = SectionCard(self.sidebar, "②  LỌC TRUNG VỊ")
        card_filter.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(card_filter.body, text="Kích thước cửa sổ:",
                      font=(FONT, 12)).pack(anchor="w", pady=(2, 0))
        self.ksize_menu = ctk.CTkOptionMenu(
            card_filter.body,
            values=["3 × 3", "5 × 5", "7 × 7", "9 × 9"],
            font=(FONT, 12),
            command=lambda _: self._on_filter_settings_changed())
        self.ksize_menu.set("3 × 3")
        self.ksize_menu.pack(fill="x", pady=4)

        ctk.CTkLabel(card_filter.body, text="Chế độ lọc:",
                      font=(FONT, 12)).pack(anchor="w", pady=(4, 0))
        self.mode_menu = ctk.CTkOptionMenu(
            card_filter.body,
            values=["optimized", "quickselect", "naive", "numba",
                    "mean", "gaussian"],
            font=(FONT, 12),
            command=lambda _: self.on_filter_mode_changed())
        self.mode_menu.set("optimized")
        self.mode_menu.pack(fill="x", pady=4)
        self.mode_hint = ctk.CTkLabel(
            card_filter.body, wraplength=260, justify="left",
            text="Median: trị nhiễu muối tiêu.",
            font=(FONT, 11), text_color="gray")
        self.mode_hint.pack(anchor="w", pady=(0, 2))

        ctk.CTkLabel(card_filter.body, text="Xử lý viền:",
                      font=(FONT, 12)).pack(anchor="w", pady=(4, 0))
        self.pad_menu = ctk.CTkOptionMenu(
            card_filter.body,
            values=["reflect", "replicate", "zero"],
            font=(FONT, 12),
            command=lambda _: self._on_filter_settings_changed())
        self.pad_menu.set("reflect")
        self.pad_menu.pack(fill="x", pady=4)

        self.btn_filter = ctk.CTkButton(
            card_filter.body, text="✨  Áp dụng lọc",
            height=36, font=(FONT, 13, "bold"),
            fg_color="#1F8A4C", hover_color="#17703D",
            command=self.on_apply_filter)
        self.btn_filter.pack(fill="x", pady=(8, 2))

        # --- Kết quả ---
        card_out = SectionCard(self.sidebar, "③  KẾT QUẢ")
        card_out.pack(fill="x", padx=12, pady=6)
        self.btn_save = ctk.CTkButton(
            card_out.body, text="💾  Lưu ảnh kết quả…",
            height=32, font=(FONT, 12),
            command=self.on_save)
        self.btn_save.pack(fill="x", pady=3)
        self.btn_bench = ctk.CTkButton(
            card_out.body, text="📊  So sánh tốc độ",
            height=32, font=(FONT, 12),
            fg_color="#555555", hover_color="#444444",
            command=self.on_benchmark)
        self.btn_bench.pack(fill="x", pady=3)
        self.dark_switch = ctk.CTkSwitch(card_out.body,
                                         text="Chế độ tối (Dark mode)",
                                         font=(FONT, 12),
                                         command=self.on_theme)
        self.dark_switch.pack(anchor="w", pady=(8, 2))
        self.dark_switch.select()

        ctk.CTkLabel(
            self.sidebar, wraplength=280, justify="left",
            text="💡 Mẹo: nhiễu muối tiêu 10–30% + cửa sổ 3×3 / 5×5 "
                 "cho kết quả tốt nhất, vẫn giữ nét viền.",
            font=(FONT, 11), text_color="gray").pack(padx=16, pady=10)

        # ===== Main area =====
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.pack(side="right", fill="both", expand=True,
                       padx=12, pady=12)

        self.status = ctk.CTkLabel(self.main, text="Sẵn sàng.",
                                   font=(FONT, 12), anchor="w")
        self.status.pack(fill="x", pady=(0, 8))
        self.progress = ctk.CTkProgressBar(self.main, mode="indeterminate")
        # progress chỉ pack khi busy (xem set_busy)

        panels = ctk.CTkFrame(self.main, fg_color="transparent")
        panels.pack(fill="both", expand=True)
        panels.columnconfigure((0, 1, 2), weight=1, uniform="img")
        panels.rowconfigure(0, weight=1)

        self.p_original = ImagePanel(panels, title="Ảnh gốc")
        self.p_original.grid(row=0, column=0, sticky="nsew", padx=5)
        self.p_noisy = ImagePanel(panels, title="Ảnh nhiễu")
        self.p_noisy.grid(row=0, column=1, sticky="nsew", padx=5)
        self.p_result = ImagePanel(panels, title="Ảnh sau lọc")
        self.p_result.grid(row=0, column=2, sticky="nsew", padx=5)

        self.metrics_card = ctk.CTkFrame(self.main, corner_radius=12)
        self.metrics_card.pack(fill="x", pady=(10, 0))
        self.metrics = ctk.CTkLabel(
            self.metrics_card,
            text="PSNR / SSIM sẽ hiển thị ở đây sau khi lọc.",
            font=(FONT, 13, "bold"), wraplength=850)
        self.metrics.pack(padx=16, pady=12)

        self.on_noise_kind_changed()
        self.on_filter_mode_changed()
        # Widget bị khóa khi worker đang chạy (tránh đổi dữ liệu giữa chừng)
        self._busy_lock = [
            self.btn_upload, self.btn_sample, self.gray_switch,
            self.noise_type, self.density_slider, self.sigma_slider,
            self.mean_slider, self.btn_noise,
            self.ksize_menu, self.mode_menu, self.pad_menu,
            self.btn_filter, self.btn_bench,
        ]

    # ---------------- helpers ----------------
    def set_busy(self, busy: bool, msg: str = ""):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for w in self._busy_lock:
            try:
                if hasattr(w, "set_state"):
                    w.set_state(state)  # LabeledSlider
                else:
                    w.configure(state=state)
            except Exception:
                pass
        if busy:
            self._busy_t0 = time.monotonic()
            self._busy_msg = msg
            self.progress.pack(fill="x", pady=(0, 8))
            self.progress.start()
            self._busy_tick()
        else:
            self.progress.stop()
            self.progress.pack_forget()
            if msg:
                self.status.configure(text=msg)
        self.update_idletasks()

    def _busy_tick(self):
        """Cập nhật 'đang chạy… Xs' mỗi 250ms để thấy app vẫn sống."""
        if not self.busy:
            return
        el = time.monotonic() - getattr(self, "_busy_t0", time.monotonic())
        self.status.configure(
            text=f"{getattr(self, '_busy_msg', 'Đang xử lý')} "
                 f"({el:.0f}s…) — vui lòng đợi")
        self.after(250, self._busy_tick)

    def get_ksize(self) -> int:
        # menu hiển thị "3 × 3" -> lấy số đầu
        return int(self.ksize_menu.get().split()[0])

    def _require_original(self) -> bool:
        if self.original is None:
            messagebox.showwarning("Chưa có ảnh",
                                   "Hãy tải ảnh lên hoặc dùng ảnh mẫu trước.")
            return False
        return True

    # ---------------- events ----------------
    def on_theme(self):
        ctk.set_appearance_mode("Dark" if self.dark_switch.get() else "Light")

    def on_noise_kind_changed(self):
        """Đổi loại nhiễu: xếp lại slider + đánh dấu kết quả cũ hết hạn."""
        self.on_noise_type_change()
        self._on_noise_settings_changed()

    def on_filter_mode_changed(self):
        """Đổi chế độ lọc: cập nhật gợi ý + đánh dấu kết quả cũ hết hạn."""
        self.on_filter_mode_change()
        self._on_filter_settings_changed()

    def _on_noise_settings_changed(self):
        self._invalidate_result(
            "Thông số nhiễu đã đổi — bấm “Tạo nhiễu” rồi “Áp dụng lọc” lại.")

    def _on_filter_settings_changed(self):
        self._invalidate_result(
            "Thông số lọc đã đổi — bấm “Áp dụng lọc” lại.")

    def _invalidate_result(self, hint: str):
        """Xóa kết quả cũ khi thông số đổi (tránh hiển thị ảnh lỗi thời)."""
        self.result = None
        self.p_result.set_image(None)
        self.metrics.configure(text=hint)

    def _active_base(self) -> np.ndarray:
        """Ảnh gốc theo chế độ xám/màu hiện tại (luôn suy từ bản màu gốc)."""
        assert self.original_color is not None
        if self.gray_mode.get():
            from utils.image_loader import to_grayscale

            return to_grayscale(self.original_color)
        return self.original_color.copy()

    def _refresh_images(self, status_msg: str = ""):
        """Dựng lại original/noisy/result từ bản màu gốc + seed đã lưu."""
        assert self.original_color is not None
        base = self._active_base()
        self.original = base
        if self._noise_seed is None:
            self.noisy = base.copy()
        else:
            self.noisy = make_noisy(
                base, self.noise_type.get(),
                density_pct=float(self.density_slider.get()),
                sigma=float(self.sigma_slider.get()),
                mean=float(self.mean_slider.get()),
                seed=self._noise_seed)
        self.result = None
        self.p_original.set_image(self.original, image_info(self.original))
        self.p_noisy.set_image(self.noisy, image_info(self.noisy))
        self.p_result.set_image(None)
        if status_msg:
            self.status.configure(text=status_msg)

    def on_filter_mode_change(self):
        m = self.mode_menu.get()
        if m in ("mean", "gaussian"):
            self.mode_hint.configure(
                text="Mean/Gaussian: trị nhiễu Gaussian. "
                     "Median yếu với loại nhiễu này.")
        else:
            self.mode_hint.configure(
                text="Median: trị nhiễu muối tiêu. "
                     "Với nhiễu Gaussian hãy chọn mean/gaussian.")

    def on_noise_type_change(self):
        t = self.noise_type.get()
        for w in (self.density_slider, self.sigma_slider, self.mean_slider):
            w.pack_forget()
        if t == "Salt & Pepper":
            self.density_slider.pack(fill="x", pady=2)
        elif t == "Gaussian":
            self.sigma_slider.pack(fill="x", pady=2)
            self.mean_slider.pack(fill="x", pady=2)
        # "Không thêm nhiễu": ẩn hết slider

    def on_upload(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh hỗ trợ", exts), ("Tất cả", "*.*")])
        if not path:
            return
        try:
            img = load_image(path)  # luôn giữ bản màu gốc
        except Exception as e:
            messagebox.showerror("Lỗi đọc ảnh", str(e))
            return
        self.original_color = img
        self._noise_seed = None
        self._guard.invalidate()  # worker lọc cũ (nếu có) thành stale
        self._refresh_images()
        self.metrics.configure(
            text="Đã tải ảnh. Hãy bấm “Tạo nhiễu” rồi “Áp dụng lọc”.")
        self.status.configure(text=f"Đã tải: {os.path.basename(path)}")

    def on_sample(self) -> bool:
        """Nạp ảnh mẫu. Trả về True nếu thành công, False nếu lỗi."""
        path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "assets", "sample.png")
        if not os.path.isfile(path):
            self._make_sample(path)
        try:
            img = load_image(path)  # luôn giữ bản màu gốc
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            return False
        self.original_color = img
        self._noise_seed = None
        self._guard.invalidate()  # worker lọc cũ (nếu có) thành stale
        self._refresh_images()
        self.metrics.configure(
            text="Đã nạp ảnh mẫu. Hãy bấm “Tạo nhiễu” rồi “Áp dụng lọc”.")
        return True

    def _make_sample(self, path: str):
        """Tạo ảnh mẫu gradient + hình học nếu chưa có."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h, w = 256, 256
        y, x = np.mgrid[0:h, 0:w]
        r = ((x - 64) % 256).astype(np.uint8)
        g = ((y - 64) % 256).astype(np.uint8)
        b = (((x + y) // 2) % 256).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        import cv2

        cv2.rectangle(img, (40, 40), (120, 120), (255, 255, 255), -1)
        cv2.circle(img, (180, 180), 40, (0, 0, 0), -1)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _load_sample_if_exists(self):
        # Chỉ báo "Sẵn sàng" khi ảnh mẫu nạp thành công thật.
        try:
            ok = self.on_sample()
        except Exception:
            ok = False
        if ok:
            self.metrics.configure(
                text="Sẵn sàng. Chỉnh thông số bên trái rồi bấm "
                     "“Tạo nhiễu” → “Áp dụng lọc”.")
        else:
            self.status.configure(text="Lỗi nạp ảnh mẫu.")
            self.metrics.configure(
                text="Không nạp được ảnh mẫu — hãy bấm “Tải ảnh lên…”.")

    def on_gray_toggle(self):
        """Đổi xám/màu: dựng lại từ bản màu gốc (không mất dữ liệu).

        Nhiễu được tạo lại với cùng seed nên giữ nguyên mẫu nhiễu cũ;
        kết quả lọc cũ bị xóa vì không còn khớp ảnh hiện tại.
        """
        if self.original_color is None:
            return
        self._guard.invalidate()  # dữ liệu nguồn đổi → worker cũ thành stale
        self._refresh_images()
        mode_txt = "ảnh xám" if self.gray_mode.get() else "ảnh màu"
        assert self.original is not None and self.noisy is not None
        self.metrics.configure(
            text=f"Đã chuyển sang {mode_txt}. "
                 f"{_quality_text(self.original, self.noisy)} — "
                 f"hãy “Áp dụng lọc” lại.")

    def on_add_noise(self):
        if not self._require_original():
            return
        assert self.original_color is not None
        # seed mới mỗi lần tạo → mẫu nhiễu khác nhau; seed được giữ lại để
        # dựng đúng ảnh nhiễu khi chuyển xám/màu
        self._noise_seed = random.randint(0, 2**31 - 1)
        try:
            base = self._active_base()
            self.original = base
            self.noisy = make_noisy(
                base, self.noise_type.get(),
                density_pct=float(self.density_slider.get()),
                sigma=float(self.sigma_slider.get()),
                mean=float(self.mean_slider.get()),
                seed=self._noise_seed)
        except Exception as e:
            messagebox.showerror("Lỗi tạo nhiễu", str(e))
            return
        self.result = None
        self._guard.invalidate()  # kết quả lọc cũ không còn hợp lệ
        self.p_original.set_image(self.original,
                                  image_info(self.original))
        self.p_noisy.set_image(self.noisy,
                               image_info(self.noisy) + "  •  đã thêm nhiễu")
        self.p_result.set_image(None)
        extra = ""
        try:
            if calculate_psnr(self.original, self.noisy) == float("inf"):
                extra = ("   ⚠️ Ảnh nhiễu giống hệt ảnh gốc — kiểm tra lại "
                         "σ/mật độ nhiễu.")
        except Exception:
            pass
        self.metrics.configure(
            text=f"Ảnh nhiễu  •  {_quality_text(self.original, self.noisy)}"
                 f"{extra}")
        self.status.configure(text="Đã tạo nhiễu.")

    def on_apply_filter(self):
        if not self._require_original():
            return
        if self.noisy is None:
            self.noisy = (self.original.copy()
                          if self.original is not None else None)
        if self.busy:
            messagebox.showinfo("Đang bận",
                                "Đang lọc/đo tốc độ — vui lòng đợi xong "
                                "rồi chạy tiếp.")
            return
        ksize = self.get_ksize()
        mode = self.mode_menu.get()
        padding = self.pad_menu.get()
        noisy = self.noisy.copy() if self.noisy is not None else None
        original = self.original.copy() if self.original is not None else None
        if noisy is None or original is None:
            return
        noise_kind = self.noise_type.get()
        h, w = noisy.shape[:2]
        channels = 1 if noisy.ndim == 2 else noisy.shape[2]
        est_ms = estimate_filter_ms(h, w, channels, ksize, mode)
        if est_ms > 3000:
            choice = messagebox.askyesnocancel(
                "Lọc chậm",
                f"Chế độ {mode} với ảnh {w}×{h} + cửa sổ {ksize}×{ksize} "
                f"ước tính ~{est_ms / 1000:.0f}s, giao diện sẽ khóa trong lúc lọc.\n\n"
                "Yes: dùng optimized (vài chục ms, khuyên dùng)\n"
                f"No: vẫn dùng {mode}\n"
                "Cancel: hủy")
            if choice is None:
                return
            if choice:
                mode = "optimized"
                self.mode_menu.set("optimized")
        seq = self._guard.start()
        self._inflight += 1

        self.set_busy(True, f"Đang lọc (cửa sổ {ksize}×{ksize}, {mode})…")

        def worker():
            try:
                t0 = time.perf_counter()
                out = median_filter(noisy, ksize, mode=mode, padding=padding)
                dt = (time.perf_counter() - t0) * 1000.0
                tq0 = time.perf_counter()
                text = _quality_text(original, out)
                qual_ms = (time.perf_counter() - tq0) * 1000.0
                try:
                    psnr_noisy = calculate_psnr(original, noisy)
                    psnr_out = calculate_psnr(original, out)
                except Exception:
                    psnr_noisy, psnr_out = float("nan"), float("nan")
            except Exception as exc:  # chốt lỗi, không closure lên `exc`
                self.after(0, self._fail_filter, seq, _freeze_error(exc))
                return
            self.after(0, self._finish_filter, seq, out, ksize, mode,
                       padding, noise_kind, dt, qual_ms, text,
                       psnr_noisy, psnr_out)

        try:
            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            # Không khởi chạy được worker → trả lại trạng thái rảnh ngay.
            self._inflight -= 1
            self._fail_filter(seq, _freeze_error(exc))

    def _release_worker(self, seq: int) -> bool:
        """Trừ worker đã xong; nếu là tác vụ cũ thì chỉ gỡ busy khi hết."""
        self._inflight = max(0, self._inflight - 1)
        if not self._guard.is_current(seq):
            if self._inflight == 0:
                self.set_busy(False, "Đã hủy tác vụ cũ.")
            return False  # worker cũ → bỏ qua, không ghi đè ảnh mới
        return True

    def _finish_filter(self, seq: int, out: np.ndarray, ksize: int,
                       mode: str, padding: str, noise_kind: str,
                       dt_ms: float, qual_ms: float, quality_text: str,
                       psnr_noisy: float = float("nan"),
                       psnr_out: float = float("nan")):
        if not self._release_worker(seq):
            return
        self.result = out
        self.p_result.set_image(out, image_info(out))
        note = ""
        if noise_kind == "Gaussian" and mode not in ("mean", "gaussian"):
            note = ("   ⚠️ Median yếu với nhiễu Gaussian — "
                    "thử lại với mean/gaussian.")
        note += improvement_note(psnr_noisy, psnr_out)
        self.metrics.configure(
            text=f"Cửa sổ {ksize}×{ksize}  •  {mode}  •  {padding}   "
                 f"|   {quality_text}   |   Lọc: {dt_ms:.1f} ms   •   "
                 f"Đo: {qual_ms:.1f} ms{note}")
        self.set_busy(False, "Lọc xong.")

    def _fail_filter(self, seq: int, err_msg: str):
        if not self._release_worker(seq):
            return
        self.set_busy(False, "Lỗi khi lọc.")
        messagebox.showerror("Lỗi lọc ảnh", err_msg)

    def on_save(self):
        if self.result is None:
            messagebox.showwarning("Chưa có kết quả",
                                   "Hãy áp dụng lọc trước khi lưu.")
            return
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp"),
                       ("TIFF", "*.tiff")])
        if not path:
            return
        try:
            save_image(path, self.result)
            self.status.configure(text=f"Đã lưu: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Lỗi lưu ảnh", str(e))

    def on_benchmark(self):
        if not self._require_original():
            return
        if self.busy:
            messagebox.showinfo("Đang bận",
                                "Đang lọc/đo tốc độ — vui lòng đợi xong "
                                "rồi chạy tiếp.")
            return
        src = (self.noisy if self.noisy is not None else self.original)
        assert src is not None
        h, w = src.shape[:2]
        bench_img = src
        if max(h, w) > 512:
            import cv2

            scale = 512 / max(h, w)
            nw, nh = int(w * scale), int(h * scale)
            bench_img = cv2.resize(src, (nw, nh))
        bw, bh = bench_img.shape[1], bench_img.shape[0]
        ksizes = [3, 5, 7, 9]
        if src.ndim == 3:
            modes = ["optimized", "quickselect", "mean", "gaussian"]
        else:
            modes = ["optimized", "quickselect", "naive", "mean", "gaussian"]
        padding = self.pad_menu.get()
        seq = self._guard.start()
        self._inflight += 1
        self.set_busy(True, "Đang đo tốc độ… (chạy nền)")

        def worker():
            try:
                results = benchmark_filters(bench_img, ksizes, modes,
                                            repeat=1, padding=padding)
            except Exception as exc:  # chốt lỗi, không closure lên `exc`
                self.after(0, self._fail_benchmark, seq, _freeze_error(exc))
                return
            self.after(0, self._finish_benchmark, seq, bw, bh, ksizes,
                       modes, results)

        try:
            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            self._inflight -= 1
            self._fail_benchmark(seq, _freeze_error(exc))

    def _finish_benchmark(self, seq: int, bw: int, bh: int,
                          ksizes: list, modes: list,
                          results: dict[str, list[float]]):
        if not self._release_worker(seq):
            return  # đã có tác vụ mới hơn → bỏ qua
        self.set_busy(False, "Đo xong.")
        import matplotlib

        matplotlib.use("TkAgg")  # lazy: nap ~0.3s chi khi can ve do thi
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 4.5))
        for m in modes:
            plt.plot(ksizes, results[m], marker="o", label=m)
        plt.xticks(ksizes)
        plt.xlabel("Kernel size")
        plt.ylabel("Thời gian (ms)")
        plt.title(f"So sánh tốc độ ({bw}×{bh})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # non-blocking: giu app chinh tuong tac duoc khi mo bieu do
        if not hasattr(self, "_bench_figs"):
            self._bench_figs = []
        self._bench_figs.append(fig)
        if len(self._bench_figs) > 5:  # chong day RAM khi mo nhieu lan
            old = self._bench_figs.pop(0)
            try:
                plt.close(old)
            except Exception:
                pass
        fig.show()

    def _fail_benchmark(self, seq: int, err_msg: str):
        if not self._release_worker(seq):
            return
        self.set_busy(False, "Lỗi khi đo tốc độ.")
        messagebox.showerror("Lỗi benchmark", err_msg)


def run():
    app = MedianFilterApp()
    app.mainloop()


if __name__ == "__main__":
    run()
