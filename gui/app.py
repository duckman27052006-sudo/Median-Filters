"""Màn hình chính Median Filter App (CustomTkinter)."""

from __future__ import annotations

import os
import sys
import threading

import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog, messagebox

# Cho phép chạy cả từ root (main.py) và trực tiếp trong gui/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filters import benchmark_filters, median_filter
from core.metrics import calculate_psnr, calculate_ssim
from core.noise import add_gaussian_noise, add_salt_pepper_noise
from gui.components import ImagePanel, LabeledSlider, RunGuard, SectionCard
from utils.image_loader import SUPPORTED_EXTS, image_info, load_image, save_image

FONT = "Segoe UI"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def _freeze_error(exc: BaseException) -> str:
    """Chốt thông điệp ngoại lệ thành chuỗi bền vững.

    Tránh closure tham chiếu trực tiếp biến `e` của khối `except` (Python
    xóa biến đó khi thoát khối, callback chạy sau sẽ gặp NameError).
    """
    return f"{type(exc).__name__}: {exc}"


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


class MedianFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Median Filters — Lọc trung vị khử nhiễu ảnh")
        self.geometry("1320x840")
        self.minsize(1150, 750)

        self.original: np.ndarray | None = None   # ảnh gốc
        self.noisy: np.ndarray | None = None      # ảnh nhiễu
        self.result: np.ndarray | None = None     # ảnh sau lọc
        self.gray_mode = ctk.BooleanVar(value=False)
        self.busy = False
        self._guard = RunGuard()  # chống worker cũ ghi đè trạng thái mới

        self._build_layout()
        self._load_sample_if_exists()

    # ---------------- layout ----------------
    def _build_layout(self):
        # ===== Sidebar =====
        self.sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=(0, 0), pady=0)

        ctk.CTkLabel(self.sidebar, text="MEDIAN FILTER",
                     font=(FONT, 20, "bold")).pack(pady=(16, 2))
        ctk.CTkLabel(self.sidebar, text="Khử nhiễu muối tiêu & Gaussian",
                     font=(FONT, 12), text_color="gray").pack(pady=(0, 12))

        # --- Ảnh đầu vào ---
        card_in = SectionCard(self.sidebar, "ẢNH ĐẦU VÀO")
        card_in.pack(fill="x", padx=12, pady=6)
        ctk.CTkButton(card_in.body, text="📂  Tải ảnh lên…",
                      height=34, font=(FONT, 13, "bold"),
                      command=self.on_upload).pack(fill="x", pady=3)
        ctk.CTkButton(card_in.body, text="🖼️  Dùng ảnh mẫu",
                      height=32, font=(FONT, 12),
                      fg_color="gray", hover_color="#5a5a5a",
                      command=self.on_sample).pack(fill="x", pady=3)
        ctk.CTkSwitch(card_in.body, text="Chuyển sang ảnh xám",
                      font=(FONT, 12),
                      variable=self.gray_mode,
                      command=self.on_gray_toggle).pack(anchor="w", pady=(8, 2))

        # --- Nhiễu ---
        card_noise = SectionCard(self.sidebar, "①  GIẢ LẬP NHIỄU")
        card_noise.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(card_noise.body, text="Loại nhiễu:",
                      font=(FONT, 12)).pack(anchor="w", pady=(2, 0))
        self.noise_type = ctk.CTkOptionMenu(
            card_noise.body,
            values=["Salt & Pepper", "Gaussian", "Không thêm nhiễu"],
            font=(FONT, 12),
            command=lambda _: self.on_noise_type_change())
        self.noise_type.set("Salt & Pepper")
        self.noise_type.pack(fill="x", pady=4)

        self.density_slider = LabeledSlider(card_noise.body,
                                            "Mật độ nhiễu", 1, 50, 10,
                                            fmt="{:.0f} %")
        self.density_slider.pack(fill="x", pady=2)
        self.sigma_slider = LabeledSlider(card_noise.body,
                                          "Độ lệch chuẩn σ", 0, 100, 25,
                                          fmt="{:.0f}")
        self.sigma_slider.pack(fill="x", pady=2)
        self.mean_slider = LabeledSlider(card_noise.body,
                                         "Giá trị trung bình μ", -50, 50, 0,
                                         fmt="{:.0f}")
        self.mean_slider.pack(fill="x", pady=2)
        ctk.CTkButton(card_noise.body, text="Tạo nhiễu",
                      height=32, font=(FONT, 12),
                      command=self.on_add_noise).pack(fill="x", pady=(6, 2))

        # --- Lọc ---
        card_filter = SectionCard(self.sidebar, "②  LỌC TRUNG VỊ")
        card_filter.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(card_filter.body, text="Kích thước cửa sổ:",
                      font=(FONT, 12)).pack(anchor="w", pady=(2, 0))
        self.ksize_menu = ctk.CTkOptionMenu(card_filter.body,
                                            values=["3 × 3", "5 × 5",
                                                    "7 × 7", "9 × 9"],
                                            font=(FONT, 12))
        self.ksize_menu.set("3 × 3")
        self.ksize_menu.pack(fill="x", pady=4)

        ctk.CTkLabel(card_filter.body, text="Chế độ lọc:",
                      font=(FONT, 12)).pack(anchor="w", pady=(4, 0))
        self.mode_menu = ctk.CTkOptionMenu(
            card_filter.body,
            values=["optimized", "quickselect", "naive", "numba",
                    "mean", "gaussian"],
            font=(FONT, 12),
            command=lambda _: self.on_filter_mode_change())
        self.mode_menu.set("optimized")
        self.mode_menu.pack(fill="x", pady=4)
        self.mode_hint = ctk.CTkLabel(
            card_filter.body, wraplength=260, justify="left",
            text="Median: trị nhiễu muối tiêu.",
            font=(FONT, 11), text_color="gray")
        self.mode_hint.pack(anchor="w", pady=(0, 2))

        ctk.CTkLabel(card_filter.body, text="Xử lý viền:",
                      font=(FONT, 12)).pack(anchor="w", pady=(4, 0))
        self.pad_menu = ctk.CTkOptionMenu(card_filter.body,
                                          values=["reflect", "replicate",
                                                  "zero"],
                                          font=(FONT, 12))
        self.pad_menu.set("reflect")
        self.pad_menu.pack(fill="x", pady=4)

        ctk.CTkButton(card_filter.body, text="✨  Áp dụng lọc",
                      height=36, font=(FONT, 13, "bold"),
                      fg_color="#1F8A4C", hover_color="#17703D",
                      command=self.on_apply_filter).pack(fill="x", pady=(8, 2))

        # --- Kết quả ---
        card_out = SectionCard(self.sidebar, "③  KẾT QUẢ")
        card_out.pack(fill="x", padx=12, pady=6)
        ctk.CTkButton(card_out.body, text="💾  Lưu ảnh kết quả…",
                      height=32, font=(FONT, 12),
                      command=self.on_save).pack(fill="x", pady=3)
        ctk.CTkButton(card_out.body, text="📊  So sánh tốc độ",
                      height=32, font=(FONT, 12),
                      fg_color="#555555", hover_color="#444444",
                      command=self.on_benchmark).pack(fill="x", pady=3)
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

        self.on_noise_type_change()
        self.on_filter_mode_change()

    # ---------------- helpers ----------------
    def set_busy(self, busy: bool, msg: str = ""):
        self.busy = busy
        if msg:
            self.status.configure(text=msg)
        self.update_idletasks()

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
            img = load_image(path, grayscale=self.gray_mode.get())
        except Exception as e:
            messagebox.showerror("Lỗi đọc ảnh", str(e))
            return
        self.original = img
        self.noisy = img.copy()
        self.result = None
        self._guard.invalidate()  # worker lọc cũ (nếu có) thành stale
        self.p_original.set_image(img, image_info(img))
        self.p_noisy.set_image(self.noisy,
                               image_info(self.noisy) + "  •  chưa thêm nhiễu")
        self.p_result.set_image(None)
        self.metrics.configure(
            text="Đã tải ảnh. Hãy bấm “Tạo nhiễu” rồi “Áp dụng lọc”.")
        self.status.configure(text=f"Đã tải: {os.path.basename(path)}")

    def on_sample(self):
        path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "assets", "sample.png")
        if not os.path.isfile(path):
            self._make_sample(path)
        try:
            img = load_image(path, grayscale=self.gray_mode.get())
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            return
        self.original = img
        self.noisy = img.copy()
        self.result = None
        self._guard.invalidate()  # worker lọc cũ (nếu có) thành stale
        self.p_original.set_image(img, image_info(img))
        self.p_noisy.set_image(self.noisy, image_info(self.noisy))
        self.p_result.set_image(None)
        self.metrics.configure(
            text="Đã nạp ảnh mẫu. Hãy bấm “Tạo nhiễu” rồi “Áp dụng lọc”.")

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
        try:
            self.on_sample()
            self.metrics.configure(
                text="Sẵn sàng. Chỉnh thông số bên trái rồi bấm "
                     "“Tạo nhiễu” → “Áp dụng lọc”.")
        except Exception:
            pass

    def on_gray_toggle(self):
        if self.original is None:
            return
        from utils.image_loader import to_grayscale, to_rgb

        self._guard.invalidate()  # dữ liệu nguồn đổi → worker cũ thành stale

        if self.gray_mode.get():
            self.original = to_grayscale(self.original)
            self.noisy = to_grayscale(self.noisy if self.noisy is not None
                                      else self.original)
            if self.result is not None:
                self.result = to_grayscale(self.result)
        else:
            self.original = to_rgb(self.original)
            self.noisy = to_rgb(self.noisy if self.noisy is not None
                                else self.original)
            if self.result is not None:
                self.result = to_rgb(self.result)
        self.p_original.set_image(self.original, image_info(self.original))
        self.p_noisy.set_image(self.noisy, image_info(self.noisy))
        self.p_result.set_image(
            self.result,
            image_info(self.result) if self.result is not None else "")

    def on_add_noise(self):
        if not self._require_original():
            return
        assert self.original is not None
        t = self.noise_type.get()
        try:
            if t == "Không thêm nhiễu":
                self.noisy = self.original.copy()
            elif t == "Salt & Pepper":
                d = float(self.density_slider.get()) / 100.0
                self.noisy = add_salt_pepper_noise(self.original, density=d)
            else:
                sigma = float(self.sigma_slider.get())
                mean = float(self.mean_slider.get())
                self.noisy = add_gaussian_noise(self.original, mean=mean,
                                               sigma=sigma)
        except Exception as e:
            messagebox.showerror("Lỗi tạo nhiễu", str(e))
            return
        self.result = None
        self._guard.invalidate()  # kết quả lọc cũ không còn hợp lệ
        self.p_noisy.set_image(self.noisy,
                               image_info(self.noisy) + "  •  đã thêm nhiễu")
        self.p_result.set_image(None)
        self.metrics.configure(
            text=f"Ảnh nhiễu  •  {_quality_text(self.original, self.noisy)}")
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
        seq = self._guard.start()

        self.set_busy(True, f"Đang lọc (cửa sổ {ksize}×{ksize}, {mode})…")

        def worker():
            import time

            try:
                t0 = time.perf_counter()
                out = median_filter(noisy, ksize, mode=mode, padding=padding)
                dt = (time.perf_counter() - t0) * 1000.0
                text = _quality_text(original, out)
            except Exception as exc:  # chốt lỗi, không closure lên `exc`
                self.after(0, self._fail_filter, seq, _freeze_error(exc))
                return
            self.after(0, self._finish_filter, seq, out, ksize, mode,
                       padding, noise_kind, dt, text)

        try:
            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            # Không khởi chạy được worker → trả lại trạng thái rảnh ngay.
            self._fail_filter(seq, _freeze_error(exc))

    def _finish_filter(self, seq: int, out: np.ndarray, ksize: int,
                       mode: str, padding: str, noise_kind: str,
                       dt_ms: float, quality_text: str):
        if not self._guard.is_current(seq):
            return  # worker cũ → bỏ qua, không ghi đè ảnh mới
        self.result = out
        self.p_result.set_image(out, image_info(out))
        note = ""
        if noise_kind == "Gaussian" and mode not in ("mean", "gaussian"):
            note = ("   ⚠️ Median yếu với nhiễu Gaussian — "
                    "thử lại với mean/gaussian.")
        self.metrics.configure(
            text=f"Cửa sổ {ksize}×{ksize}  •  {mode}  •  {padding}   "
                 f"|   {quality_text}   |   Thời gian: {dt_ms:.1f} ms{note}")
        self.set_busy(False, "Lọc xong.")

    def _fail_filter(self, seq: int, err_msg: str):
        if not self._guard.is_current(seq):
            return  # worker cũ → bỏ qua
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
            self._fail_benchmark(seq, _freeze_error(exc))

    def _finish_benchmark(self, seq: int, bw: int, bh: int,
                          ksizes: list, modes: list,
                          results: dict[str, list[float]]):
        if not self._guard.is_current(seq):
            return  # đã có tác vụ mới hơn → bỏ qua
        self.set_busy(False, "Đo xong.")
        plt.figure(figsize=(7, 4.5))
        for m in modes:
            plt.plot(ksizes, results[m], marker="o", label=m)
        plt.xticks(ksizes)
        plt.xlabel("Kernel size")
        plt.ylabel("Thời gian (ms)")
        plt.title(f"So sánh tốc độ ({bw}×{bh})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _fail_benchmark(self, seq: int, err_msg: str):
        if not self._guard.is_current(seq):
            return
        self.set_busy(False, "Lỗi khi đo tốc độ.")
        messagebox.showerror("Lỗi benchmark", err_msg)


def run():
    app = MedianFilterApp()
    app.mainloop()


if __name__ == "__main__":
    run()
