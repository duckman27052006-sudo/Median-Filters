# Median Filters — Ứng dụng lọc trung vị khử nhiễu ảnh

Ứng dụng trực quan: tải ảnh → giả lập nhiễu (Salt & Pepper / Gaussian) →
lọc nhiễu → so sánh trực quan + đo **PSNR / SSIM / thời gian xử lý**.

- **Median** (`optimized` / `quickselect` / `naive` / `numba`): trị nhiễu
  **muối tiêu** rất tốt (PSNR 13 → 34 dB ở mật độ 20%).
- **Mean / Gaussian blur**: bù chỗ yếu của Median với nhiễu **Gaussian**
  (Median chỉ 20 → 28 dB, Mean đạt ~34 dB cùng ảnh σ = 25).

Thực hiện theo `k_ho_ch_x_y_d_ng_ng_d_ng_median_filters.md` (Phase 1 → 4);
các điểm rà soát trong `CODE_REVIEW.md` đã được sửa hết.

## Cấu trúc

```text
median-filter-app/
├── assets/sample.png      # ảnh mẫu tự tạo nếu thiếu
├── core/
│   ├── filters.py         # median (naive/quickselect/numba/opencv),
│   │                      # mean, gaussian + timed_filter, benchmark_filters
│   ├── noise.py           # salt & pepper (đúng mật độ), gaussian
│   └── metrics.py         # PSNR, SSIM (tự thu cửa sổ ảnh nhỏ)
├── gui/
│   ├── app.py             # màn hình chính: threading, RunGuard, benchmark nền,
│   │                      # _quality_text/_freeze_error
│   └── components.py      # ImagePanel, LabeledSlider, SectionCard, RunGuard
├── utils/
│   └── image_loader.py    # load/save JPG/PNG/BMP/TIFF/WEBP, xám/màu, image_info
├── tests/
│   ├── test_filters.py    # lọc, nhiễu, PSNR/SSIM, padding, benchmark
│   └── test_gui_helpers.py# RunGuard, _freeze_error, _quality_text
├── main.py                # entry point (mở GUI)
├── benchmark.py           # benchmark CLI → PNG chart
├── requirements.txt
├── README.md
└── CODE_REVIEW.md         # báo cáo rà soát + trạng thái sửa
```

## Cài đặt

```powershell
cd "median-filter-app"
pip install -r requirements.txt
```

Yêu cầu: Python 3.10+ (đã kiểm trên 3.14), Windows/macOS/Linux.

## Chạy

```powershell
python main.py
```

Thao tác trên GUI:

1. **Ảnh đầu vào**: 📂 *Tải ảnh lên…* (JPG/PNG/BMP/TIFF/WEBP) hoặc
   🖼️ *Dùng ảnh mẫu*; công tắc *Chuyển sang ảnh xám* đổi qua lại xám/màu.
2. **① Giả lập nhiễu** → **Tạo nhiễu**
   - Salt & Pepper: *Mật độ nhiễu* 1–50% (đúng tỉ lệ pixel đã chọn).
   - Gaussian: *σ* 0–100, *μ* −50–50.
   - *Không thêm nhiễu*: giữ nguyên ảnh gốc để test lọc.
3. **② Lọc trung vị**: chọn *Kích thước cửa sổ* (3×3–9×9), *Chế độ lọc*,
   *Xử lý viền* (`reflect`/`replicate`/`zero` — áp dụng cho **mọi** mode) →
   ✨ **Áp dụng lọc** (chạy nền, không đơ giao diện).
4. **③ Kết quả**: xem PSNR/SSIM/thời gian, 💾 *Lưu ảnh*, 📊 *So sánh tốc độ*
   (đo nền theo kernel, vẽ biểu đồ Matplotlib).

Mẹo:

- Muối tiêu 10–30% + cửa sổ 3×3/5×5 + `optimized` → sạch nhiễu mà giữ viền.
- Nhiễu Gaussian → chọn `mean` hoặc `gaussian` thay vì Median (app sẽ hiện
  cảnh báo ⚠️ nếu bạn dùng Median cho nhiễu Gaussian).
- Ảnh đang lọc/đo mà bấm tiếp → app báo “Đang bận”; tải ảnh khác giữa chừng
  thì kết quả cũ tự bị hủy, không ghi đè nhầm.

## So sánh chế độ lọc

| Mode | Thuật toán | Mạnh với | Tốc độ (FullHD RGB) |
|---|---|---|---|
| `optimized` | `cv2.medianBlur` (C++) | Muối tiêu | k3 ~2 ms, k5 ~7.5 ms |
| `quickselect` | `np.partition` (NumPy thuần) | Muối tiêu | chậm hơn ~1000× (vòng lặp Python, để học) |
| `naive` | `np.median` full-sort | Muối tiêu (học tập) | chậm nhất |
| `numba` | Numba JIT (thiếu lib → fallback `quickselect`) | Muối tiêu | ≈ quickselect khi fallback |
| `mean` | `cv2.blur` | **Gaussian** | ≈ optimized |
| `gaussian` | `cv2.GaussianBlur` | **Gaussian** | ≈ optimized |

## Test

```powershell
python -m pytest tests/ -v
# 34 passed (13 core + 21 hồi quy cho CODE_REVIEW)
```

Bao phủ: padding mọi mode, SSIM ảnh 1×1–10×10, stale-worker, callback lỗi,
benchmark helper, mật độ S&P chính xác/tách muối-tiêu/seed lặp lại.

## Benchmark CLI (không cần mở GUI)

```powershell
python benchmark.py --image assets/sample.png --kernels 3 5 7 9 --repeat 3
# -> benchmark_result.png + bảng thời gian (ms, best-of-repeat)
```

## Đóng gói .exe (tùy chọn)

```powershell
pip install pyinstaller
pyinstaller --onefile --windowed --name MedianFilterApp main.py
# file exe nằm trong dist/
```

## Ghi chú kỹ thuật

- Padding: `optimized` + `replicate` gọi OpenCV trực tiếp (hành vi gốc);
  `reflect`/`zero` tự pad → blur → cắt viền nên **khớp chính xác** bản `naive`.
- S&P: `density` là tỉ lệ pixel mục tiêu chính xác (chọn duy nhất không lặp,
  muối/tiêu rời nhau); cùng `seed` → cùng ảnh.
- PSNR ảnh giống hệt nhau trả về `inf`. SSIM tự thu cửa sổ 7/5/3 cho ảnh nhỏ;
  ảnh dưới 3×3 hiện “SSIM không tính được (ảnh quá nhỏ)” nhưng ảnh lọc vẫn hiển thị.
- Threading: `RunGuard` đánh phiên worker — callback cũ (stale) bị bỏ qua;
  lỗi worker được chốt bằng `_freeze_error()` nên thông báo gốc không bao giờ mất.
