# Báo cáo rà soát mã nguồn — Median Filter App

Phạm vi rà soát: mã nguồn, giao diện, bộ lọc, sinh nhiễu, chỉ số đánh giá, kiểm thử và tài liệu dự án.

## Tóm tắt

Đã ghi nhận 6 điểm cần cải thiện trong mã nguồn và 1 điểm cần cập nhật trong tài liệu:

- **P2 — Trung bình:** 4 vấn đề ảnh hưởng trực tiếp đến kết quả hoặc trải nghiệm sử dụng.
- **P3 — Thấp:** 2 vấn đề về độ chính xác của sinh nhiễu và xử lý lỗi bất đồng bộ.
- **Tài liệu:** README ghi số lượng test không còn khớp.

Các test hiện có đã chạy thành công: **13 passed**. Việc test hiện tại qua không loại trừ các lỗi ở trường hợp biên hoặc luồng thao tác chưa được kiểm thử.

## Các điểm cần cải thiện

### 1. Chế độ `optimized` không áp dụng lựa chọn padding

- **Mức độ:** P2 — Trung bình
- **Vị trí:** [`core/filters.py`](./core/filters.py), hàm `median_filter` và `median_filter_opencv`
- **Hiện trạng:** Dispatcher chuyển `optimized`/`opencv` trực tiếp sang `median_filter_opencv` mà không truyền tham số `padding`. `cv2.medianBlur` sử dụng quy tắc xử lý biên riêng.
- **Ảnh hưởng:** Người dùng có thể chọn `reflect`, `zero` hoặc `replicate` trong giao diện nhưng kết quả ở chế độ `optimized` không thay đổi theo lựa chọn đó. Pixel gần mép ảnh có thể khác kết quả mong đợi và không đồng nhất với các mode khác.
- **Hướng cải thiện:** Hoặc triển khai padding có chủ đích cho mode OpenCV, hoặc vô hiệu hóa/ẩn lựa chọn padding khi mode không hỗ trợ và thông báo rõ quy tắc biên đang dùng. Giữ hành vi cũ có chủ ý thì cần ghi rõ trong giao diện và README.
- **Kiểm thử cần bổ sung:** Dùng ảnh nhỏ có pixel khác biệt sát biên; xác nhận từng giá trị padding tạo kết quả đúng theo quy tắc được công bố. Kiểm tra thêm mode `mean` và `gaussian`.

### 2. SSIM thất bại với ảnh nhỏ hơn cửa sổ mặc định

- **Mức độ:** P2 — Trung bình
- **Vị trí:** [`core/metrics.py`](./core/metrics.py), hàm `calculate_ssim`; được gọi trong luồng áp dụng lọc tại [`gui/app.py`](./gui/app.py)
- **Hiện trạng:** SSIM dùng cửa sổ mặc định 7×7. Ảnh có chiều nhỏ hơn 7 pixel gây `ValueError` từ scikit-image.
- **Ảnh hưởng:** Trong GUI, bộ lọc có thể hoàn tất nhưng bước tính SSIM thất bại trước khi kết quả được hiển thị. Người dùng không nhận được ảnh đã lọc.
- **Hướng cải thiện:** Chọn kích thước cửa sổ SSIM lẻ phù hợp với kích thước ảnh nhỏ nhất (nếu hợp lệ), hoặc báo lỗi đầu vào cụ thể và vẫn hiển thị kết quả lọc khi chỉ số SSIM không tính được. Không nên âm thầm bỏ qua lỗi.
- **Kiểm thử cần bổ sung:** Ảnh xám và RGB ở kích thước 1×1, 3×3, 6×6, 7×7; kiểm tra ảnh nhỏ xử lý theo hành vi đã chọn và ảnh từ 7×7 trở lên vẫn giữ cách tính hiện tại.

### 3. Tác vụ lọc nền cũ có thể ghi đè trạng thái ảnh mới

- **Mức độ:** P2 — Trung bình
- **Vị trí:** [`gui/app.py`](./gui/app.py), hàm `on_apply_filter` và callback `done`
- **Hiện trạng:** Tác vụ nền lọc trên bản sao của `noisy` và `original`. Trong thời gian đó, giao diện vẫn cho phép tải ảnh khác hoặc đổi chế độ ảnh xám. Callback hoàn tất không xác nhận tác vụ còn tương ứng với trạng thái hiện tại trước khi cập nhật `self.result` và panel kết quả.
- **Ảnh hưởng:** Kết quả ảnh cũ có thể xuất hiện cạnh ảnh gốc mới, hoặc bị lưu nhầm. Đổi chế độ xám trong lúc lọc cũng có thể khiến các panel hiển thị ảnh không đồng nhất.
- **Hướng cải thiện:** Ngăn các thao tác làm đổi dữ liệu nguồn trong khi đang lọc, hoặc gắn mã phiên/tác vụ vào lần chạy và chỉ áp dụng callback nếu nó vẫn là tác vụ mới nhất trên cùng ảnh/trạng thái. Đảm bảo trạng thái `busy` được khôi phục nếu không thể khởi chạy worker.
- **Kiểm thử cần bổ sung:** Mô phỏng lọc chậm, tải ảnh khác hoặc đổi chế độ xám trước khi worker hoàn tất; xác nhận callback cũ không ghi đè ảnh mới và không thể lưu kết quả sai.

### 4. Benchmark đồng bộ làm giao diện không phản hồi

- **Mức độ:** P2 — Trung bình
- **Vị trí:** [`gui/app.py`](./gui/app.py), hàm `on_benchmark`
- **Hiện trạng:** Tất cả tổ hợp kernel/mode được chạy tuần tự ngay trong callback Tkinter. `set_busy` cập nhật trạng thái nhưng không chuyển việc tính toán khỏi luồng giao diện.
- **Ảnh hưởng:** Các mode chậm, đặc biệt `naive`, có thể làm cửa sổ đứng trong lúc benchmark chạy; người dùng không thể tương tác hoặc hủy tác vụ.
- **Hướng cải thiện:** Chạy benchmark trong worker nền như luồng áp dụng filter; đưa kết quả và mọi cập nhật widget trở lại luồng Tkinter qua `after`. Bảo đảm trạng thái busy được giải phóng khi hoàn tất hoặc gặp lỗi.
- **Kiểm thử cần bổ sung:** Chạy benchmark trên ảnh/kernels đại diện; xác nhận cửa sổ tiếp tục phản hồi, trạng thái hoàn tất được cập nhật và lỗi không làm giao diện bị kẹt.

### 5. Mật độ nhiễu Salt & Pepper thực tế thấp hơn cấu hình

- **Mức độ:** P3 — Thấp
- **Vị trí:** [`core/noise.py`](./core/noise.py), hàm `add_salt_pepper_noise`
- **Hiện trạng:** Số tọa độ muối và tiêu được lấy bằng cách lấy mẫu có hoàn lại. Tọa độ lặp lại làm giảm số pixel khác nhau bị nhiễu; cùng một pixel cũng có thể bị muối và tiêu ghi đè.
- **Ảnh hưởng:** Tham số `density` không biểu thị đáng tin cậy tỷ lệ pixel bị thay đổi. Ví dụ đã tái hiện: ảnh 100×100, mật độ 50%, seed 0 chỉ đổi khoảng 39,02% pixel.
- **Hướng cải thiện:** Chọn pixel nhiễu duy nhất, phân chia tập pixel thành muối và tiêu theo `salt_vs_pepper`, rồi gán giá trị cho từng nhóm. Làm rõ trong docstring liệu mật độ là tỷ lệ pixel mục tiêu hay xác suất nhiễu độc lập.
- **Kiểm thử cần bổ sung:** Xác nhận số pixel đổi gần tỷ lệ mục tiêu theo quy tắc đã chọn, không có giao nhau giữa nhóm muối/tiêu, hoạt động ở density 100%, và kết quả lặp lại khi dùng cùng seed.

### 6. Callback lỗi của worker có thể không truy cập được ngoại lệ

- **Mức độ:** P3 — Thấp
- **Vị trí:** [`gui/app.py`](./gui/app.py), hàm `on_apply_filter`, khối `except` và callback `fail`
- **Hiện trạng:** Callback `fail()` tham chiếu biến ngoại lệ `e` được khai báo trong `except`. Python xóa biến ngoại lệ sau khi thoát khỏi khối này; callback được lên lịch chạy sau đó có thể gặp `NameError`.
- **Ảnh hưởng:** Khi lọc lỗi, thông báo lỗi dự kiến có thể không xuất hiện; callback lỗi lại thay thế lỗi ban đầu và làm giảm khả năng chẩn đoán.
- **Hướng cải thiện:** Lưu nội dung ngoại lệ vào biến cục bộ bền vững trước khi đăng ký callback, hoặc truyền thông tin lỗi thành đối số mặc định của callback. Giữ nguyên thông báo lỗi gốc.
- **Kiểm thử cần bổ sung:** Ép worker phát sinh ngoại lệ, thực thi callback Tkinter đã lên lịch và xác nhận trạng thái busy được xóa, thông báo lỗi gốc được hiển thị, không phát sinh `NameError`.

## Cập nhật tài liệu

### Số lượng test trong README không khớp

- **Vị trí:** [`README.md`](./README.md)
- **Hiện trạng:** README mô tả 11 unit tests và ví dụ đầu ra `11 passed`; lần chạy hiện tại cho kết quả **13 passed**.
- **Hướng cải thiện:** Cập nhật số lượng test được ghi cứng, hoặc bỏ con số dễ lỗi thời và hướng người đọc chạy lệnh pytest để xem kết quả hiện tại.

## Gợi ý thứ tự xử lý

1. Sửa hành vi padding của `optimized` và xử lý SSIM cho ảnh nhỏ.
2. Ngăn kết quả worker cũ ghi đè trạng thái ảnh mới; xử lý an toàn callback lỗi.
3. Chuyển benchmark khỏi luồng giao diện.
4. Điều chỉnh cách lấy mẫu Salt & Pepper để mật độ đúng với định nghĩa.
5. Cập nhật README và bổ sung test hồi quy cho từng trường hợp.
