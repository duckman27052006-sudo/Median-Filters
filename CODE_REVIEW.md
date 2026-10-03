# Báo cáo rà soát mã nguồn — Median Filter App

Phạm vi: mã nguồn, giao diện, xử lý ảnh, benchmark, kiểm thử và tài liệu dự án.

## Tóm tắt

Lần rà soát mới nhất ghi nhận **3 điểm cần cải thiện trong code**:

- **P2 — Trung bình:** Benchmark không phân biệt lỗi cấu hình với lỗi của một phép đo.
- **P3 — Thấp:** API lọc chưa từ chối ảnh rỗng; PSNR chưa kiểm tra `max_val`.
- **Tài liệu:** Báo cáo cũ không còn phản ánh các sửa đổi và trạng thái code mới nhất.

Kiểm thử hiện tại: **45 passed**. Một số trường hợp trong báo cáo này chưa có test hồi quy.

## Các điểm cần cải thiện

### 1. Benchmark biến mọi lỗi thành `NaN`, che lỗi cấu hình

- **Mức độ:** P2 — Trung bình
- **Vị trí:** [`core/filters.py`](./core/filters.py), `benchmark_filters`
- **Hiện trạng:** Hàm bắt `Exception` cho mọi lần chạy một mode/kernel rồi ghi `NaN`, không trả hoặc ghi log nguyên nhân. Ví dụ `modes=["misspelled"]` trả `{"misspelled": [nan]}` mà không báo mode không hợp lệ.
- **Ảnh hưởng:** Sai tên mode, padding sai, hoặc lỗi runtime bị trình bày giống một phép đo thất bại thông thường. Người dùng CLI/GUI khó biết cấu hình sai hay filter thực sự gặp sự cố; biểu đồ có thể chứa điểm thiếu mà không giải thích.
- **Hướng cải thiện:** Kiểm tra đầu vào benchmark (mode, kernel, repeat, padding) trước khi đo và báo lỗi cấu hình rõ ràng. Với lỗi runtime theo từng ô đo, có thể tiếp tục benchmark nhưng cần giữ thông tin lỗi trong kết quả/log/UI thay vì chỉ trả `NaN`.
- **Kiểm thử cần bổ sung:** Mode không tồn tại phải tạo lỗi cấu hình có ý nghĩa; lỗi runtime của một mode hợp lệ phải được báo rõ và không làm mất kết quả các mode khác.

### 2. Bộ lọc chấp nhận ảnh rỗng ở bước kiểm tra ban đầu

- **Mức độ:** P3 — Thấp
- **Vị trí:** [`core/filters.py`](./core/filters.py), `_validate`
- **Hiện trạng:** `_validate` kiểm tra kiểu, dtype, số chiều và kernel nhưng không xác nhận chiều cao/rộng khác 0. Ảnh `(0, 8)` hoặc `(8, 0)` tiếp tục vào xử lý rồi gây lỗi từ `np.pad`/OpenCV.
- **Ảnh hưởng:** Người gọi nhận lỗi cấp thấp, phụ thuộc implementation, thay vì lỗi đầu vào mô tả ảnh không hợp lệ. Các API xử lý ảnh cũng không có hợp đồng rõ cho mảng rỗng.
- **Hướng cải thiện:** Từ chối ảnh có kích thước bất kỳ bằng 0 trong `_validate` bằng thông báo `ValueError` nêu yêu cầu ảnh có chiều cao và chiều rộng dương.
- **Kiểm thử cần bổ sung:** Kiểm tra ảnh grayscale và RGB có một chiều bằng 0 đều bị từ chối ngay ở validation; ảnh kích thước hợp lệ tiếp tục cho kết quả cũ.

### 3. `calculate_psnr` không xác thực `max_val`

- **Mức độ:** P3 — Thấp
- **Vị trí:** [`core/metrics.py`](./core/metrics.py), `calculate_psnr`
- **Hiện trạng:** Hàm dùng `max_val` trong biểu thức PSNR mà không kiểm tra giá trị dương và hữu hạn. Với ảnh khác nhau và `max_val=0`, hàm phát cảnh báo chia cho 0 rồi trả `-inf`; giá trị âm hoặc NaN cũng cho kết quả không phù hợp với ý nghĩa của peak signal value.
- **Ảnh hưởng:** Chỉ số có thể sai hoặc phát warning thay vì từ chối tham số cấu hình không hợp lệ.
- **Hướng cải thiện:** Yêu cầu `max_val` hữu hạn và lớn hơn 0; nêu rõ lỗi bằng `ValueError` trước khi tính.
- **Kiểm thử cần bổ sung:** Tham số `0`, âm, `NaN`, `+inf` và `-inf` phải bị từ chối; giá trị dương hữu hạn và trường hợp hai ảnh giống hệt vẫn giữ hành vi hợp lệ.

## Các điểm từ lần rà soát trước đã được khắc phục

Các vấn đề sau đã được sửa trong code hiện tại:

- Quy ước padding `reflect` của median, mean và Gaussian được chuẩn hóa theo `np.pad(mode="reflect")`/`cv2.BORDER_REFLECT_101`.
- `naive` dùng full-sort vector hóa, `quickselect` dùng `partition`; hai mode có implementation riêng và cùng cho kết quả chính xác.
- CLI benchmark in kết quả theo chỉ số, kể cả khi danh sách kernel có phần tử lặp.
- Thông báo PSNR phân biệt được kết quả giảm chất lượng và định dạng chênh lệch có dấu đúng.
- Khởi động chỉ hiện “Sẵn sàng” nếu nạp ảnh mẫu thành công.
- Các phát hiện cũ về padding của `optimized`, SSIM ảnh nhỏ, worker lỗi/stale, benchmark GUI và mật độ Salt & Pepper đã được sửa.

## Cập nhật tài liệu

### Báo cáo rà soát trước đó đã lỗi thời

- **Vị trí:** [`CODE_REVIEW.md`](./CODE_REVIEW.md)
- **Hiện trạng:** Báo cáo trước còn liệt kê 5 điểm đã được sửa là vấn đề hiện tại, ghi kết quả 40 test và chưa nêu ba vấn đề xác nhận được ở lần rà soát này.
- **Hướng cải thiện:** Nội dung file này đã được cập nhật để tách các vấn đề đã khắc phục khỏi các điểm còn mở và ghi kết quả kiểm thử mới nhất.

README hiện ghi **45 passed**, khớp với lần chạy test mới nhất; bảng mô tả thuật toán đã phản ánh hai đường `sort` và `partition`.

## Kiểm chứng

- Lệnh: `python -m pytest tests -q`
- Kết quả: **45 passed**
- Các trường hợp mode benchmark sai, ảnh rỗng và `max_val` không hợp lệ đã được tái hiện qua kiểm tra trực tiếp, nhưng chưa có test hồi quy trong suite.

## Thứ tự đề xuất

1. Xác thực tham số benchmark và không làm mất nguyên nhân của phép đo thất bại.
2. Từ chối ảnh rỗng ngay trong `_validate`.
3. Xác thực `max_val` của PSNR.
4. Thêm test hồi quy cho ba trường hợp trên.
