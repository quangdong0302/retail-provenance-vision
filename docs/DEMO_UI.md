# Giao diện demo RPV

Demo dùng Gradio và service hiện có. Màn hình chính ưu tiên thao tác ở quầy:
chụp/tải ảnh → chọn SKU/lô → kiểm tra → đọc quyết định và lý do.

![Quầy kiểm tra trên desktop](ui/desktop.png)

## Các màn hình

| Tab | Công việc |
|---|---|
| Quầy kiểm tra | Đối chiếu ảnh, SKU và lô; xem quyết định, hướng xử lý và lý do |
| Đăng ký mặt hàng | Tạo hồ sơ, thêm ảnh theo nhóm/vai trò, kiểm tra nhầm lẫn và đăng ký |
| Nhập hồ sơ lô | Nhập mặt hàng, nhà cung cấp, nguồn gốc, thời hạn và bản ghi chứng từ |
| Truy xuất nguồn gốc | Tra hồ sơ, chứng từ, sự kiện và dữ liệu EPCIS; quản lý trạng thái lô |
| Nhật ký kiểm toán | Kiểm tra hash chain và các quyết định gần nhất |
| Hướng dẫn demo | Trình tự chuẩn bị và các tình huống trình bày trước hội đồng |

Số liệu trên các thẻ tổng quan được đọc từ bank và DB thật. Chưa có danh mục/hiệu chỉnh
thì hiển thị hướng dẫn tương ứng. Kết quả cũ được xóa khi đổi đầu vào kiểm tra.
Không hiển thị khoảng cách hay p-value như xác suất nhận diện đúng.

Lô được lọc theo SKU đã chọn; bỏ trống lô thì service chọn từ hồ sơ hiện có.
Ảnh nhận diện SKU không chứng minh vật thể thuộc lô đã chọn.
Sau khi đăng ký mặt hàng hoặc nhập lô, các danh sách ở quầy được cập nhật.

JSON, giá tùy chọn và các thao tác quản lý ít dùng được thu gọn.
Chứng từ mới mặc định **chưa xác minh**; form demo lưu bản ghi khai báo,
chưa có quy trình tải và kiểm chứng tệp chứng từ gốc.

## Chạy trên PowerShell

Trong thư mục dự án, dùng môi trường đã cài dependencies:

```powershell
conda activate build_ML
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:PYTHONIOENCODING = 'utf-8'
python -m app.demo
```

Mở http://127.0.0.1:7860. Có thể đổi cổng qua `GRADIO_SERVER_PORT`.
Nếu dùng model có head, đặt đúng `RPV_PROJECTION_CKPT` và bank/calibration tương ứng
như hướng dẫn [triển khai](DEPLOYMENT.md). Demo có thể chạy riêng với service trực tiếp.
Nếu bank vận hành chưa được tạo, giao diện hiển thị danh mục trống; không tự dùng
artifact smoke làm model chính.

## Hiện thực và kiểm chứng

- `app/demo.py`: backend callbacks, validation giá/ngày trước ghi và điểm khởi chạy.
- `app/ui.py`: bố cục, component/event và tương thích Gradio 4/5/6.
- `app/presentation.py`: CSS, SVG nội bộ và render HTML có escape giá trị dữ liệu.
- `app/enrollment.py`: giao diện đăng ký dùng cùng EnrollmentService.

Kiểm thử trong môi trường Gradio **6.26.0**: Ruff đạt, bộ unittest có **171 test,
170 đạt và 1 bỏ qua**. Chromium đã kiểm tra các tab, thiếu ảnh, làm mới và audit;
desktop 1440px và mobile 390px không tràn ngang tại quầy, không có lỗi JavaScript.
Đã thử luồng nhập ba ảnh qua DINOv2, tạo lô, kiểm tra ngày không hợp lệ,
đối chiếu cần nhân viên khi thiếu calibration, xem hồ sơ và chặn sau thu hồi lô.
Gradio 4/5 được giữ cách truyền CSS/theme cũ; chưa chạy browser riêng trên hai phiên bản đó.

Ảnh chụp dùng DB/bank/audit riêng trong `var/ui_preview`, không ghi vào hồ sơ vận hành.
Trong ảnh ban đầu danh mục còn trống. Đây là kiểm chứng giao diện; đánh giá chất lượng
model và đối chiếu PDF nằm trong [báo cáo dự án](PROJECT_ASSESSMENT.md).

Ảnh mobile: [mở ảnh đầy đủ](ui/mobile.png).
Ảnh kết quả: [cần nhân viên xác nhận](ui/result.png), [chặn lô thu hồi](ui/recalled.png).
