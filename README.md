> Mô hình sử dụng DINOv2 kết hợp head dùng chung và mẫu đại diện tạo từ 2 ảnh cho mỗi SKU. Kết quả đánh giá gồm khả năng nhận diện và mức độ tự xử lý ở mục tiêu risk dưới 5%. Cách chia dữ liệu và các điều kiện thực nghiệm được ghi trong [giao thức](docs/SELECTED_PROTOCOL.md).

# RPV — Retail Provenance Vision

## Cấu hình và đầu ra của mô hình

Cấu hình sử dụng DINOv2 ViT-S/14 đóng băng, ảnh 336 px có đệm giữ tỷ lệ, head dùng chung và đặc trưng màu HS. Mỗi SKU được biểu diễn bằng trung bình chuẩn hóa của hai ảnh support. Sau khi học head, dựng danh mục và hiệu chỉnh ngưỡng Energy OOD cùng ngưỡng selective. Việc thêm SKU chỉ bổ sung mẫu đại diện, không học lại toàn bộ classifier.

Dữ liệu đã chuẩn bị được lưu tại `data/dataset_v2/`. Checkpoint, prototype, calibration, cache vector và báo cáo là đầu ra của từng lượt chạy. Ở bước huấn luyện từ đầu, `RPV_PROJECTION_CKPT` có thể để trống; `RPV_HEAD_OUT` là vị trí lưu head mới. Các ô mã, phần thuyết minh và comment trong notebook được giữ nguyên; kết quả hiển thị cùng số thứ tự thực thi được xóa ở bản khởi tạo để phân biệt với kết quả của lượt chạy sau.

Mỗi lượt chạy mới bắt đầu bằng kernel đã nạp lại cấu hình. Các thư mục `var/thesis_v4/` và `var/embed_cache/` được tạo khi huấn luyện hoặc trích xuất vector.

Phần thị giác trong dự án phục vụ nhận diện SKU và kiểm tra nhầm lẫn khi đăng ký mặt hàng mới. Kết quả được đối chiếu với hồ sơ lô; căn cứ và quyết định xử lý được lưu để truy vết. Thông tin nguồn gốc đến từ hồ sơ riêng, không được suy ra chỉ từ ảnh.

## Tổ chức các notebook

Các notebook thực nghiệm sử dụng kernel **Python (build_ML)**. Nội dung của từng notebook được ghi tại [notebooks/README.md](notebooks/README.md), theo thứ tự:

1. Thống kê và xem dữ liệu.
2. Huấn luyện head.
3. Dựng prototype và hiệu chỉnh ngưỡng.
4. Đánh giá, so sánh baseline và lưu báo cáo.
5. Thực hiện các phép kiểm thử bổ sung.
6. Trình bày demo bằng mô hình vừa huấn luyện.

Notebook, terminal và `run_all` dùng chung `.env`; đường dẫn tương đối đều tính từ thư mục dự án. Dữ liệu được bố trí theo [cấu trúc dữ liệu](data/README.md). DINOv2 sử dụng trọng số có sẵn trong cache hoặc tải khi thiếu, nên tệp `.pth` không bắt buộc phải nằm trong kho mã. Đầu ra được lưu tại `var/thesis_v4/` và `var/embed_cache/`. Các ô notebook sử dụng cùng cấu hình với terminal; phần lệnh và tài liệu liên quan được liệt kê bên dưới.

## Các lệnh khi chạy từ đầu

Ngoài notebook, có thể thực hiện toàn bộ lượt huấn luyện trong môi trường `build_ML`, từ thư mục gốc dự án:

```powershell
conda activate build_ML
python scripts/run_all.py --train-head
```

Lệnh trên tạo head, dựng prototype, hiệu chỉnh và đánh giá. Sau khi huấn luyện, gán đường dẫn của head mới cho `RPV_PROJECTION_CKPT` trong terminal hoặc `.env` để API và ứng dụng dùng đúng checkpoint:

```powershell
$env:RPV_PROJECTION_CKPT = (Resolve-Path 'var/thesis_v4/projection_head.pt').Path
```

Trong notebook, các bước sau huấn luyện lấy head tại `RPV_HEAD_OUT`. Những công cụ tìm kiếm cấu hình và so sánh confidence sử dụng head cùng cache đã tạo ở các bước trước.

## Tài liệu tham khảo trong dự án

| Tài liệu | Nội dung |
|---|---|
| [Cài đặt](docs/SETUP.md) | Conda, cấu hình, GPU/CPU và chạy offline |
| [Huấn luyện và đánh giá](docs/TRAINING.md) | Dữ liệu, lệnh chạy, baseline, đầu ra và chạy lại |
| [Chạy ứng dụng](docs/DEPLOYMENT.md) | API, Gradio, Docker và nạp lại model |
| [Đăng ký mặt hàng](docs/ENROLLMENT.md) | Kiểm tra nhầm lẫn, bổ sung ảnh, API/CLI |
| [Kiến trúc](docs/ARCHITECTURE.md) | Các module, luồng model, policy và audit |
| [Giao thức thực nghiệm](docs/THESIS.md) | Chia lớp 70/30, metric, đối chứng và đánh giá |
| [Các yêu cầu kỹ thuật](docs/REVIEW_CHANGES.md) | Decision 4 trạng thái, Energy OOD, risk–coverage, nhiễu và đo latency |
| [Tiến độ và giới hạn](docs/PROJECT_ASSESSMENT.md) | Phương pháp, giới hạn và bằng chứng cần thu thập trong lượt chạy mới |
| [Rà soát logic và đề xuất mô hình](docs/MODEL_LOGIC_REVIEW.md) | Các quy tắc nhất quán, kế hoạch cải thiện và cách kiểm chứng |
| [Giao diện demo](docs/DEMO_UI.md) | Luồng sử dụng mới, ảnh desktop/mobile và cách chạy |
| [Pháp lý và nghiệp vụ](docs/LEGAL_BUSINESS.md) | Phạm vi áp dụng tại Việt Nam và quy trình POS |
| [Cơ sở dữ liệu](docs/DATABASE.md) | Sơ đồ và ràng buộc của 7 bảng quan hệ |

## Cấu trúc dự án

```text
data/dataset_v2/  Ảnh và manifest đã chuẩn bị; xem data/README.md
notebooks/       Các bước dữ liệu → train → hiệu chỉnh → đánh giá → demo
rpv/             Model, dữ liệu, nghiệp vụ và đánh giá
api/             FastAPI
app/             Giao diện Gradio
scripts/         Chạy pipeline, thí nghiệm và kiểm tra nhanh
tests/           Kiểm thử với dữ liệu tạm, tách biệt với kết quả thực nghiệm
docs/            Tài liệu theo chủ đề và đối chiếu yêu cầu kỹ thuật
var/             Tự tạo khi chạy: checkpoint, prototype, báo cáo, DB và cache
requirements.txt Danh sách thư viện cần cài
requirements-reproduction.txt Phiên bản tham khảo để tái lập môi trường
```

Kết quả thực nghiệm được lưu trong `var/`; từng báo cáo gắn với dataset, cấu hình cùng giao thức đã dùng. Checkpoint hoặc kết quả cũ chỉ được dùng làm đối chứng khi xác định được nguồn gốc và điều kiện đánh giá. Kết luận về hiệu quả dựa trên các lượt chạy có dữ liệu đã khóa, kèm giới hạn thực nghiệm.

