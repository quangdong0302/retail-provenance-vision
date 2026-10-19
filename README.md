> Cấu hình: DINOv2 + head dùng chung + prototype trung bình từ 2 ảnh support/SKU. Risk mục tiêu thực nghiệm dưới 5%; xem [giao thức và cách chạy](docs/SELECTED_PROTOCOL.md).

# RPV — Retail Provenance Vision

## Cấu hình và trạng thái ban đầu

Mặc định trong `.env`/`.env.example`: DINOv2 ViT-S/14 đóng băng, ảnh 336 px pad, head dùng chung, màu HS và một prototype trung bình chuẩn hóa từ hai ảnh support/SKU. Energy OOD và ngưỡng selective được hiệu chỉnh sau khi train. Đăng ký SKU không huấn luyện lại classifier.

Kho mã nguồn chưa kèm head đã train, prototype, calibration, cache vector hay kết quả thực nghiệm. `RPV_PROJECTION_CKPT` để trống; `RPV_HEAD_OUT` chỉ vị trí sẽ lưu head mới. Notebook đã bỏ output và số thứ tự thực thi, giữ nguyên code, markdown và comment. Dữ liệu đã chuẩn bị vẫn nằm trong `data/dataset_v2/`.

Restart kernel notebook trước lượt chạy mới. `var/thesis_v4/` và `var/embed_cache/` sẽ được tạo khi train hoặc trích xuất vector.

Nhận diện mặt hàng bằng DINOv2 và học ít mẫu, kiểm tra nhầm lẫn khi đăng ký SKU
mới, kết hợp đối chiếu hồ sơ lô và lưu lịch sử quyết định. Ảnh hỗ trợ nhận diện
SKU; thông tin nguồn gốc phải lấy từ hồ sơ riêng.

## Chạy một lượt model mới — notebook

Mở [hướng dẫn notebook](notebooks/README.md), chọn **Python (build_ML)** và chạy theo thứ tự:

1. Dữ liệu và trực quan.
2. Train head mới.
3. Dựng prototype và hiệu chỉnh.
4. Đánh giá, baseline và báo cáo.
5. Kiểm thử bổ sung nếu cần.
6. Mở demo bằng model vừa train.

Đặt dữ liệu đã chuẩn bị vào `data/dataset_v2/` theo [bố cục dữ liệu](data/README.md).
Notebook và terminal/`run_all` dùng chung `.env`, với đường dẫn tương đối trong dự án.
DINOv2 pretrained tự dùng cache hoặc tải khi thiếu; không cần đóng kèm `.pth`.
Đầu ra theo `.env` nằm trong `var/thesis_v4/` và `var/embed_cache/`.
Mỗi ô notebook có chức năng rõ ràng; không có file cấu hình riêng cho notebook.
Các lệnh CLI và cách triển khai khác xem trong tài liệu bên dưới.

## Chạy từ đầu bằng terminal

Tại thư mục gốc dự án, dùng môi trường `build_ML`:

```powershell
conda activate build_ML
python scripts/run_all.py --train-head
```

Lệnh này tạo head, dựng prototype, hiệu chỉnh và đánh giá. Sau khi train xong, để terminal/API nạp head vừa tạo, đặt biến trong shell hoặc điền cùng đường dẫn vào `.env`:

```powershell
$env:RPV_PROJECTION_CKPT = (Resolve-Path 'var/thesis_v4/projection_head.pt').Path
```

Notebook tự dùng đường dẫn `RPV_HEAD_OUT` cho các bước sau train. Các công cụ tìm kiếm cấu hình và review confidence cần head/cache đã tạo, không phải bước khởi động từ trạng thái trống.

## Đọc tài liệu theo việc cần làm

| Tài liệu | Nội dung |
|---|---|
| [Cài đặt](docs/SETUP.md) | Conda, cấu hình, GPU/CPU và chạy offline |
| [Huấn luyện và đánh giá](docs/TRAINING.md) | Dữ liệu, lệnh chạy, baseline, đầu ra và chạy lại |
| [Chạy ứng dụng](docs/DEPLOYMENT.md) | API, Gradio, Docker và nạp lại model |
| [Đăng ký mặt hàng](docs/ENROLLMENT.md) | Kiểm tra nhầm lẫn, bổ sung ảnh, API/CLI |
| [Kiến trúc](docs/ARCHITECTURE.md) | Các module, luồng model, policy và audit |
| [Khóa luận](docs/THESIS.md) | Giao thức 70/30, metric, thực nghiệm và đóng góp |
| [Sửa theo nhận xét giảng viên](docs/REVIEW_CHANGES.md) | Decision 4 trạng thái, Energy OOD, risk–coverage, nhiễu và đo latency |
| [Đối chiếu yêu cầu giảng viên](docs/PROJECT_ASSESSMENT.md) | Phương pháp, giới hạn và bằng chứng cần thu thập trong lượt chạy mới |
| [Rà soát logic và đề xuất mô hình](docs/MODEL_LOGIC_REVIEW.md) | Các quy tắc nhất quán, kế hoạch cải thiện và cách kiểm chứng |
| [Giao diện demo](docs/DEMO_UI.md) | Luồng sử dụng mới, ảnh desktop/mobile và cách chạy |
| [Pháp lý và nghiệp vụ](docs/LEGAL_BUSINESS.md) | Động lực nghiên cứu tại Việt Nam và quy trình POS |
| [Cơ sở dữ liệu](docs/DATABASE.md) | Sơ đồ và ràng buộc của 7 bảng quan hệ |

## Cấu trúc dự án

```text
data/dataset_v2/  Ảnh và manifest đã chuẩn bị; xem data/README.md
notebooks/       Các bước dữ liệu → train → hiệu chỉnh → đánh giá → demo
rpv/             Model, dữ liệu, nghiệp vụ và đánh giá
api/             FastAPI
app/             Giao diện Gradio
scripts/         Chạy pipeline, thí nghiệm và kiểm tra nhanh
tests/           Kiểm thử với dữ liệu tạm, không phải kết quả khóa luận
docs/            Tài liệu theo chủ đề và đối chiếu nhận xét giảng viên
var/             Tự tạo khi chạy: checkpoint, prototype, báo cáo, DB và cache
requirements.txt Danh sách thư viện cần cài
requirements-reproduction.txt Phiên bản tham khảo để tái lập môi trường
```

`var/` được tạo khi cần. Checkpoint, báo cáo mẫu và kết quả cũ không được đóng
kèm làm bằng chứng accuracy. Kết quả khóa luận phải lấy từ lượt chạy mới trên
dataset đã khóa và ghi rõ giới hạn thực nghiệm.

