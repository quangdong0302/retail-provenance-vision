# RPV — Retail Provenance Vision

Nhận diện mặt hàng bằng DINOv2 và học ít mẫu, kiểm tra nhầm lẫn khi đăng ký SKU
mới, kết hợp đối chiếu hồ sơ lô và lưu lịch sử quyết định. Ảnh hỗ trợ nhận diện
SKU; thông tin nguồn gốc phải lấy từ hồ sơ riêng.

## Chạy lại từ đầu — Git Bash

Tại thư mục dự án, với dữ liệu thật đã chuẩn bị ở `data/dataset_v2`:

```bash
conda activate build_ML
export MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONIOENCODING=utf-8
export RPV_PROJECTION_CKPT=""
python -m pip install -r requirements.txt
python scripts/run_all.py --data-dir data/dataset_v2 --encoder dinov2 --train-head
```

Không chạy lại `prepare-data` khi dataset này đã có. Pipeline tạo embedding,
train head mới, dựng prototype, hiệu chỉnh và xuất kết quả đánh giá mới.
Sau khi hoàn tất, mở ứng dụng theo [hướng dẫn chạy](docs/DEPLOYMENT.md).

## Đọc tài liệu theo việc cần làm

| Tài liệu | Nội dung |
|---|---|
| [Cài đặt](docs/SETUP.md) | Conda, cấu hình, GPU/CPU và chạy offline |
| [Huấn luyện và đánh giá](docs/TRAINING.md) | Dữ liệu, lệnh chạy, baseline, đầu ra và chạy lại |
| [Chạy ứng dụng](docs/DEPLOYMENT.md) | API, Gradio, Docker và nạp lại model |
| [Đăng ký mặt hàng](docs/ENROLLMENT.md) | Kiểm tra nhầm lẫn, bổ sung ảnh, API/CLI |
| [Kiến trúc](docs/ARCHITECTURE.md) | Các module, luồng model, policy và audit |
| [Khóa luận](docs/THESIS.md) | Giao thức 70/30, metric, thực nghiệm và đóng góp |

## Cấu trúc dự án

```text
data/dataset_v2/  Ảnh thật: known, unknown và manifest.json
rpv/             Model, dữ liệu, nghiệp vụ và đánh giá
api/             FastAPI
app/             Giao diện Gradio
scripts/         Chạy pipeline, thí nghiệm và kiểm tra nhanh
tests/           Kiểm thử với dữ liệu tạm, không phải kết quả khóa luận
docs/            Sáu tài liệu theo chủ đề
var/             Đầu ra tự sinh khi chạy; không phải dữ liệu huấn luyện gốc
requirements.txt Danh sách thư viện duy nhất
```

`var/` được tạo khi cần. Checkpoint, báo cáo mẫu và kết quả cũ không được đóng
kèm làm bằng chứng accuracy. Kết quả khóa luận phải lấy từ lượt chạy mới trên
dataset đã khóa và ghi rõ giới hạn thực nghiệm.
