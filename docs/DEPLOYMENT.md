# Chạy API, giao diện và Docker

Bản khóa luận chưa có đăng nhập/phân quyền cho thao tác ghi danh mục, hồ sơ và
trạng thái lô. Các lệnh và Compose mặc định chỉ mở cổng trên máy chạy
(`127.0.0.1`). Khi triển khai cho nhiều người dùng cần bổ sung xác thực/phân
quyền trước khi mở cổng ra mạng. Gradio dùng `GRADIO_SERVER_NAME=127.0.0.1`
mặc định trên host; Docker đặt `0.0.0.0` bên trong container nhưng chỉ publish
cổng về localhost của host.

## Chạy trên Git Bash

Hoàn tất [pipeline huấn luyện](TRAINING.md) trước khi chạy bộ model có head.
Trong mỗi terminal, kích hoạt môi trường và đặt đúng checkpoint:

```bash
conda activate build_ML
export MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONIOENCODING=utf-8
export RPV_PROJECTION_CKPT="var/thesis/projection_head.pt"
```

Terminal thứ nhất:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal thứ hai:

```bash
python -m app.demo
```

- UI: http://localhost:7860
- API: http://localhost:8000/docs
- Health: http://localhost:8000/health

Giao diện Gradio dùng service trực tiếp nên có thể chạy riêng, không bắt buộc
mở API cùng lúc. Nếu dùng bank frozen không có head, đặt `RPV_PROJECTION_CKPT=""`
và dùng bank/calibration được tạo cho frozen. Không trộn hai bộ artifact.

Kiểm tra health/log để biết encoder, kích thước danh mục và trạng thái
calibration. Nhận diện ảnh không tự tạo hồ sơ lô; nhập nhà cung cấp, lô và
chứng từ phù hợp trước khi thử quầy. Hướng dẫn tính năng mới:
[đăng ký có kiểm tra](ENROLLMENT.md).

`/health` trả HTTP 503 nếu lớp thị giác chưa khởi tạo hoặc bank mới không
nạp được; service khởi tạo được với danh mục rỗng vẫn trả 200 và báo số SKU
bằng 0. API/demo từ chối tiếp tục dùng bank đã bị xóa trong lúc đang chạy.

## Docker với bộ model đã train

Cần Docker Engine và Compose v2; trên Windows dùng Linux containers. Container
có môi trường Python riêng. Dockerfile cài torch CPU và cùng `requirements.txt`.

```bash
ls var/thesis/prototypes.json var/thesis/calibration.json var/thesis/projection_head.pt data/dataset_v2/manifest.json
docker compose -f docker-compose.yml -f docker-compose.trained.yml config
docker compose -f docker-compose.yml -f docker-compose.trained.yml up --build -d api demo
docker compose -f docker-compose.yml -f docker-compose.trained.yml logs --tail 100 api demo
```

Override dùng DINOv2 `vits14`, pad 224 và TTA; mount `var/thesis` vào `/data`
để đọc model và ghi trạng thái, mount dataset chỉ đọc. Trên Linux, user `rpv`
trong container phải có quyền ghi thư mục artifact. Lần đầu DINOv2 có thể cần
mạng để tải trọng số; cache ở `/data/torch`.

Compose chờ API healthy trước khi mở demo. Healthcheck API gọi cổng 8000;
healthcheck demo gọi cổng 7860 của chính demo.

Nếu chỉ chạy `docker compose up --build -d api demo`, hệ thống dùng volume
`rpv-data`; volume mới có danh mục rỗng. Không có bootstrap nạp dữ liệu giả.
Volume Docker là trạng thái riêng: dọn `var/` trên host không xóa volume đó.
Không chạy ứng dụng Conda và Docker cùng ghi vào thư mục artifact của host.

## Đổi model và dừng ứng dụng

Bank có thể được nạp lại theo thời gian sửa file. Checkpoint/calibration mới
cần restart; dừng ứng dụng trước khi thay bộ artifact cho khớp nhau.
API/demo dùng chung khóa file khi đăng ký vào cùng bank: đọc danh mục hiện hành,
kiểm tra rồi ghi nguyên tử nằm trong cùng vùng khóa. Đăng ký có kiểm tra bị từ
chối nếu danh mục đã đổi sau đánh giá; cần đánh giá lại hồ sơ. Không xóa các file
`.lock` trong lúc service chạy. Khóa này dành cho các process cùng filesystem;
không phải giao dịch phân tán giữa bank JSON, SQLite và audit. Nếu ghi lỗi hoặc
máy mất điện giữa các bước, cần đối chiếu bank, hồ sơ đăng ký và audit trước
khi thử xuất bản lại. Dừng service trước khi chạy `calibrate` dựng lại toàn bộ bank.
Sau đăng ký SKU mới, calibration cũ không còn phù hợp thì hệ thống yêu cầu
nhân viên xác nhận. `calibrate` dựng lại bank từ dataset, không dùng để cấp
quyền cho danh mục có SKU vừa đăng ký ngoài dataset.

Git Bash: Ctrl+C trong từng terminal. Docker, dùng đúng cấu hình đã khởi động:

```bash
docker compose -f docker-compose.yml -f docker-compose.trained.yml restart api demo
docker compose -f docker-compose.yml -f docker-compose.trained.yml down
```

## Kiểm tra code mà không train

```bash
python -m unittest discover -s tests -t .
python -m ruff check rpv api app scripts tests
```

Các test dùng dữ liệu tạm. `scripts/smoke_thesis.py` và
`scripts/smoke_enrollment.py` là kiểm tra nhanh có tạo đầu ra trong `var/`,
không phải benchmark khóa luận. Vector backbone lưu tại `var/embed_cache/`
và được giữ khi train lại head; xem `python scripts/cache_embeddings.py`.
