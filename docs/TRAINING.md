# Dữ liệu, huấn luyện và đánh giá

Phiên bản hiện tại dùng head NoAux 336px, conformal relative, nearest-ratio
và risk mục tiêu 2% trên known-calibration cùng extra-development.
Xem [SELECTED_PROTOCOL](SELECTED_PROTOCOL.md) để chạy cấu hình đã chọn.
Trainer `episodic` và các cấu hình trước giữ lại cho đối chứng.
Cách chạy stress test, hard negative, benchmark POS và cách đọc risk/coverage:
[REVIEW_CHANGES](REVIEW_CHANGES.md).

## 1. Dữ liệu dùng cho dự án

```text
data/dataset_v2/
├── known/<class>/<source_folder>/*.jpg
├── unknown/<class>/<source_folder>/*.jpg
└── manifest.json
```

Dataset hiện có 104.442 ảnh, 104 lớp known và 45 lớp unknown. 70/30 là tỷ lệ
lớp, không phải tỷ lệ ảnh. Train/validation/support/calibration/test được
quản lý bằng manifest, không cần thêm thư mục dataset khác để chạy model.
Đây là thống kê dữ liệu, không phải kết quả accuracy.

Nguồn ảnh trên máy: `D:/Documents/Back/images`. Khi nhập bộ dữ liệu mới, chọn
thư mục đích mới; không chạy lại lệnh này để ghi đè dataset hiện có:

```bash
python -m rpv.cli prepare-data --source "D:/Documents/Back/images" --destination data/dataset_v3 --known-ratio 0.7 --seed 1
```

Sau đó đổi `RPV_DATA_DIR` và train/hiệu chỉnh theo bộ mới. Nếu có nhóm vật thể
hoặc phiên chụp, thêm `--groups-csv groups.csv` với cột `path,group_id`.
Quy tắc chia và giới hạn khi chưa có metadata: [giao thức khóa luận](THESIS.md).

## 2. Chạy toàn bộ pipeline từ đầu

Sau bước [cài đặt](SETUP.md), trong Git Bash:

```bash
export RPV_PROJECTION_CKPT=""
python scripts/run_all.py --data-dir data/dataset_v2 --encoder dinov2 --train-head
```

Runner kiểm tra dataset, tạo DB rỗng nếu chưa có, warm-cache, train head,
calibrate, evaluate, chạy unit test rồi kiểm tra audit. `--train-head` chủ
động bỏ head cũ trước bước lấy embedding và chỉ dùng head mới sau khi train
thành công. Backbone DINOv2 vẫn frozen. Các tham số head mặc định xem bằng
`python -m rpv.cli train-head --help`.

Không kết hợp `--train-head` với `--encoder classical`; runner từ chối tổ hợp
này trước khi tính embedding. Đường dẫn tương đối của runner được hiểu theo
thư mục gốc dự án, kể cả khi gọi script từ thư mục khác.

Runner không tự xóa lịch sử đã có. Khi cần một lần reset hoàn toàn, dừng
API/demo và dọn đầu ra như mục 5 trước khi chạy. Unit test dùng ảnh tạm riêng;
không có lệnh sinh dataset giả vào `data/`.

## 3. Chạy từng bước và so sánh baseline

DINOv2 frozen, chưa có head:

```bash
export RPV_PROJECTION_CKPT=""
python -m rpv.cli warm-cache
python -m rpv.cli calibrate
python -m rpv.cli evaluate
```

Train head, hiệu chỉnh và đánh giá trên cùng dataset/split:

```bash
export RPV_PROJECTION_CKPT=""
python -m rpv.cli train-head --epochs 20 --out-dim 384
export RPV_PROJECTION_CKPT="var/thesis/projection_head.pt"
python -m rpv.cli calibrate
python -m rpv.cli evaluate
```

Mỗi `evaluate` lưu báo cáo riêng. `calibrate` cập nhật bank/calibration đang
dùng, do đó chỉ chạy khi ứng dụng đã dừng. Ngưỡng không được fit bằng test.
Sau thay checkpoint, cần tạo lại bank/calibration đúng phiên bản.

So sánh các baseline frozen nhiều seed:

```bash
python scripts/thesis_experiments.py --data-dir data/dataset_v2 --encoders classical dinov2 --seeds 1 2 3
```

Script này không train head. Mean/std đo biến thiên theo support seed trên
cùng test; không phải khoảng tin cậy cho quần thể người dùng.

## 4. Kết quả nằm ở đâu?

| Đường dẫn | Nội dung tự sinh |
|---|---|
| `var/embed_cache/` | Embedding tính từ ảnh, có thể tái tạo |
| `var/thesis/projection_head.pt` | Head vừa train |
| `var/thesis/projection_head.training.json` | Lịch sử train và checkpoint được chọn |
| `var/thesis/projection_head.splits.json` | Split tương ứng với head |
| `var/thesis/prototypes.json`, `calibration.json` | Danh mục thị giác và ngưỡng đang dùng |
| `var/thesis/reports/<run>/` | JSON/Markdown, predictions JSONL và artifact của mỗi lượt đánh giá |
| `var/thesis/experiments/` | Các lượt baseline và bảng tổng hợp nhiều seed |
| `var/thesis/smoke/`, `enrollment_smoke/` | Đầu ra kiểm tra nhanh, không dùng làm benchmark |
| `var/thesis/rpv.db`, `audit.jsonl`, `audit.jsonl.lock` | Trạng thái ứng dụng và lịch sử quyết định |
| `var/thesis/enrollment/`, `bank_backups/` | Hồ sơ đăng ký, ảnh upload và bản sao bank |

`--scenarios` chỉ dùng khi cần thử policy bằng hồ sơ lô mô phỏng trong DB
riêng của run. Không coi dữ liệu kịch bản là hồ sơ truy xuất thật.

## 5. Phân biệt reset model và reset toàn bộ ứng dụng

- Khi train lại head, giữ `var/embed_cache/` để dùng lại vector backbone.
  Có thể tạo head và bank/calibration ở đường dẫn mới để giữ bản đối chứng;
  giữ `data/dataset_v2` cùng `manifest.json` để so sánh trên cùng split.
- Reset toàn bộ: bỏ thêm DB, audit và hồ sơ đăng ký trong `var/thesis`. Các
  ảnh nhập qua ứng dụng thuộc hồ sơ đăng ký; lưu ra ngoài nếu cần dùng lại.
- Dừng mọi process API/demo/train trước khi dọn. File `.lock` đi cùng audit;
  không bỏ riêng file này khi có writer đang chạy.

Khi không cần giữ trạng thái cũ, `var/` có thể để trống; chương trình tự tạo
thư mục khi chạy. Không xóa `data/`, `.env`, mã nguồn hoặc trọng số backbone
pretrained chỉ để reset kết quả. Bản sao lưu nằm ngoài dự án không được runtime
tự nạp; muốn khôi phục cần chủ động đưa đúng bộ artifact trở lại.

Tiếp theo: [chạy ứng dụng](DEPLOYMENT.md) hoặc [phân tích cho khóa luận](THESIS.md).
