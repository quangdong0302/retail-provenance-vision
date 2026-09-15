# Cài đặt và cấu hình

## Môi trường Git Bash + Conda

Chạy từ thư mục có `README.md`. Nếu Git Bash chưa nhận lệnh Conda, trên máy này:

```bash
source /c/Users/user/miniconda3/etc/profile.d/conda.sh
```

Sau đó:

```bash
conda activate build_ML
python -m pip install -r requirements.txt
export MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONIOENCODING=utf-8
```

Chỉ có một `requirements.txt`, dùng cho model, API, UI và Ruff. Bản torch CUDA
đã có được giữ nếu đáp ứng yêu cầu. Môi trường dự án đang dùng Python 3.10;
Docker dùng Python 3.11 và torch CPU.

Trên máy mới chưa có môi trường này, tạo trước bằng
`conda create -n build_ML python=3.10 -y`, rồi kích hoạt và cài requirements.
Đường dẫn `source .../conda.sh` ở trên là ví dụ của máy hiện tại; máy nhận cần
đổi sang thư mục cài Miniconda/Anaconda của mình.

Nếu chưa có `.env`, sao chép mẫu; lệnh này giữ nguyên file đã có:

```bash
test -f .env || cp .env.example .env
```

Biến môi trường trong terminal được ưu tiên hơn `.env`. Khi train mới:

```bash
export RPV_PROJECTION_CKPT=""
```

Khi chạy head đã train, đặt đường dẫn checkpoint tương ứng trước khi mở app.
Không để đường dẫn tới checkpoint đã xóa trong terminal cũ.

Trong notebook hoặc test, sau khi đổi `os.environ`, gọi
`rpv.config.reload_settings()` trước khi tạo encoder/service mới. Các module
đọc cấu hình hiện hành; đối tượng model đã tạo không tự thay checkpoint.
File `.bat` và `.sh` chạy bằng `python` của môi trường đang kích hoạt.

## Các đường dẫn mặc định

| Biến | Giá trị |
|---|---|
| `RPV_DATA_DIR` | `data/dataset_v2` |
| `RPV_HEAD_OUT` | `var/thesis/projection_head.pt` |
| `RPV_BANK` | `var/thesis/prototypes.json` |
| `RPV_CALIB` | `var/thesis/calibration.json` |
| `RPV_REPORTS` | `var/thesis/reports` |
| `RPV_CACHE_DIR` | `var/embed_cache` |
| `RPV_DB` | `var/thesis/rpv.db` |
| `RPV_AUDIT` | `var/thesis/audit.jsonl` |

Cấu hình Python mặc định và `.env.example` dùng cùng bố cục này. File `.env`
có thể đổi các đường dẫn cho một thí nghiệm riêng. Dữ liệu gốc luôn ở `data/`;
đầu ra của model và trạng thái ứng dụng nằm dưới `var/`.

## CPU, GPU và offline

Kiểm tra GPU từ môi trường đã kích hoạt:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`RPV_DEVICE` để trống sẽ tự chọn CUDA khi có; đặt `cpu` để dùng CPU. Với lỗi
OpenMP trên môi trường Conda này, dùng `MKL_THREADING_LAYER=SEQUENTIAL` như trên.
Ghi lại cấu hình này cùng phần cứng khi đo thời gian suy luận.

DINOv2 cần mã backbone và trọng số pretrained. Chạy lần đầu khi có mạng để
cache, hoặc cấu hình đường dẫn có sẵn:

```bash
export RPV_DINOV2_REPO="D:/models/dinov2"
export RPV_DINOV2_WEIGHTS="D:/models/dinov2_vits14_pretrain.pth"
```

Thay các đường dẫn ví dụ bằng vị trí thật. Cache backbone pretrained khác với
checkpoint head học trên dataset của bạn; không cần xóa backbone để train
head lại từ đầu. `RPV_EXTERNAL_ADAPTERS=0` tắt adapter nghiệp vụ bên ngoài,
không thay thế việc chuẩn bị DINOv2 để chạy offline.

Tiếp theo: [huấn luyện và đánh giá](TRAINING.md).
