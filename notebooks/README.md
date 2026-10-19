# Tổ chức phần thực nghiệm trong notebook

Các notebook ghi lại các bước từ dữ liệu đến demo, với dữ liệu đặt tại `data/dataset_v2/` theo [bố cục đã chuẩn bị](../data/README.md). `.env` và `.env.example` dùng đường dẫn tương đối, 2-shot, class seed 1 và split seed 401. Cấu hình và đầu ra được dùng chung với terminal. Kernel trên máy hiện tại là **Python (build_ML)**, đã cài các thư viện trong `requirements.txt` và `ipykernel`. Kernel cần được khởi động lại khi thay `.env`; biến môi trường đã có vẫn được ưu tiên như ở CLI.

| Bước | Notebook | Lệnh/chức năng tương ứng ở terminal |
|---|---|---|
| 1 | [Data visualization](01_data_visualization.ipynb) | `python -m rpv.cli dataset-info`; thống kê và xem ảnh |
| 2 | [Train](02_train.ipynb) | `python -m rpv.cli train-head`; epoch/episode mặc định giống run_all |
| 3 | [Calibration](03_calibration.ipynb) | `python -m rpv.cli calibrate` với head vừa train |
| 4 | [Evaluation](04_evaluation.ipynb) | `python -m rpv.cli evaluate`; thêm baseline và kịch bản mô phỏng |
| 5 | [Additional tests](05_additional_tests.ipynb) | Unknown bổ sung, `evaluate --robustness` và `benchmark`; tùy chọn |
| 6 | [Demo](06_demo.ipynb) | `init-db`, `seed-scenarios`, `python -m app.demo` |

Các ô ở notebook 01–04 được thực hiện lần lượt. Với notebook 05, mỗi ô tương ứng một phép thử bổ sung; notebook 06 được chạy từng ô để mở và dừng demo. `notebook_utils.py` giúp gọi CLI, vẽ hình và đọc báo cáo, còn thuật toán huấn luyện, mô hình và cấu hình vẫn dùng chung với terminal.

Lượt huấn luyện cũng có thể được thực hiện từ terminal thay cho notebook 02 bằng lệnh lưu dưới đây:

```powershell
python scripts/run_all.py --encoder dinov2 --train-head
```

Notebook 03–06 phục vụ hiệu chỉnh, đánh giá và trình bày demo sau lượt huấn luyện ở terminal, với cùng head và đường dẫn đầu ra. Notebook 02 tạo head mới và ghi đè `RPV_HEAD_OUT`; khi thay tham số và học lại, cũng thực hiện lại calibration và evaluation. Nếu lượt học bị ngắt, vector trong cache được dùng lại nhưng optimizer bắt đầu từ đầu. DINOv2 pretrained được tải hoặc lấy từ cache; chỉ projection head được học mới.

Đầu ra dùng cùng cấu hình với terminal: `var/thesis/projection_head.pt`, `prototypes.json`, `calibration.json`, `rpv.db`, `reports/` và cache tại `var/embed_cache/`. Các baseline có artifact riêng trong reports để không ghi đè bank và calibration của mô hình đang sử dụng.

Các lệnh chạy demo trực tiếp bằng PowerShell sau khi đã có head:

```powershell
$env:RPV_PROJECTION_CKPT = 'var/thesis/projection_head.pt'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -m app.demo
```

Notebook lấy `RPV_PROJECTION_CKPT` khi biến này đã được cấu hình; nếu để trống thì các bước sau train dùng head tại `RPV_HEAD_OUT`. Checkpoint để trống ở bước bắt đầu train mới. Các bước gọi trực tiếp chức năng của CLI và `run_all`. Unknown chính được tách khỏi train, chọn head và calibration; dữ liệu lô cùng chứng từ trong demo là mô phỏng. Ghi chú mục tiêu risk 1% cần được đối chiếu với kết quả test của đúng cấu hình, không xem là mức đã đạt.