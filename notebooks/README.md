# Notebook dùng chung cấu hình với terminal

Chép dữ liệu theo [data/README.md](../data/README.md) vào `data/dataset_v2/`.
`.env` và `.env.example` dùng đường dẫn tương đối, 2-shot, class seed 1, split seed 401.
Không có file cấu hình riêng hay thư mục lượt chạy riêng cho notebook.
Chọn kernel đã cài `requirements.txt` và `ipykernel` (máy hiện tại: **Python (build_ML)**).
Restart Kernel sau khi thay đổi `.env`; biến môi trường đã có ưu tiên hơn file `.env` như CLI.

| Bước | Notebook | Lệnh/chức năng tương ứng ở terminal |
|---|---|---|
| 1 | [Data visualization](01_data_visualization.ipynb) | `python -m rpv.cli dataset-info`; thống kê và xem ảnh |
| 2 | [Train](02_train.ipynb) | `python -m rpv.cli train-head`; epoch/episode mặc định giống run_all |
| 3 | [Calibration](03_calibration.ipynb) | `python -m rpv.cli calibrate` với head vừa train |
| 4 | [Evaluation](04_evaluation.ipynb) | `python -m rpv.cli evaluate`; thêm baseline và kịch bản mô phỏng |
| 5 | [Additional tests](05_additional_tests.ipynb) | Unknown bổ sung, `evaluate --robustness` và `benchmark`; tùy chọn |
| 6 | [Demo](06_demo.ipynb) | `init-db`, `seed-scenarios`, `python -m app.demo` |

Notebook 01–04 có thể Run All. Notebook 05 chọn các phép thử cần thiết;
notebook 06 chạy từng ô để mở và dừng demo. `notebook_utils.py` chỉ hỗ trợ gọi CLI,
vẽ hình và đọc báo cáo; dùng lại model/config hiện có, không tạo thuật toán train khác.

Có thể train bằng terminal thay cho notebook 02:

```powershell
python scripts/run_all.py --encoder dinov2 --train-head
```

Sau đó mở notebook 03–06 để hiệu chỉnh/đánh giá/demo bằng cùng head và đường dẫn đầu ra.
Notebook 02 train mới và ghi đè `RPV_HEAD_OUT`; chạy lại ô này sau khi đổi tham số
thì cần calibrate/evaluate lại. Nếu bị ngắt, cache được dùng lại, optimizer train từ đầu.
DINOv2 pretrained tự tải/cache; chỉ projection head được học mới.

Đầu ra dùng cùng cấu hình terminal: `var/thesis/projection_head.pt`, `prototypes.json`,
`calibration.json`, `rpv.db`, `reports/`; cache tại `var/embed_cache/`.
Baseline có artifact riêng trong thư mục reports để không ghi đè bank/calibration vận hành.

Để mở demo trực tiếp bằng PowerShell sau khi train:

```powershell
$env:RPV_PROJECTION_CKPT = 'var/thesis/projection_head.pt'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -m app.demo
```

Notebook tự chọn `RPV_PROJECTION_CKPT` nếu đã cấu hình; khi để trống, dùng head tại
`RPV_HEAD_OUT` cho các bước sau train. Để checkpoint trống khi bắt đầu train mới.
CLI/run_all giữ nguyên code. Dữ liệu unknown không dùng train/chọn head/calibrate;
hồ sơ lô/chứng từ demo là mô phỏng. Risk mục tiêu 1% phải đối chiếu số liệu test thực tế.
