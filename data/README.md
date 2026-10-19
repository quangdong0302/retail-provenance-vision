# Đặt dữ liệu vào dự án

Chép **nội dung đã chuẩn bị** từ bộ PackEat của bạn vào `data/dataset_v2/`.
Không chép lồng thêm một thư mục `data` bên trong. Không cần tải/prepare lại.
`.env` đã dùng đường dẫn tương đối `RPV_DATA_DIR=data/dataset_v2`.

```text
data/
├── README.md
└── dataset_v2/
    ├── known/
    │   └── <tên lớp>/...jpg
    ├── unknown/
    │   └── <tên lớp>/...jpg
    ├── manifest.json
    ├── metadata.csv
    ├── model_profile.json
    ├── splits_model_k2.json
    ├── _extra_unknown/          # thêm nếu chạy unknown bổ sung
    │   └── <tên lớp>/...jpg
    └── hard_pairs.csv          # thêm nếu kiểm thử các cặp dễ nhầm
```

| Thành phần | Dùng để làm gì | Cần chép? |
|---|---|---|
| `known/`, `unknown/`, `manifest.json` | Ảnh và manifest hợp lệ để train/calibrate/evaluate bằng CLI | Bắt buộc |
| `metadata.csv` | Trực quan, thống kê phiên chụp, nhãn có/không túi và unknown bổ sung | Bắt buộc cho bộ notebook này |
| `model_profile.json`, `splits_model_k2.json` | Đối chiếu cấu hình 2-shot và split đã chốt | Nên chép |
| `_extra_unknown/` | Đánh giá OOD bổ sung ở notebook 05 | Khi cần phép thử này |
| `hard_pairs.csv` | Kiểm chứng lời khai sai giữa các giống dễ nhầm | Khi cần phép thử này |
| `groups.csv`, `classes.csv`, `label_mapping_vi.csv`, `quality_report.json` | Thông tin nhóm, nhãn và chất lượng để làm báo cáo | Nên chép để lưu hồ sơ dữ liệu |
| Các file `*.txt`, báo cáo `*.md`, `model_split_counts.csv` | Danh sách split/báo cáo để tham khảo | Tùy chọn; model tự dựng split theo manifest và seed |

Không cần chép `_archive/`, `_excluded/`, `_preview/`, `_sources/` hoặc `_preparation_log.txt`.
Checkpoint, prototype, calibration, DB và embedding cache là đầu ra trong `var/`, không đặt vào `data/`.

Giữ nguyên tên lớp, tên ảnh và cấu trúc bên trong ba pool. Các tên lớp tiếng Việt hiện tại
đã nằm trong manifest; đổi tên sẽ làm dataset không khớp fingerprint. Tên file notebook
là tiếng Anh và không ảnh hưởng tên nhãn dữ liệu.

Bộ chính hiện tại có **17 known / 9.924 ảnh**, **7 unknown / 4.200 ảnh**;
unknown bổ sung có **38 lớp / 21.202 ảnh**. Nếu chỉ chép bộ chính, vẫn dùng metadata gốc;
chỉ bỏ qua ô preview extra ở notebook 01 và ô extra unknown ở notebook 05.

Sau khi chép, tại thư mục gốc dự án:

```powershell
python -m rpv.cli dataset-info
```

Lệnh sẽ kiểm tra manifest, hash ảnh và nhóm độc lập. Sau đó chạy notebook 01–06 hoặc:

```powershell
python scripts/run_all.py --encoder dinov2 --train-head
```
