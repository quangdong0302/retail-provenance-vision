# Bố cục dữ liệu

Dữ liệu PackEat sau khi chuẩn bị được lưu trực tiếp trong `data/dataset_v2/`. Các thư mục ảnh và tệp metadata nằm ngay tại đây, không lồng thêm thư mục `data`. Bộ dữ liệu này đã qua bước chuẩn bị nên được dùng trực tiếp trong các thí nghiệm, không tải hoặc chuẩn bị lại. Đường dẫn khai báo trong `.env` là `RPV_DATA_DIR=data/dataset_v2`.

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

| Thành phần | Chức năng | Yêu cầu khi sao chép |
|---|---|---|
| `known/`, `unknown/`, `manifest.json` | Ảnh và manifest hợp lệ để train/calibrate/evaluate bằng CLI | Bắt buộc |
| `metadata.csv` | Trực quan, thống kê phiên chụp, nhãn có/không túi và unknown bổ sung | Bắt buộc cho bộ notebook này |
| `model_profile.json`, `splits_model_k2.json` | Đối chiếu cấu hình 2-shot và split đã chốt | Nên chép |
| `_extra_unknown/` | Đánh giá OOD bổ sung ở notebook 05 | Khi cần phép thử này |
| `hard_pairs.csv` | Kiểm chứng lời khai sai giữa các giống dễ nhầm | Khi cần phép thử này |
| `groups.csv`, `classes.csv`, `label_mapping_vi.csv`, `quality_report.json` | Thông tin nhóm, nhãn và chất lượng để làm báo cáo | Nên chép để lưu hồ sơ dữ liệu |
| Các file `*.txt`, báo cáo `*.md`, `model_split_counts.csv` | Danh sách split/báo cáo để tham khảo | Tùy chọn; model tự dựng split theo manifest và seed |

Hồ sơ chuẩn bị dữ liệu được tách riêng với dữ liệu dùng để chạy mô hình. Các mục `_archive/`, `_excluded/`, `_preview/`, `_sources/` và `_preparation_log.txt` thuộc hồ sơ chuẩn bị nên không bắt buộc phải sao chép vào bộ chạy. Checkpoint, prototype, calibration, DB và embedding cache được tạo trong `var/`, không đặt lẫn vào `data/`.

Tên lớp, tên ảnh và cấu trúc bên trong ba pool được giữ nguyên để khớp với manifest. Tên lớp tiếng Việt đã nằm trong manifest; đổi tên sẽ làm thay đổi fingerprint của dataset. Tên notebook bằng tiếng Anh chỉ phục vụ tổ chức mã, không ảnh hưởng nhãn dữ liệu.

Bộ dữ liệu chính có **17 known / 9.924 ảnh** và **7 unknown / 4.200 ảnh**. Phần unknown bổ sung có **38 lớp / 21.202 ảnh**. Khi chỉ làm thí nghiệm trên bộ chính, vẫn giữ metadata gốc và bỏ qua riêng ô xem trước dữ liệu bổ sung ở notebook 01 cùng ô đánh giá unknown bổ sung ở notebook 05.

Lệnh kiểm tra dữ liệu từ thư mục gốc:

```powershell
python -m rpv.cli dataset-info
```

Lệnh này kiểm tra manifest, hash ảnh và nhóm độc lập. Các bước tiếp theo nằm trong notebook 01–06 hoặc được thực hiện qua `run_all`:

```powershell
python scripts/run_all.py --encoder dinov2 --train-head
```
