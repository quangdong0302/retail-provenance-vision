# Giao thức prototype few-shot

Source gốc là `D:\Downloads\Source`. Giữ cấu trúc `rpv/`, `notebooks/`, `scripts/`, `tests/`, `api/`, `app/`, `docs/`, `data/` và `var/`; giữ công cụ train, đánh giá, nghiệp vụ và các comment. Bỏ `catalog_classifier.py`, test dành riêng cho classifier và `clean_caches.py`. Bank/calibration logistic cũ không dùng chung với prototype.

Mô hình dùng DINOv2 ViT-S/14 đóng băng, ảnh 336 px pad, head dùng chung, không TTA. Mỗi SKU mặc định là trung bình chuẩn hóa của **hai ảnh support**; có thể đối chiếu biến thể hai prototype riêng bằng validation. Khoảng cách là Euclidean; thêm SKU không fit lại classifier. Head học bằng episode từ nhiều ảnh base: few-shot áp dụng cho support/catalog và đăng ký lớp mới, không có nghĩa học toàn bộ backbone từ hai ảnh.

Trạng thái khởi động: chưa có checkpoint, prototype, calibration hay kết quả kèm theo. Train head trước khi chạy calibration và evaluation. Các giá trị trong `.env` là cấu hình cần chạy, không phải bằng chứng về chất lượng đã đạt.

Trainer dùng cùng prototype trung bình trong loss và validation. Support lấy từ hai nhóm chụp khi đủ nhóm; query ở nhóm khác. Chỉ known-train nhận gradient. Chọn head bằng trung bình accuracy known-validation và novel-development 2-shot; giữ epoch 0 để tránh ghi đè head bằng kết quả kém hơn. Có thể warm-start với `train-head --init-head <head.pt>`.

Extra được khóa theo lớp: 12 development để chọn head, 12 reserved và 14 holdout. Reserved tách thành sáu lớp calibration và sáu lớp selection, seed 20261013. Không dùng lại các lớp chọn head để fit risk. Main unknown-test và extra-holdout không chọn head/cấu hình/ngưỡng. Kiểm tra hash và tách nhóm vẫn được thực hiện đầy đủ.

Energy OOD fit trên known-calibration; selective risk fit trên known-calibration và sáu reserved-calibration. So sánh `conformal` và `prototype_selective` trên known-validation và sáu reserved-selection. Cách thứ hai chọn prototype gần nhất, yêu cầu Energy OOD cùng ngưỡng confidence đã hiệu chỉnh; điểm đồng hạng chuyển nhân viên. Tập conformal vẫn báo riêng. Confidence margin/energy không phải xác suất đúng nhãn. SKU đăng ký có hướng dẫn vẫn cần nhân viên theo quy trình hiện có.

Ưu tiên phương án tăng đồng thời accuracy known và coverage so với đối chứng đã định nghĩa; chọn coverage lớn nhất trong nhóm đó. Risk known, risk hỗn hợp theo số ảnh và risk hỗn hợp cân bằng known/unknown 50:50 đều phải dưới 5%. Đây là ràng buộc thực nghiệm cần đo trong lượt chạy mới, không bảo đảm trên mọi dữ liệu. Chỉ đưa cấu hình vào sử dụng sau khi ghi rõ giao thức và kết quả validation.

Đặc trưng màu HS foreground của Source được thử như một cue ghép với vector DINO trước khi dựng prototype. Không có bước fit logistic. So sánh các trọng số màu trên validation và chọn theo cùng ràng buộc risk; `RPV_PROTOTYPE_COLOR_WEIGHT=0` tắt cue này. Màu có cache độc lập 96 chiều, còn backbone có cache 384 chiều.

## Metric dùng chung tùy chọn

Có thể học ma trận co giãn theo covariance nội lớp từ known-train, dùng regularization để tránh khuếch đại các chiều có phương sai nhỏ. Centering, độ co giãn và trọng số màu phải được chọn bằng validation/novel-development, không dùng test. Khi áp dụng, matrix/bias được lưu trong checkpoint và dùng chung cho mọi SKU; thêm SKU mới không fit lại matrix.

Đây là một biến thể metric dùng chung cần có ablation, không tuyên bố phát minh phép whitening hay mặc định rằng whitening làm accuracy tăng. Trainer từ đầu không tự thực hiện tìm kiếm metric; một head vừa train và một head có metric đã chọn là hai ứng viên khác nhau, phải được định danh và đánh giá riêng.

## Vector lưu bền

`var/embed_cache/*.npz` chứa vector backbone float32. Khóa gồm hash trọng số/preprocessing, SHA256 nội dung ảnh và TTA. Head áp dụng **sau** cache, vì vậy đổi/train lại head dùng lại vector. Thay ảnh/backbone/resize/TTA tạo khóa khác. Cache tự lưu mỗi 256 ảnh và khi thoát vòng nhúng, ghi nguyên tử và hợp nhất giữa các notebook.

```powershell
python scripts/run_all.py --train-head
$env:RPV_PROJECTION_CKPT = (Resolve-Path 'var/thesis_v4/projection_head.pt').Path
```

Sau khi đã có head, có thể chuẩn bị cache và nghiên cứu cấu hình. Các script search/finalize cần đầu vào của bước trước; đọc tham số và điều kiện trước khi chạy:

```powershell
python scripts/cache_embeddings.py
python scripts/improve_prototypes.py --wait-cache
python scripts/cache_embeddings.py --color
python scripts/search_color_prototypes.py
python scripts/search_shared_metric.py
python scripts/finalize_prototypes.py
```

Search ghi ứng viên/đối chiếu, không tự thay mô hình triển khai. Finalize chốt bằng validation, dựng bank/calibration rồi đánh giá test. Khi head đã có và đã được cấu hình, chạy `python -m rpv.cli calibrate`, sau đó `python -m rpv.cli evaluate --use-catalog --scenarios`. Notebook 02 có `run.cache_info()`; gọi `run.warm_cache()` để nhúng toàn bộ ảnh một lần. Terminal và notebook dùng chung `.env`; artifact sẽ được ghi vào các đường dẫn đầu ra đã cấu hình.

Đóng góp có thể kiểm chứng: episode tách nhóm chụp; hình học mean nhất quán; chọn head cân bằng base/novel; tách calibration/selection theo lớp; cổng selective với ablation coverage/risk; tích hợp provenance, policy và audit. Cần đo hiệu quả thay vì mặc định xem ghép backbone với prototype là một thuật toán mới.

Ảnh nhận diện SKU; nguồn gốc lấy từ hồ sơ. PackEat chưa thay thế dữ liệu POS Việt Nam và hard negative mà giảng viên yêu cầu. Thêm lớp mới cần đo lại calibration/risk; accuracy few-shot sau enrollment không đồng nghĩa coverage tự động đã hiệu chỉnh.
