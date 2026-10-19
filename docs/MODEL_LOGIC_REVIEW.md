# Rà soát logic và đề xuất cải thiện mô hình

Phạm vi: prototype, confidence/OOD, hiệu chỉnh, kiểm chứng SKU, Policy Engine, báo cáo và luồng notebook/terminal. Tài liệu giữ các quy tắc thiết kế và kế hoạch kiểm chứng; không kèm checkpoint hay số liệu của lượt chạy cũ. SKU đăng ký có hướng dẫn vẫn phải tuân thủ cờ bắt buộc nhân viên duyệt.

## Các lỗi đã sửa

| Vấn đề | Hậu quả trước sửa | Hành vi sau sửa |
|---|---|---|
| Verifier bỏ qua Recognizer khi OOD/risk rỗng | Chế độ prototype_selective có thể trả confirmed dù chưa có gate | Chuyển nhân viên; không xác nhận tự động khi thiếu gate |
| Cờ operator_required không được kiểm tra ở nhánh verifier cũ | SKU cần duyệt có thể bị xác nhận qua API kiểm chứng | Verifier cũng phải tuân thủ cờ duyệt |
| Policy không ràng buộc SKU thị giác với SKU hồ sơ | Xác nhận A có thể đi cùng hồ sơ hợp lệ của B và vẫn allow_warning | Block, trạng thái mismatch, ghi lý do VISION_MISMATCH |
| Bản đồ khoảng cách rỗng với catalog không rỗng | Bị hiểu nhầm là catalog rỗng | Báo đầu vào không hợp lệ |
| Notebook chỉ kiểm tra calibration có tồn tại | Sau train lại có thể dùng báo cáo/head/ngưỡng cũ | Kiểm tra hash head qua encoder tag, fingerprint prototype và calibration của báo cáo |
| Đánh giá notebook tự fit lại calibration | Không nhất thiết đánh giá đúng ngưỡng đang vận hành | Thêm evaluate --use-catalog; notebook và run_all dùng artifact đã lưu |
| Coverage sau OOD lấy tập quyết định top-1 | Trộn độ phủ conformal với độ đúng của top-1 | Tính từ tập conformal sau OOD; lưu thêm decision_set_label_coverage riêng |
| Log unknown thiếu tập conformal/confidence | Khó phân tích và so sánh cùng schema với known | Known/unknown cùng lưu các trường cần thiết |
| Báo cáo chỉ gắn target_met với known, lấy target từ config hiện thời | Có thể gây hiểu nhầm mixed risk cũng đạt hoặc dùng sai mục tiêu của artifact | Known và mixed có target/target_met riêng, lấy mục tiêu của calibration đã lưu |
| CLI Classical thừa hưởng màu/quy tắc của model chính | Baseline terminal khác baseline notebook | Tắt màu, dùng conformal như baseline notebook |
| Lý do chuyển người cho tie/operator không rõ | Nhật ký thiếu nguyên nhân quyết định | Có rejection_reason riêng cho tie và operator_required |
| Publish dataset gặp khóa Windows ngắn hạn | run_all có thể dừng ở test chuẩn bị dataset tạm | Retry có giới hạn, kiểm tra phạm vi staging, giữ nguyên lỗi quyền vĩnh viễn và không ghi đè đích |

Test hồi quy: `tests/test_logic_audit.py`. Các thay đổi bảo vệ tính đúng của luồng quyết định; không được diễn giải là làm accuracy của encoder tăng. Kết quả kiểm tra phải lấy từ lượt chạy mới.

## Kế hoạch thử confidence

Sau khi có head, prototype, calibration và cache, có thể dùng `scripts/review_confidence.py` để so sánh confidence trên development. Script lấy vector từ cache; nếu thiếu vector, script dừng, không trích xuất lại. Prototype và head đóng băng. Không dùng main known-test, novel-test hay extra holdout để tính điểm/chọn ứng viên.

So sánh Energy, nearest-ratio và score kết hợp `s = 1 - d1/d2 - beta * E`. Fit các ngưỡng trên known-calibration và extra-calibration; so sánh trên known-validation và extra-selection khác lớp. Công bố lưới tham số trước khi đánh giá final và báo cả ứng viên không đạt. Không mặc định rằng score kết hợp tốt hơn Energy.

Đánh giá known risk, mixed risk theo số ảnh, mixed risk cân bằng và các kịch bản tỷ lệ unknown đã công bố. Chưa có kết quả thực nghiệm kèm theo tài liệu này.

Accuracy không đổi khi head và prototype không đổi. Thay confidence chỉ thay việc chấp nhận/từ chối; không hứa nâng val accuracy bằng cách đổi gate.

## Các khoảng trống phương pháp còn lại

1. **Risk calibration là thực nghiệm.** Tỷ lệ lỗi trên một số lớp extra không bảo đảm giữ nguyên trên lớp unknown khác hay mọi tỷ lệ giao dịch unknown. Với a = tỷ lệ known được chấp nhận, e = tỷ lệ known vừa sai vừa được chấp nhận, f = tỷ lệ unknown được chấp nhận, và pi = tỷ lệ unknown trong luồng, mixed risk là `((1-pi)*e + pi*f) / ((1-pi)*a + pi*f)`. Báo riêng known risk, unknown false-accept và các kịch bản pi đã công bố. Kịch bản 75% là stress giả định, không phải thống kê POS.
2. **Chọn head và chọn toàn bộ hệ thống là hai bước.** Notebook train chọn theo accuracy base/novel; một head mới không tự kế thừa phép biến đổi metric đã chọn ở một cấu hình khác và có thể không đạt yêu cầu risk–coverage. Hash guard ngăn trộn artifact, nhưng không chứng minh head mới tốt hơn.
3. **Enrollment hiện đo nhận dạng, chưa chứng minh tự động hóa an toàn.** Accuracy sau enrollment không thay thế mixed risk sau enrollment. Chuẩn hóa Energy theo log số lớp không giải quyết toàn bộ thay đổi prototype, margin và nhãn gần nhất. Giữ cơ chế cần nhân viên duyệt cho đến khi có bằng chứng riêng.
4. **Few-shot phải tính đúng số ảnh.** Có thể học encoder/metric từ nhiều ảnh base, nhưng SKU mới chỉ có 2–3 ảnh support theo giao thức. Nếu dùng thêm ảnh gán nhãn của SKU mới để fit ngưỡng, phải công bố; không gọi đó là chỉ cần 2–3 ảnh. Query final không dùng để chọn phương án/ngưỡng.
5. **Tập novel mất cân bằng theo lớp/nhóm chụp.** Báo cả macro-F1, accuracy từng SKU, số query và nhiều lần lấy support theo nhóm chụp. Một điểm accuracy gộp trên một support seed chưa chứng minh tính ổn định.
6. **Tính độc lập của test phụ thuộc lịch sử sử dụng dữ liệu.** Dọn artifact không làm một tập test đã quan sát trở thành holdout mới. Thực nghiệm cần holdout/capture mới hoặc công bố rõ giới hạn này.

## Đề xuất tối ưu tiếp theo

### 1. Chọn toàn bộ pipeline theo yêu cầu, thay vì chỉ chọn head theo accuracy

Sau khi train, lưu head đối chứng cùng định danh hash. Mỗi phương án có riêng head, bank, calibration, split và báo cáo; chỉ đưa vào sử dụng khi qua các điều kiện validation đã công bố. Không tự thay bản đã chốt chỉ vì notebook train vừa chạy xong.

Trình tự: học biểu diễn trên base-train → dựng prototype từ support → fit ngưỡng trên calibration → chọn phương án bằng known-validation và novel-development → đóng băng → đánh giá final. Cùng một quy tắc score/eligible/tie/OOD phải được dùng trong nghiên cứu và runtime.

Tiêu chí: tăng coverage so với đối chứng, known/mixed risk <5% trên các kịch bản được công bố; novel 2/3-shot không suy giảm đáng kể; độ trễ đạt ngân sách. Nếu không có ứng viên đạt, báo không đạt thay vì nới điều kiện hoặc chọn ngưỡng từ test.

### 2. Ưu tiên khả năng phân tách biểu diễn trước khi mở rộng score

Thử một số ít shared metric có shrinkage và residual với biến đổi đồng nhất làm đối chứng; chỉ fit bằng base-train. Dựng lại prototype từ đúng 2 ảnh sau phép biến đổi, không biến đổi riêng tâm đã trung bình rồi xem là tương đương.

Chọn bằng cả known-validation và novel-development; bỏ phương án chỉ cải thiện các lớp base nhưng làm novel kém đi. Có thể dùng Mahalanobis với covariance dùng chung làm đối chứng; không ước lượng covariance riêng từng SKU từ 2–3 ảnh. Chưa đưa Power Transform, nhiều backbone hay nhiều gate vào cùng lúc khi chưa có ablation chứng minh lợi ích.

Metric/encoder cải thiện mới có khả năng tăng accuracy; score chỉ điều tiết risk–coverage. Cần phân tích theo cặp nhầm và nhóm chụp để phân biệt vấn đề biểu diễn, support không đại diện và dữ liệu.

### 3. Thiết kế thí nghiệm risk sau enrollment đúng few-shot

Tạo episode trên các lớp development, mỗi SKU mới 2 hoặc 3 support thuộc nhóm chụp khác query. Hiệu chỉnh ở mức episode trên các lớp/capture development riêng, không học ngưỡng bằng query final của SKU mới. Đánh giá đồng thời known cũ, SKU vừa thêm và các unknown chưa đăng ký.

Lặp nhiều support seed, công bố kích thước catalog và macro theo lớp. Bất kỳ chuẩn hóa theo số lớp nào cũng cần đối chứng trước/sau; không coi nó là bảo đảm risk. Giữ chế độ nhân viên duyệt trong vận hành cho đến khi giao thức này qua kiểm tra.

Đặt 2/3-shot làm giao thức chính: dành tối đa ba nhóm support, dùng cùng query cho hai mức shot. Giữ 5-shot thành thí nghiệm phụ riêng, vì dành trước năm nhóm support có thể làm một lớp ít nhóm gần như không còn query. Tạo phiên bản giao thức riêng, không thay split hoặc sửa lại số liệu đã công bố để làm kết quả đẹp hơn.

### 4. Chứng minh giá trị Vision-to-Provenance

So sánh cùng giao dịch giữa vision-only và vision + khai báo SKU + hồ sơ lô. Báo đủ Match, Mismatch, Abstain, Provenance Exception; đo chấp nhận sai, chuyển người và phát hiện sai hồ sơ. Hình ảnh không chứng minh xuất xứ.

Thử túi/lưới, độ chín, hard negative; ghi rõ dữ liệu công khai hay tự chụp. Đo p95 vision/policy/tổng trên phần cứng được công bố, phân biệt có/không cache và hồ sơ mô phỏng với vận hành tại POS thật.

## Đóng góp có thể bảo vệ

Đóng góp nên là giao thức few-shot có kiểm soát rủi ro khi danh mục thay đổi, cơ chế chọn pipeline tái lập được và Policy Engine đối chiếu đa nguồn có bằng chứng thực nghiệm. Energy, Mahalanobis và prototype là phương pháp có sẵn; việc ghép chúng không tự tạo ra tính mới hay bảo đảm đóng góp đủ lớn. Cần ablation và kết quả đo cho từng thành phần.

Tài liệu nền: [Prototypical Networks](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html), [Energy OOD](https://papers.nips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html), [Mahalanobis OOD](https://papers.nips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html). Nếu muốn phát biểu bảo đảm thống kê thay vì ràng buộc thực nghiệm, phải xác định giả thiết và phương pháp phù hợp, chẳng hạn [Learn then Test](https://arxiv.org/abs/2110.01052); không suy ra bảo đảm đó từ một ngưỡng có lỗi calibration dưới 5%.

## Chạy và kiểm tra

```powershell
python scripts/run_all.py --train-head
$env:RPV_PROJECTION_CKPT = (Resolve-Path 'var/thesis_v4/projection_head.pt').Path
python -m rpv.cli evaluate --use-catalog --scenarios
python scripts/review_confidence.py --out var/confidence_review.json
```

Chọn môi trường/kernel `build_ML`, chạy tại thư mục gốc dự án. Các lệnh sau lệnh train yêu cầu head đã được tạo thành công. Có thể ghi đường dẫn head vào `.env` để dùng lại trong terminal/API khác. Notebook dùng đường dẫn `RPV_HEAD_OUT` cho các bước sau train.

Restart kernel trước lượt chạy mới. Khi đánh giá, ghi riêng coverage của tập conformal, coverage sau OOD và coverage tự chấp nhận. Kiểm tra CLI/notebook dùng cùng artifact, lưu số lỗi/số trả lời và kiểm tra audit. Không coi việc có test trong source là bằng chứng test đã chạy hoặc model đã đạt yêu cầu.
