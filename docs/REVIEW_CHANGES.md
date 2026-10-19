# Đối chiếu nhận xét của giảng viên ngày 17/09/2026

Tài liệu này đối chiếu 5 yêu cầu trong PDF với phần hiện thực và cách chạy.
Các chức năng đã được bổ sung không thay thế bằng chứng thực nghiệm trên ảnh
thật tại POS. Không mặc định tuyên bố đạt risk 1% hay ngân sách độ trễ.

| Yêu cầu | Hiện thực | Kiểm chứng |
|---|---|---|
| Decision(I,S,L), 4 trạng thái | `rpv/provenance/policy.py`, `DecisionState`, trường `decision_state` trong API/demo/DB payload/audit | `tests/test_teacher_review.py`; `evaluate --scenarios` |
| OOD bằng Energy/Mahalanobis | Chọn Energy trên logits prototype trong `rpv/vision/rejection.py`; ngưỡng từ known-calibration | AUROC, FAR, FPR@95TPR; đối chiếu distance baseline |
| Risk–Coverage, risk ≤1% | Chọn ngưỡng confidence trên known-calibration; khóa trước test | `operating_point_known`, `operating_point_known_and_unknown`, đường cong test |
| Túi nilon/lưới, màu/độ chín, hard negative | `evaluate --robustness --hard-pairs ...`; verifier với SKU sai gần nhất | Báo cáo từng điều kiện và FAR; cần bổ sung ảnh POS thật |
| Phần cứng và <150/<50/<300 ms | `benchmark`, đo p50/p95/p99 có/không cache, cấu hình máy thực tế | Kiểm tra p95 từng giai đoạn, báo cả trường hợp vượt ngân sách |
| Chương pháp lý/nghiệp vụ, sơ đồ 7 bảng | [Pháp lý và nghiệp vụ](LEGAL_BUSINESS.md), [7 bảng](DATABASE.md), bố cục cập nhật trong [THESIS](THESIS.md) | Đối chiếu schema SQLite và nguồn văn bản gốc |

## Quy tắc toán học và tín hiệu vận hành

Với ảnh I, SKU khai báo S và lô L, encoder tạo z = normalize(h(DINOv2(I))).
Khoảng cách d_c(z) là khoảng cách Euclid nhỏ nhất tới các prototype của lớp c.
Logit prototype f_c(z) = −d_c(z)². Energy:

\[
E_T(z)=-T\log\sum_{c\in\mathcal C}\exp(-d_c(z)^2/T).
\]

T cố định trước test (`RPV_ENERGY_TEMPERATURE=0.05`), hoặc được chọn trên
validation bằng một thí nghiệm riêng. Ngưỡng τ_E là order statistic nhỏ nhất
đạt `RPV_TARGET_KNOWN_RECALL` trên known-calibration. E>τ_E → OOD. Đây là
ứng dụng công thức Energy cho logits khoảng cách prototype, không phải mô
hình được huấn luyện bằng energy loss của bài báo gốc.
[Liu et al., 2020](https://arxiv.org/abs/2010.03759).

Tập conformal Γ(I) vẫn được tính từ khoảng cách đến từng lớp. Chỉ kết luận
thị giác chắc chắn khi E≤τ_E, Γ có đúng một phần tử, SKU không cần review
đăng ký và confidence q=−E đạt ngưỡng rủi ro t. P-value conformal là hạng
nonconformity, **không phải** P(Identity(I)=S). Code chưa có bộ hiệu chỉnh
xác suất nên không dùng p-value như xác suất trong công thức Decision.

Ký hiệu H(I) là điều kiện chắc chắn trên, A(I) là phần tử duy nhất của Γ,
V(S,L) là lô tồn tại, thuộc S, đang active, có hạn sử dụng hợp lệ và chưa
hết hạn. C(L) là hồ sơ có nước xuất xứ, đủ chứng từ bắt buộc đang verified
và còn hiệu lực, không có chứng từ bị từ chối hay lô bị chặn. Policy dùng:

\[
Decision(I,S,L)=
\begin{cases}
Mismatch & H(I)\land A(I)\ne S\\
Abstain & \neg H(I)\\
ProvenanceException & H(I)\land A(I)=S\land\neg(V(S,L)\land C(L))\\
Match & H(I)\land A(I)=S\land V(S,L)\land C(L).
\end{cases}
\]

Nhận diện tự do dùng S=A(I) khi có kết luận chắc chắn. SKU chưa có prototype,
tập mơ hồ, thiếu calibration hoặc không có ngưỡng risk khả thi đều cần nhân
viên. Bốn trạng thái nghiên cứu được tách khỏi hành động vận hành:

| `decision_state` | Hành động thường gặp |
|---|---|
| `match` | `allow`, hoặc `allow_warning` nếu gần hạn/chứng từ tùy chọn pending |
| `mismatch` | `block` |
| `abstain` | `assist`; có thể `block` nếu đồng thời phát hiện lô thu hồi/hết hạn |
| `provenance_exception` | `assist` khi thiếu hồ sơ/chứng từ; `block` khi hết hạn/thu hồi/sai SKU của lô |

Luật nghiêm trọng nhất vẫn quyết định hành động. Khi nhiều lỗi cùng xuất
hiện, ưu tiên tên trạng thái Mismatch → Abstain → ProvenanceException;
mọi mã lý do được giữ, kể cả lỗi lô xuất hiện cùng lỗi thị giác.
`audit_notes.decision_inputs` lưu H, V và kết quả compliance để truy vết.
Các bản calibration cũ vẫn đọc được để tương thích; phải chạy **calibrate
lại** để có Energy và risk gate mới. Không coi artifact cũ là đã nâng cấp.

## Risk và coverage phải phân biệt

Với g_t là điều kiện trả lời tự động, coverage=Σg_t/n và selective
risk=Σ[g_t·1(A(I)≠y)]/Σg_t. Unknown được trả lời như known được tính là lỗi
trong báo cáo hỗn hợp. Không có câu trả lời thì risk là `null`, không phải 0.
Đây là đánh đổi selective classification theo
[Geifman & El-Yaniv, 2017](https://arxiv.org/abs/1705.08500).

Chọn t trên known-calibration để coverage lớn nhất với risk thực nghiệm
≤`RPV_TARGET_RISK=0.01`; chỉ xét các nhóm score bằng nhau đầy đủ. Nếu không
có ngưỡng khả thi, t=`null` và dừng trả lời tự động. Test được chạy sau khi
khóa t; `target_met_on_test` báo kết quả thực tế, không chỉnh t theo test.

`coverage_at_risk_1pct` hiện có là điểm mô tả đường cong của top-1 classifier
trên known-test; nó không phải coverage của gate triển khai. Coverage chứa
nhãn đúng của tập conformal cũng không phải coverage trả lời tự động.
`conformal_prediction_set` lưu tập trước OOD; `prediction_set` lưu tập sau
OOD. Coverage conformal được đo trên tập gốc, báo riêng coverage sau OOD;
ghép thêm các gate không giữ nguyên bảo đảm coverage của conformal.
Ràng buộc risk hiện là **thực nghiệm**, không có bảo đảm risk dân số ở mức
1% với xác suất cao; cần nhiều mẫu/session độc lập và một thủ tục bound
được thiết kế riêng nếu muốn tuyên bố bảo đảm thống kê. Unknown không được
đưa vào fit gate trong giao thức hiện tại.

Báo cáo xuất thêm `*.risk_coverage.csv`, `*.risk_coverage.svg` và
`*.risk_coverage.png`; Markdown nhúng hình PNG. Điểm gate triển khai được
vẽ riêng, không mặc định nằm trên đường cong top-1 vì quy tắc khác nhau.

Sau đăng ký novel, số lớp trong logsumexp thay đổi. Thí nghiệm 1/3/5-shot
chỉ chuyển ngưỡng conformal pooled; **không chuyển Energy/risk gate known**
sang bank mới. Bổ sung SKU trong vận hành vẫn vô hiệu hóa calibration,
và cần hiệu chỉnh lại trước khi cho phép tự động.

## Chạy và đọc kết quả

Từ môi trường `build_ML`, dùng bộ dữ liệu `data/dataset_v2` đã có:

```powershell
conda activate build_ML
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:PYTHONIOENCODING = "utf-8"
python -m rpv.cli calibrate --encoder dinov2
python -m rpv.cli evaluate --encoder dinov2 --scenarios --robustness
python -m unittest discover -s tests -t .
python -m ruff check .
```

Nếu dùng head đã train, đặt `RPV_PROJECTION_CKPT` tới đúng checkpoint và
giữ nguyên split/seed/k-shot của checkpoint. Không chuẩn bị lại dataset.
`evaluate` lưu bank/calibration trong run riêng; `calibrate` cập nhật bộ
artifact vận hành. Restart API/demo sau khi thay bộ artifact.

Chỉ định hard negative có ground truth do người thu thập dữ liệu xác nhận:

```csv
true_sku,claimed_sku,kind
apple_10,apple_11,cultivar
```

```powershell
python -m rpv.cli evaluate --robustness --hard-pairs data/hard_pairs.csv
python -m rpv.cli benchmark "path/to/held_out.jpg" --sku SKU_THAT --lot LOT_THAT --repeats 50 --warmup 5
```

`kind=origin` chỉ dùng khi danh mục thực sự có hai SKU khác xuất xứ và nhãn
đã xác minh; không tạo nhãn xuất xứ từ tên thư mục. Trường hợp cùng SKU,
khác lô được kiểm tra riêng bằng `same_sku_different_lot`.

Stress test sinh overlay túi/lưới, color shift, ánh sáng yếu và blur trên
ảnh held-out, giữ nguyên model/ngưỡng. Color shift chỉ là nhiễu màu, chưa
phải đánh giá độ chín vật lý. Cần chụp thật từng vật thể/session qua túi,
lưới và các mức chín, chia nhóm trước học/hiệu chỉnh/test, báo số mẫu và
kết quả theo từng điều kiện. `--stress-max-per-class` mặc định 5 để giới
hạn chi phí; ghi rõ số mẫu trong báo cáo, tăng khi làm thực nghiệm chính.

Benchmark dùng bản sao DB và audit riêng trong thư mục tạm; không ghi
giao dịch giả vào DB/audit vận hành. Model đã warm, batch 1, có và không
cache; total gồm tra hồ sơ, policy, ghi DB/audit và sự kiện bán hàng, chưa
gồm camera/network/HTTP upload/model startup. Chạy với SKU/lô thật để đo
nhánh có hồ sơ; báo rõ nếu chạy nhánh thiếu hồ sơ. `meets_budget_p95` dùng
điều kiện nghiêm ngặt <150/<50/<300 ms, không suy từ thời gian cache ra
khả năng xử lý ảnh mới.

Cấu hình POS đề xuất để kiểm chứng: CPU x86-64 ≥4 lõi, RAM 16 GB, SSD,
GPU NVIDIA ≥4 GB VRAM, DINOv2 ViT-S/14, ảnh 224 px, batch 1. Đây là cấu
hình mục tiêu, chưa phải cấu hình tối thiểu đã chứng minh. Báo cáo ghi CPU,
GPU, OS, PyTorch, số thread, encoder/head, p50/p95/p99 và từng mẫu thời gian.

`scripts/smoke_review.py` kiểm tra tích hợp trên một phần nhỏ ảnh thật bằng
DINOv2; các artifact ghi rõ `smoke_only`, không phải kết quả khóa luận.

## Câu hỏi nghiên cứu và tên đề tài

- Q1: Khả năng nhận SKU từ 2–3 ảnh, AUROC/FAR OOD trước đăng ký, accuracy novel sau đăng ký.
- Q2: Risk–Coverage và coverage/risk tại ngưỡng calibration đã khóa, riêng known và hỗn hợp unknown.
- Q3: Số chấp nhận sai của image-only so với full-policy trên cùng kịch bản có ground truth; báo ảnh cùng SKU nhưng lô khác.
- Q4: p95/p99, ngân sách từng giai đoạn, fingerprint model, tính nhất quán DB và kiểm tra audit hash chain.

Hai tên thầy gợi ý vẫn là lựa chọn biên tập, chưa tự đổi tên dự án:
“Nghiên cứu kiến trúc Vision-to-Provenance kiểm chứng nguồn gốc sản phẩm bán
lẻ không bao bì dựa trên học ít mẫu và cơ chế từ chối có kiểm soát rủi ro”;
hoặc “Xây dựng hệ thống định danh nông sản tươi sống và xác minh hồ sơ lô
hàng tại quầy POS tích hợp thị giác máy tính và Policy Engine đối chiếu đa nguồn”.
