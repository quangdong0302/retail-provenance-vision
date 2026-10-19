# Đánh giá dự án và đối chiếu nhận xét giảng viên

Tài liệu đối chiếu thiết kế trong mã nguồn với nhận xét giảng viên và đề cương.
Không kèm số liệu hay báo cáo của lượt chạy cũ. Hiệu quả mô hình phải được đánh giá
sau khi train, hiệu chỉnh và khóa giao thức cho lượt chạy mới. Các gợi ý tên đề tài
được dùng để định hướng nội dung, không phải lệnh tự đổi tên.

## 1. Kết luận

**Dự án có nền tảng kỹ thuật tốt và phù hợp để phát triển thành khóa luận, nhưng chưa đủ
bằng chứng để khẳng định đã đáp ứng toàn bộ tiêu chí của thầy.** Phần hiện thực đi đúng hướng;
phần thực nghiệm trên POS thực tế và lập luận thống kê cần hoàn thiện.

Phân biệt chức năng đã có trong code với bằng chứng kiểm thử và kết quả trên dữ liệu
độc lập ở điều kiện vận hành mục tiêu. Trạng thái khởi động không kèm bằng chứng thực nghiệm;
không suy hiệu quả từ việc chỉ có module hoặc test trong source. Đây không phải điểm chính thức của giảng viên.

| Khía cạnh | Nhận xét |
|---|---|
| Ý tưởng và phạm vi | Tốt: bài toán SKU không mã vạch kết hợp hồ sơ lô có luồng nghiệp vụ rõ |
| Kiến trúc và chất lượng hiện thực | Khá tốt: service dùng chung giữa CLI/API/demo, có kiểm thử nhất quán và khóa ghi |
| Phương pháp nghiên cứu | Có nền tảng: DINOv2, prototype, Energy OOD, conformal và selective risk |
| Bằng chứng hiệu quả | Cần thu thập bằng lượt chạy mới, đối chứng cùng giao thức và test POS thật |
| Khả năng bảo vệ khóa luận | Có nền tảng prototype; cần số liệu trước khi kết luận đạt mục tiêu risk hoặc sẵn sàng triển khai |

## 2. Điểm mạnh

- **Tách đúng Identity và Provenance.** Ảnh hỗ trợ nhận SKU; nguồn gốc đến từ hồ sơ lô,
  nhà cung cấp và chứng từ. Đây là lựa chọn quan trọng nhất của kiến trúc.
- **Policy có giải thích.** Có bốn `decision_state` nghiên cứu, hành động vận hành riêng,
  mã lý do, quy tắc tăng mức nghiêm trọng và lưu đầu vào quyết định.
- **Không tự tin giả khi thiếu hiệu chỉnh.** Bank thay đổi làm ngưỡng cũ mất hiệu lực;
  mặt hàng đăng ký có kiểm tra vẫn yêu cầu nhân viên xác nhận.
- **Luồng đăng ký có ích hơn chỉ “thêm vài ảnh”.** Kiểm tra chất lượng, nhóm ảnh,
  SKU dễ nhầm, lịch sử và hướng dẫn bổ sung mẫu nằm trong service riêng.
- **Có hạ tầng đánh giá.** Metric OOD/verifier, đường cong risk–coverage, novel few-shot,
  kịch bản image-only/full-policy, perturbation và benchmark được hiện thực.
- **Có kiểm tra tính nhất quán.** Prototype/calibration gắn fingerprint; SQLite có ràng buộc,
  audit có hash chain và khóa nhiều tiến trình. Test tập trung vào những lỗi có thể làm sai kết luận.

Nguồn: `rpv/services/checkout.py`, `rpv/services/enrollment.py`, `rpv/vision/`,
`rpv/provenance/policy.py`, `rpv/evaluation/`, `tests/`.

## 3. Đối chiếu yêu cầu giảng viên

| Yêu cầu của thầy | Bằng chứng hiện có | Đánh giá | Phần cần bổ sung |
|---|---|---|---|
| 1. Toán học hóa `Decision(I,S,L)` và bốn trạng thái | Công thức H(I), V(S,L), C(L) trong `REVIEW_CHANGES.md`; `DecisionState`, policy, API/audit; test bốn trạng thái | **Đã hiện thực; đạt phần định nghĩa ở mức prototype** | Giải thích rõ H(I) là gate hiệu chỉnh, không phải xác suất P(Identity=S); nếu dùng ký hiệu xác suất trong luận văn phải có bộ hiệu chỉnh xác suất hoặc đổi công thức nhất quán |
| 2a. Energy Score hoặc Mahalanobis thay ngưỡng khoảng cách thủ công | `rejection.py`: Energy với logits −distance²; ngưỡng chọn từ known-calibration; áp dụng cả recognizer và verifier | **Đã hiện thực Energy**, đáp ứng lựa chọn “hoặc” | Chạy đối chứng Energy/distance trên cùng tập và seed; không cần thêm Mahalanobis chỉ để đủ số phương pháp |
| 2b. Risk–Coverage và coverage tại risk mục tiêu dưới 5% | Fit gate trên calibration, khóa trước test; có CSV/SVG/PNG, điểm vận hành known và hỗn hợp | **Đã có cơ chế; cần chứng minh bằng lượt chạy mới và dữ liệu POS** | Báo risk, coverage, số trả lời, số lỗi và độ bất định trên test độc lập; tránh dùng đường cong test để chọn lại ngưỡng |
| 3. Túi nilon/lưới, màu/độ chín, hard negative khác giống/xuất xứ | Sáu perturbation sinh từ ảnh held-out; hỗ trợ CSV hard pairs; verifier kiểm SKU sai gần nhất | **Mới đáp ứng một phần** | Thu ảnh thật qua túi/lưới, nhiều độ chín; xác minh nhãn giống/SKU/xuất xứ; chạy hard pairs có ground truth. Color shift chưa phải độ chín thật; cần báo FAR của hard negative có nhãn |
| 4. Phần cứng POS và <150 / <50 / <300 ms | Có công cụ ghi cấu hình phần cứng, p50/p95/p99 và benchmark cache/không cache | **Có công cụ đo; cần kết quả trên phần cứng mục tiêu** | Đo lặp nhiều lần trên danh mục đầy đủ, ảnh mới và nhánh hồ sơ hợp lệ; đo thêm camera/upload/API khi nói về thời gian trải nghiệm toàn hệ thống |
| 5. Chương pháp lý/nghiệp vụ riêng và sơ đồ bảy bảng | `LEGAL_BUSINESS.md`, `DATABASE.md`, schema SQL và bố cục bảy chương trong `THESIS.md` | **Đã có trong tài liệu dự án** | Đưa vào bản luận văn cuối, kiểm tra nguồn trích dẫn/phạm vi văn bản. Chưa có bản luận văn hoàn chỉnh để xác nhận đã trình bày đầy đủ |

Risk dưới 5% là mục tiêu thực nghiệm cần đánh giá; giá trị fit cụ thể lấy từ cấu hình.
Báo rõ known risk và mixed risk, không suy ra bảo đảm cho mọi phân phối. `p95` là cách
repo dùng để đánh giá ngân sách latency; cần công bố cách đo và phạm vi thời gian.

## 4. Bằng chứng cần tạo trong lượt chạy mới

### Dữ liệu và giao thức

Đọc manifest và metadata của dataset đang dùng để báo số lớp, số ảnh và các vai trò
train/support/calibration/validation/test. Kiểm tra trùng nội dung và tách nhóm vật thể,
phiên chụp; hash khác nhau không chứng minh là vật thể hay phiên chụp khác nhau.

Tên thư mục là nhãn thị giác; cần bảng ánh xạ và metadata để chứng minh các nhãn là
SKU/giống khác nhau. Không suy nhãn xuất xứ từ tên thư mục.

### Báo cáo nhận dạng và từ chối

Train head, dựng prototype từ support và chọn ngưỡng trên calibration trước khi đánh
giá test. Báo accuracy, macro-F1, AUROC/FPR, coverage và risk tại ngưỡng vận hành,
số trả lời, số lỗi và tỷ lệ unknown bị chấp nhận nhầm. Tách known và hỗn hợp known/unknown.

Coverage tại một mức risk trong đường cong quét test chỉ là chẩn đoán, không phải
coverage của gate đang vận hành và không dùng để chọn lại ngưỡng. Kịch bản policy dùng
hồ sơ mô phỏng phải được ghi rõ, không thay thế dữ liệu lô hàng thực tế.

### Độ trễ và phần cứng

Ghi CPU/RAM/GPU, kích thước danh mục, batch size, số lượt đo và warm-up. Báo p95
vision, policy và toàn service; đối chiếu ngân sách dưới 150/50/300 ms. Tách chế độ
có/không cache và phân biệt thời gian khởi tạo, camera/upload/API với thời gian service.
Không dùng tốc độ cache để đại diện cho ảnh mới hoặc coi cấu hình đề xuất là cấu hình
tối thiểu đã được chứng minh.

## 5. Đối chiếu bốn câu hỏi nghiên cứu

| Câu hỏi | Đã có | Còn thiếu để trả lời thuyết phục |
|---|---|---|
| Q1: Few-shot 2–3 ảnh và OOD | Pipeline known/unknown, novel 2/3-shot và các mốc đối chứng, metric OOD | Đối chiếu chính 2/3-shot, baseline và nhiều support seed; dữ liệu vật thể/phiên độc lập; đánh giá OOD sau khi danh mục thay đổi |
| Q2: Trade-off risk/coverage | Đường cong và gate khóa trên calibration | Kết quả đủ lớn, số lỗi/số trả lời, độ bất định; báo riêng known/unknown và chi phí chuyển nhân viên |
| Q3: Lợi ích policy so với image-only | Có công cụ đối chứng image-only/full-policy trên kịch bản hồ sơ | Kịch bản độc lập/lặp lại, hồ sơ thật, tình huống chọn sai lô còn hạn cùng SKU; không tuyên bố ảnh xác định được lô vật lý |
| Q4: Latency, consistency, audit | Benchmark, fingerprint, khóa thread/process, test DB/bank/audit | Danh mục đầy đủ; đo service và trải nghiệm đầu cuối; cơ chế recovery khi DB và file ghi lệch; neo hash độc lập nếu muốn phát hiện cắt đuôi log |

Novel few-shot hiện tắt Energy/risk gate của bank cũ khi thêm lớp. Đây là xử lý đúng để
không dùng ngưỡng không còn phù hợp, nhưng chưa chứng minh một hệ thống đồng thời giữ
khả năng few-shot và kiểm soát risk sau mỗi lần mở rộng danh mục.

## 6. Rủi ro kỹ thuật/nghiệp vụ cần trình bày khi bảo vệ

1. **Chứng từ demo chưa có tệp gốc thực tế.** `intake()` tạo URI tham chiếu theo loại chứng từ;
   checkbox verified là lời xác nhận của người thao tác. Chưa có upload, hash và quy trình kiểm chứng tệp gốc hoàn chỉnh.
2. **Đúng SKU không chứng minh đúng lô vật lý.** Khi hai lô cùng SKU đều có hồ sơ hợp lệ,
   cần quy trình POS/quầy để liên kết vật thể với lô; camera không giải quyết được điều này.
3. **Hash chain có giới hạn.** Phát hiện sửa/xóa giữa chuỗi; không tự phát hiện xóa đuôi
   hoặc viết lại toàn bộ khi chưa neo head bên ngoài.
4. **DB và file chưa là giao dịch nguyên tử chung.** Prototype JSON, SQLite và audit có thể
   lệch nếu ghi lỗi/mất điện giữa các bước. Khóa nhiều tiến trình không thay thế recovery/outbox.
5. **Chưa có xác thực/phân quyền cho thao tác ghi.** Phù hợp demo localhost; cần bổ sung nếu vận hành nhiều nhân viên.
6. **Nhập hàng có thể ghi một phần trước khi đăng ký ảnh thất bại.** Kiểm tra giá/ngày nay
   được thực hiện trước khi ghi, nhưng chưa biến toàn luồng intake thành giao dịch có rollback.

## 7. Giao diện đã cải thiện trong lượt này

- Tông xanh trầm, nền sáng, phân cấp chữ và thẻ nội dung thống nhất; bố cục thích nghi desktop/mobile.
- Quầy kiểm tra là tab đầu. Ảnh và dữ liệu ở bên trái; kết quả, lý do và hồ sơ ở bên phải.
- Bốn hành động vận hành có nội dung/màu riêng; không biến khoảng cách hoặc p-value thành phần trăm chắc chắn.
- Số SKU, lô và quyết định lấy từ service thật; trạng thái thiếu hiệu chỉnh/thiếu danh mục hiển thị rõ.
- Lô được lọc theo SKU. Thay ảnh/mã/lô/giá/chế độ làm kết quả cũ mất hiệu lực trên màn hình.
- Có làm mới lượt kiểm tra, cập nhật danh mục và hướng dẫn demo. Khi đăng ký/nhập lô, danh sách ở quầy được cập nhật.
- Form nhập lô chia theo mặt hàng, nhà cung cấp, nguồn gốc/thời hạn và chứng từ; giá/ngày được kiểm tra trước khi ghi.
- Chứng từ mới mặc định chưa xác minh; JSON và các thao tác ít dùng nằm trong accordion.
- Đăng ký có kiểm tra chia thành vùng nhập ảnh và vùng xem mẫu/kết quả; chỉ hiện SKU hàng khác khi cần.
- Tương thích cách truyền theme/CSS của Gradio 4/5 và Gradio 6; không cần tải font từ Internet.

Cần kiểm tra lại giao diện trên desktop/mobile và các phiên bản thư viện thực tế.
Thử thiếu ảnh, đổi SKU/lô, thiếu calibration, nhập chứng từ, thu hồi lô và xem audit.
Mọi kịch bản ghi dữ liệu nên dùng DB/audit riêng. Kiểm tra giao diện không thay thế
đánh giá hiệu quả nhận dạng hoặc thử nghiệm tại quầy.

## 8. Thứ tự hoàn thiện trước bảo vệ

1. **Ưu tiên cao nhất: chất lượng giao thức dữ liệu.** Bổ sung metadata vật thể/phiên,
   nhận diện ảnh gần trùng và khóa split mới cho thí nghiệm chính. Lưu lại dataset/manifest cũ
   để truy nguyên; không sửa âm thầm rồi so sánh với kết quả cũ.
2. **Thu test POS thật có nhãn.** Túi/lưới/độ chín, ánh sáng, background và hard pairs
   khác giống/SKU; nguồn gốc phải có hồ sơ xác minh độc lập.
3. **Chạy ma trận thí nghiệm đủ Q1–Q3.** Classical, DINOv2 raw một/nhiều prototype,
   head nếu sử dụng; 1/2/3/5-shot, nhiều seed; distance/Energy; image-only/full-policy.
4. **Báo risk/coverage đúng gate.** Khóa ngưỡng trước test, báo số trả lời/số lỗi và
   độ bất định; ghi cả những điều kiện không có câu trả lời. Không điều chỉnh bằng test.
5. **Đo Q4 trên cấu hình mục tiêu.** Danh mục đầy đủ, ảnh chưa cache, nhánh lô/chứng từ
   thực tế, p50/p95/p99 và số lượt đủ lớn; tách model startup, camera/upload khỏi service.
6. **Hoàn thiện luận văn và kịch bản bảo vệ.** Công thức nhất quán, nghiên cứu liên quan,
   chương pháp lý/nghiệp vụ, ERD bảy bảng, kết quả đối chứng, thất bại và giới hạn.

**Đủ tốt làm nền tảng khóa luận: có. Đủ bằng chứng nói “đã đạt tất cả yêu cầu của thầy”: chưa.**
Phần thiếu quan trọng hiện là dữ liệu/giao thức và chứng minh thực nghiệm, không phải thêm nhiều tính năng giao diện.
