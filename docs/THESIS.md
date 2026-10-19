# Giao thức thực nghiệm và nội dung khóa luận

Thiết kế model: [kiến trúc](ARCHITECTURE.md). Lệnh thực thi: [huấn luyện và đánh giá](TRAINING.md).

Các bổ sung theo nhận xét 17/09/2026: [đối chiếu yêu cầu, công thức policy,
Energy và ngưỡng risk](REVIEW_CHANGES.md). Cơ sở pháp lý/nghiệp vụ trình bày
riêng tại [LEGAL_BUSINESS](LEGAL_BUSINESS.md); sơ đồ 7 bảng tại [DATABASE](DATABASE.md).

## Hai thư mục 70/30

```text
data/dataset_v2/
├── known/
│   └── apple_10/Apple 10/*.jpg
├── unknown/
│   └── almonds_1/unknown_Almonds 1/*.jpg
└── manifest.json
```

70/30 được tính **theo lớp**, không theo số ảnh. Làm tròn số lớp known về số
nguyên gần nhất, giữ tối thiểu hai lớp known và một lớp unknown. Số ảnh hai
phía có thể khác tỷ lệ 70/30 vì mỗi lớp có số ảnh khác nhau.

- Tên lớp được chuẩn hóa chữ thường, khoảng trắng thành `_`. Các tên tương
  đương như `Apple Braeburn 1` và `apple_braeburn_1` được ghép trước khi chia.
- Các lớp có tiền tố `unknown_` ở nguồn được giữ trong pool unknown; tiền tố
  được bỏ khỏi mã lớp mới vì membership đã có trong tên thư mục/manifest.
- Nếu số lớp unknown định sẵn vượt ngân sách unknown, importer báo lỗi để
  người dùng điều chỉnh tỷ lệ/dữ liệu, không âm thầm chuyển chúng vào train.
- Ảnh trùng SHA-256 cùng lớp chỉ giữ một bản trong dataset mới; manifest ghi
  cả đường dẫn đã bỏ và bản được giữ. Trùng nội dung nhưng khác nhãn gây lỗi.
- Thư mục nguồn luôn được giữ nguyên. Mỗi lần chuẩn bị tạo một thư mục đích
  mới; đích đã tồn tại bị từ chối. File ảnh, bản đồ alias, seed, nhóm chụp và
  hash được ghi vào manifest. Đọc dữ liệu phát hiện file thay đổi/thiếu/thêm.
- Thư mục không có ảnh hợp lệ không tạo thành một lớp thực nghiệm.

Tập nguồn đang dùng do người dùng cung cấp ở `D:/Documents/Back/images`. Không
suy ra xuất xứ sản phẩm hoặc giấy phép sử dụng ảnh từ tên lớp. Khi viết luận
văn cần xác nhận nguồn bộ ảnh, quyền sử dụng và cách định nghĩa mỗi nhãn.

## Các vai trò ảnh trong known

Trong mỗi lớp known, chia theo nhóm ảnh độc lập:

| Vai trò | Cách lấy | Dùng cho |
|---|---|---|
| validation | khoảng 15% nhóm, tối thiểu 2 | Chọn checkpoint head |
| calibration | khoảng 15% nhóm, tối thiểu 1 | Chọn ngưỡng sau khi khóa head |
| test | khoảng 20% nhóm, tối thiểu 1 | Báo cáo cuối |
| support | k nhóm từ phần còn lại, lấy 1 ảnh/nhóm | Dựng prototype |
| train | Các nhóm còn lại, khoảng 50% trừ support | Tối ưu tham số head |

Các vai trò không trùng nhau. Khi nhóm support có nhiều frame, các frame còn
lại trong nhóm đó được giữ ngoài train/calib/test để tránh rò rỉ. Lớp không đủ
nhóm bị báo lỗi rõ ràng, không âm thầm biến mất khỏi kết quả.

Số nhóm validation/calib/test do `RPV_CALIB_TEST_SEED` quyết định. Thay seed
support (`RPV_CLASS_SPLIT_SEED`) hoặc k không đổi các tập này. Khi head đã train,
phải dùng lại đúng manifest chia dữ liệu để hiệu chỉnh/đánh giá; head khác split
bị từ chối. Muốn đổi split phải train head tương ứng.

Không tạo thêm nhiều bản sao vật lý theo vai trò. `dataset-info --out ...`
xuất `splits.json` chứa đường dẫn tương đối đến ảnh trong hai pool.

## Unknown: hai thí nghiệm riêng

**A. Trước đăng ký:** bank chỉ chứa known. Toàn bộ ảnh unknown được dùng làm
test để đo khả năng từ chối. Unknown không tham gia train head, validation,
prototype known hay calibration. Ngưỡng được chọn từ dữ liệu known, không
được chỉnh bằng kết quả unknown-test.

**B. Sau đăng ký:** tạo bank thử nghiệm riêng, giữ known và bổ sung prototype
của các lớp unknown từ 1, 3 hoặc 5 ảnh mẫu. Các lớp này lúc đó được gọi là novel
đã đăng ký. Query loại toàn bộ các nhóm support dự trữ, dùng cùng tập query
cho cả ba mức k. Nếu một lớp thiếu support, mức k đó được báo skipped.

Hai thí nghiệm cố ý tái sử dụng pool unknown ở hai trạng thái danh mục khác
nhau; không coi chúng là hai tập dữ liệu độc lập. Không chỉnh head hoặc ngưỡng
sau khi quan sát A rồi báo B như một kết quả xác nhận độc lập.

Novel chỉ được cấp k support; không dùng thêm calib có nhãn của novel. Ngưỡng
pooled của known được chuyển sang novel và ghi rõ **không có bảo đảm coverage
theo từng lớp novel**. Báo cáo accuracy sau đăng ký là phép đo thực nghiệm.

## Nhóm vật thể/phiên chụp và giới hạn dữ liệu hiện có

Nếu có metadata, truyền CSV cho `prepare-data --groups-csv`:

```csv
path,group_id
Apple 10/r0_0.jpg,apple10_object01
Apple 10/r0_1.jpg,apple10_object01
```

CSV phải bao phủ mọi ảnh nguồn. `group_id` phải định danh nhóm vật thể/phiên
trong một lớp; nhóm dùng chung giữa nhiều lớp hiện bị từ chối vì cần một giao
thức chia nhóm liên lớp riêng. Cần đủ nhóm cho mọi vai trò.

Nếu chưa có metadata, mỗi nội dung ảnh là một nhóm. Cách này phát hiện trùng
tệp nhưng **không loại trừ các frame gần giống hoặc cùng một quả chụp liên
tiếp**. Báo cáo tự thêm cảnh báo. Các tên `r0_0.jpg`, `r0_1.jpg` chưa đủ bằng
chứng để tự gán danh tính vật thể. Trước bảo vệ cần bổ sung metadata đáng tin
hoặc một tập ảnh chụp độc lập để đánh giá khả năng tổng quát hóa.

## Phép đo và tính tái lập

- Coverage = tỷ lệ nhãn đúng nằm trong tập dự đoán. Tỷ lệ tập không rỗng được
  lưu dưới tên khác. Báo thêm kích thước tập, số singleton và accuracy singleton.
- Open-set mặc định dùng Energy từ logits −distance² cho cả known/unknown,
  ngưỡng chọn từ known-calib. `distance_baseline` giữ phép đo nearest-distance
  để đối chiếu. Gate thực tế kết hợp Energy, tập conformal singleton và
  ngưỡng selective risk; tỷ lệ từ chối cuối được báo riêng với ROC của score.
- Verifier FAR/FRR được báo riêng cho lời khai sai ngẫu nhiên và SKU sai gần
  nhất. FAR verifier không đồng nghĩa tỷ lệ policy tự động cho phép thanh toán.
- Risk–coverage xét tất cả các ngưỡng phân biệt và không chia đôi nhóm score
  bằng nhau. EER dùng tìm kiếm trên score đã sắp xếp để tránh chi phí bậc hai.
- Ngưỡng selective risk 1% được fit trên known-calibration và khóa trước
  test. Báo risk/coverage thực tế tại ngưỡng đó; điểm coverage-at-risk trên
  đường cong test là mô tả, không dùng để chọn ngưỡng triển khai. Báo cáo
  xuất CSV, SVG và PNG của đường cong để đưa vào luận văn.
- Score conformal được giữ đủ độ chính xác. Nếu không đủ mẫu cho thứ hạng
  yêu cầu, ngưỡng là vô hạn; serialization JSON giữ được ý nghĩa này khi đọc lại.
- Ngưỡng theo lớp chỉ dùng khi đủ `RPV_MIN_CALIB` ảnh calibration; các lớp
  còn lại dùng ngưỡng toàn cục. Khi báo cáo đóng góp này, phải nêu số lớp có
  ngưỡng riêng và số lớp fallback của chính lượt chạy, không suy từ cấu hình.
- Train/calibrate/evaluate chỉ nhận dataset có manifest known/unknown từ
  `prepare-data`. Ảnh gốc chưa chia không được tự động đưa vào split legacy.
  Tỷ lệ known/unknown được xác lập khi prepare-data, không chia lại lúc train.
- Cache/bank định danh cả trọng số backbone, preprocessing, head và TTA.
  Calibration liên kết fingerprint của prototype. Thay prototype đòi hỏi
  hiệu chỉnh lại; khi chưa có calibration, nhận diện không tự xác nhận chắc chắn.
- Mỗi `evaluate` tạo một run directory mới với split manifest, prototype,
  calibration, JSON/Markdown và dự đoán từng ảnh trong JSONL.
- `calibrate` chỉ trích embedding support/calibration; không dùng test để fit
  hay chọn ngưỡng (kiểm tra toàn vẹn vẫn đọc hash mọi ảnh). `evaluate`
  dựng lại bank/ngưỡng từ đúng support/calib trước khi đo test và lưu vào run.
- Kịch bản hồ sơ lô chỉ chạy khi truyền `--scenarios`; dùng SQLite/audit riêng
  của run và được hiểu là dữ liệu nghiệp vụ mô phỏng, không phải hồ sơ thật.

`scripts/thesis_experiments.py` chạy baseline frozen nhiều seed và tổng hợp
mean/std. Đây là độ biến thiên chọn support trên cùng test, không phải khoảng
tin cậy dân số. Để công bố khoảng tin cậy cần thêm phân tích theo vật thể/phiên
chụp và đủ mẫu unknown, đặc biệt khi tuyên bố FAR nhỏ.

## Nội dung và thứ tự thực nghiệm khóa luận

1. Định nghĩa rõ lớp hình ảnh, SKU và lô; mô tả dữ liệu, alias, ảnh trùng và nhóm chụp.
2. Khóa 70/30 lớp cùng các vai trò ảnh; công bố ngân sách support/calibration.
3. So sánh classical, DINOv2 raw một/nhiều prototype và DINOv2 + head.
4. So sánh pad/crop, k-shot, số prototype trên validation. Giữ test cho báo cáo.
5. Báo unknown trước đăng ký và novel 1/3/5-shot sau đăng ký; phân tích cặp khó.
6. Đánh giá đóng góp policy/hồ sơ bằng các kịch bản có ground truth, gồm hai lô
   cùng SKU. Nhận đúng SKU không chứng minh vật thể thuộc đúng lô.
7. Báo thời gian suy luận không cache, thời gian dùng cache, phần cứng, phiên
   bản thư viện và giới hạn chuyển từ bộ ảnh hiện tại sang quầy thực tế.

Đóng góp đề tài nằm ở giao thức, thiết kế và bằng chứng thực nghiệm trên bài
toán cụ thể. DINOv2, Prototypical Networks và conformal prediction đã có trong
tài liệu khoa học; không mô tả chúng là thuật toán mới do đề tài phát minh.

Nguồn lý thuyết: [DINOv2 chính thức](https://github.com/facebookresearch/dinov2),
[Prototypical Networks](https://arxiv.org/abs/1703.05175),
[Conformal prediction](https://arxiv.org/abs/2107.07511).


## Đánh giá tính năng đăng ký có kiểm tra

So sánh đăng ký trực tiếp với đăng ký có kiểm tra trên cùng encoder, danh mục
và ngân sách support. Ghi lại số ảnh yêu cầu bổ sung, số lượt kiểm tra, thời
gian thao tác và số lỗi gán nhầm trước/sau. Tập ảnh dùng để hướng dẫn bổ sung
là dữ liệu phát triển; cần tập test theo vật thể/phiên độc lập để kết luận.
Công bố cả trường hợp hệ thống không thể đưa ra cải thiện.

Tính năng hiện luôn yêu cầu nhân viên xác nhận khi vận hành. Không dùng tỷ lệ
đạt kiểm tra nội bộ làm tỷ lệ thanh toán tự động hoặc bảo đảm lỗi ngoài quầy.
Chi tiết ngân sách ảnh và điều kiện: [đăng ký mặt hàng](ENROLLMENT.md).

## Bố cục khóa luận

| Chương | Nội dung | Bằng chứng cần có |
|---|---|---|
| 1 | Bài toán, người dùng, phạm vi và câu hỏi nghiên cứu | Kịch bản quầy và nhu cầu được khảo sát |
| 2 | Cơ sở pháp lý và nghiệp vụ bán lẻ Việt Nam | Phạm vi NĐ 43/2017, NĐ 111/2021; truy xuất theo TT 02/2024; quy trình cửa hàng |
| 3 | Cơ sở lý thuyết và nghiên cứu liên quan | DINOv2, prototype, Energy OOD, conformal, selective risk; tài liệu gốc và giả định |
| 4 | Dữ liệu và giao thức | Nguồn ảnh, alias, nhóm chụp, manifest 70/30, test điều kiện túi/lưới/độ chín, hard negative |
| 5 | Hiện thực hệ thống | Decision(I,S,L), 4 trạng thái, sơ đồ 7 bảng, đăng ký có kiểm tra, policy và audit |
| 6 | Thực nghiệm theo Q1–Q4 | Baseline/head, Energy/distance, risk tại ngưỡng đã khóa, image-only/full-policy, latency POS |
| 7 | Thảo luận và kết luận | Cặp nhầm, giới hạn dữ liệu, lô vật lý, chi phí vận hành, risk/latency chưa đạt nếu có |

## Trả lời các phản biện chính

**Đóng góp của đề tài là gì nếu few-shot đã phổ biến?**

Không nhận DINOv2 hay Prototypical Networks là thuật toán mới. Phần cần bảo vệ
là thiết kế quy trình đăng ký có kiểm tra nhầm lẫn, tổ chức thực nghiệm tránh
rò rỉ, tích hợp hồ sơ lô và bằng chứng định lượng trên bài toán đã chọn. Có
code triển khai chưa đồng nghĩa đã chứng minh hiệu quả; kết luận phải dựa vào
các so sánh ở chương thực nghiệm.

**Vì sao người dùng chọn hệ thống này?**

Giả thuyết là người nhập danh mục cần biết mặt hàng mới dễ nhầm với ai và cần
thêm ảnh gì trước khi đưa vào sử dụng. Để khẳng định giá trị, cần khảo sát
người dùng và đo giảm lỗi/thời gian thao tác so với quy trình hiện có. Repo
chưa chứng minh nhu cầu thị trường hoặc việc người dùng sẵn sàng trả tiền.

**Ảnh có chứng minh nguồn gốc không?**

Ảnh hỗ trợ nhận diện SKU. Thông tin lô, nhà cung cấp và chứng từ được đối
chiếu riêng. Nhận đúng SKU không chứng minh vật thể thuộc đúng lô hay hồ sơ
được nhập là xác thực.

**Có thể dùng kết quả kiểm tra đăng ký làm accuracy khóa luận không?**

Không, nếu đã xem kết quả đó để chọn/bổ sung ảnh. Báo kết quả trên tập test
độc lập, nêu rõ ngân sách support và số mẫu kiểm định, giữ lại các lần thất bại.
