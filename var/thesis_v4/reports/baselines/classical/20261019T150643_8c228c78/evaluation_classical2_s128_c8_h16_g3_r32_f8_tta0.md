# Báo cáo đánh giá hệ thống RPV

- Thời điểm: 2026-10-19T15:16:49+00:00
- Encoder: `classical2_s128_c8_h16_g3_r32_f8_tta0`
- Backbone: `n/a` · dim 146 · ảnh n/apx · thiết bị `n/a` · không dùng projection head
- k-shot: 2 · alpha: 0.1 · TTA: False
- Dữ liệu: `data\dataset_v2`
- Protocol: `known_unknown_70_30_v1` · dataset fingerprint: `dc420dd70c2db61562551644f187199d91f5bf7d179efabc5ae11bc231118e93`

## 1. Phân chia dữ liệu

| Hạng mục | Giá trị |
|---|---|
| protocol | known_unknown_70_30_v1 |
| n_train_images | 2878 |
| n_validation_images | 1887 |
| n_known_classes | 17 |
| n_train_classes | 17 |
| n_novel_classes | 7 |
| n_support_images | 34 |
| n_calib_images | 1915 |
| n_test_images | 1927 |
| n_unknown_calib | 0 |
| n_unknown_test | 4200 |
| n_novel_support | 35 |
| n_novel_query | 836 |

> Prototype dựng từ support; ngưỡng chọn trên calib; số liệu đo trên test. Lớp novel chưa từng tham gia học biểu diễn.

## 2. Nhận diện (closed-set, đo trên test)

| Metric | Giá trị |
|---|---|
| Accuracy toàn danh mục | 14.79% |
| Macro-F1 toàn danh mục | 0.1419 |
| Macro-F1 (gồm lớp 'unknown') | 0.0821 |
| Số ảnh test | 1927 |

## 3. Open-set (khả năng nói "không biết")

| Metric | Giá trị |
|---|---|
| Ngưỡng toàn cục (chọn trên calib) | 0.0973 |
| Known recall | 78.98% |
| Unknown reject rate | 21.00% |
| False accept rate (hàng lạ bị nhận) | 79.00% |
| AUROC | 0.4920 |
| Score OOD | energy |
| FPR @ 95% TPR | 96.17% |

## 4. Dự đoán có kiểm soát rủi ro (conformal / selective)

| Metric | Giá trị |
|---|---|
| Tỉ lệ hệ thống chủ động không kết luận | 100.00% |
| Coverage chứa nhãn đúng | 89.05% |
| Coverage chứa nhãn đúng sau gate OOD | 73.07% |
| Kích thước tập trung bình | 12.7353 |
| Accuracy của câu trả lời tự động sau các gate | n/a |
| Hàng lạ bị từ chối đúng | 21.00% |
| Hàng lạ được trả lời tự động | 0.00% |
| Hàng lạ chuyển nhân viên (OOD hoặc mơ hồ/risk) | 100.00% |
| Coverage tại rủi ro 1% | 0.00% |
| Coverage tại rủi ro 5% | 0.00% |
| Coverage tại rủi ro 10% | 0.00% |
| Phân bố trạng thái | `{'ambiguous': 1522, 'unknown': 405}` |

![Risk–Coverage](evaluation_classical2_s128_c8_h16_g3_r32_f8_tta0.risk_coverage.png)

Đường cong top-1 và điểm gate vận hành dùng hai quy tắc chọn lọc khác nhau; CSV lưu toàn bộ ngưỡng thực nghiệm.

### Ngưỡng vận hành khóa trên known-calibration

- Rủi ro mục tiêu: 4.00%; coverage known-test: 0.00%; risk known-test: n/a.
- Trên known + unknown: coverage 0.00%; risk n/a.
- Đạt mục tiêu trên known-test: False. Không có câu trả lời thì risk chưa xác định, không coi là đạt.
- Coverage tại 1/5/10% trên đường cong chỉ là mô tả test; không chọn lại ngưỡng vận hành từ test. Ràng buộc calibration là thực nghiệm, chưa bảo đảm risk dân số.

## 5. Kiểm chứng lời khai (ảnh + SKU khai báo)

| Metric | Giá trị |
|---|---|
| Số cặp thật / cặp giả mạo | 1927 / 1927 |
| FAR (chấp nhận lời khai sai) | 0.00% |
| FRR (từ chối lời khai đúng) | 100.00% |
| EER | 39.23% |
| AUROC | 0.6525 |
| FAR với SKU sai gần nhất | 0.00% |

FAR ở đây thuộc verifier, chưa phải tỷ lệ policy cho phép thanh toán.

## Nhận diện lớp unknown sau khi đăng ký few-shot

| k-shot | Số query | Accuracy | Macro-F1 |
|---|---|---|---|
| 1 | 836 | 3.83% | 0.0494 |
| 2 | 836 | 3.95% | 0.0583 |
| 3 | 836 | 4.67% | 0.0625 |
| 5 | 836 | 12.92% | 0.1360 |

Pooled known conformal calibration transferred to novel classes; no class-conditional guarantee for novel. Energy/risk gates disabled because the catalog changed.

## 6. Hiệu chỉnh

- Ngưỡng riêng theo lớp: 17 SKU; dùng ngưỡng toàn cục: 0 SKU.
- Số liệu trên calib: `{'known_recall_on_calib': 0.8, 'unknown_reject_rate_on_calib': 0.0, 'target_known_recall': 0.8, 'n_known_calib': 1915, 'n_unknown_calib': 0, 'risk_calibration': 'known', 'n_extra_development': 0}`
- CẢNH BÁO: Ngưỡng chỉ được chọn từ known-calib. Khả năng từ chối hàng ngoài danh mục phải được đánh giá trên unknown-test độc lập; không đưa unknown-test vào hiệu chỉnh.

## 8. Phân tích lỗi

Cặp nhầm lẫn hay gặp nhất:

| Nhãn thật | Nhãn dự đoán | Số ca |
|---|---|---|
| Cà chua hồng | Cà chua tim bò | 96 |
| Cà rốt | Ớt chuông vàng | 64 |
| Bắp ngô | Súp lơ trắng | 52 |
| Táo Ligol | Ớt chuông vàng | 45 |
| Bắp ngô | Táo Red Delicious | 42 |
| Cà chua hồng | Táo Red Delicious | 39 |
| Táo Granny Smith | Súp lơ trắng | 38 |
| Cà chua hồng | Ớt chuông vàng | 33 |
| Bắp ngô | Bắp cải trắng | 30 |
| Táo Granny Smith | Bắp cải trắng | 30 |

SKU có recall thấp nhất (ưu tiên bổ sung ảnh mẫu):

- `Cà rốt`: 0.00%
- `Táo Red Delicious`: 0.00%
- `Xoài`: 0.00%
- `Cà chua hồng`: 2.14%
- `Táo Ligol`: 4.08%
- `Bắp ngô`: 8.42%
- `Hành tây tím`: 10.47%
- `Cà chua tim bò`: 10.71%

## Cảnh báo về cách đọc số liệu

- Ngưỡng chỉ được chọn từ known-calib. Khả năng từ chối hàng ngoài danh mục phải được đánh giá trên unknown-test độc lập; không đưa unknown-test vào hiệu chỉnh.

## Giới hạn phải nêu trong luận văn

- Hệ thống KHÔNG suy đoán xuất xứ từ hình ảnh. Kết luận nguồn gốc luôn xuất phát từ hồ sơ lô và chứng từ; thiếu chứng từ thì trạng thái là "chưa xác minh".
- Số liệu trên dữ liệu mô phỏng chỉ để kiểm chứng pipeline. Kết quả đưa vào luận văn phải đo trên ảnh thật thu tại quầy, nhiều ngày và nhiều điều kiện sáng.
- Chưa xử lý chống giả mạo (ảnh in, ảnh trên màn hình điện thoại) và chưa xử lý nhiều vật thể trong một khung hình.
- Dữ liệu từ nguồn cộng đồng chỉ mang tính tham khảo, không có giá trị chứng từ.