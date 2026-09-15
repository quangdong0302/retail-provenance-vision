# Đăng ký mặt hàng có kiểm tra nhầm lẫn

Mở ứng dụng theo [hướng dẫn chạy](DEPLOYMENT.md), rồi chọn tab **Đăng ký có kiểm tra**.
Cần encoder/bank khớp nhau; không cần train lại head cho mỗi mặt hàng mới.

## Quy trình sử dụng

1. Nhập SKU và tên hiển thị, bấm **Tạo hồ sơ**. Hồ sơ nháp chưa thay đổi danh
   mục đang sử dụng. Có thể mở lại hồ sơ trong danh sách sau khi khởi động lại.
2. Chọn **Ảnh mẫu để đăng ký**, nhập mã nhóm vật thể/phiên và thêm ảnh. Bắt đầu
   với 3–5 ảnh từ ít nhất hai nhóm. Cùng vật thể phải giữ cùng mã nhóm, kể cả
   khi đổi góc hoặc ngày chụp; không đặt mã khác chỉ để vượt điều kiện.
3. Bấm **Kiểm tra nhầm lẫn**. Giao diện hiển thị tối đa ba SKU gần nhất và ảnh
   tham chiếu khi tìm được trong dataset/hồ sơ đã lưu. Đây là khoảng cách thị
   giác; chưa có ảnh kiểm tra thì không gọi nó là tỷ lệ nhầm thực tế.
4. Thêm **Ảnh kiểm tra đúng mặt hàng** từ các vật thể/nhóm độc lập với support.
   Thêm **Ảnh kiểm tra mặt hàng khác**, nhập SKU thật của ảnh đó. Ưu tiên các
   SKU dễ nhầm mà hệ thống liệt kê. Không chọn ảnh prototype đang dùng làm
   ảnh kiểm tra; không đưa ảnh test khóa luận vào luồng cải tiến này.
5. Chạy kiểm tra, xem lỗi và hướng dẫn. Có thể bỏ ảnh support không phù hợp,
   thêm support từ góc chuẩn/mẫu vật khác, rồi kiểm tra lại trên cùng bộ ảnh
   kiểm tra. Lịch sử lưu mọi lượt, chỉ so sánh trực tiếp khi tập kiểm tra và
   danh mục không đổi. Bổ sung validation/negative sẽ làm mất điều kiện so
   sánh trực tiếp với lượt trước.
6. Bấm **Đăng ký vào danh mục — cần nhân viên xác nhận** khi muốn xuất bản.
   Cần ít nhất ba support đạt kiểm tra chất lượng và báo cáo còn khớp phiên
   bản ảnh/danh mục. Có thể xuất bản ở chế độ nhân viên xác nhận khi chưa đủ
   kiểm định; giao diện không tự cấp quyền bán tự động.

Ảnh support có thể bị loại vì kích thước quá nhỏ, quá sáng/tối, ít chi tiết
hoặc có dấu hiệu mờ. Ảnh trùng nội dung/điểm ảnh hoặc rất gần nhau theo thumbnail
bị từ chối. Bộ lọc có thể từ chối nhầm ảnh hợp lệ; đây là heuristic chẩn đoán.
Không có bộ phát hiện vật thể để đo vật thể chiếm bao nhiêu phần trăm khung ảnh.

Không xóa validation/negative để che ảnh khó. Khi nhập nhầm nhãn kiểm tra,
tạo hồ sơ mới, giữ hồ sơ cũ làm bằng chứng. Ảnh support đã bỏ vẫn được nhớ để
ngăn đưa lại vào vai trò kiểm tra.

## Ý nghĩa trạng thái

| Trạng thái | Ý nghĩa |
|---|---|
| `needs_samples` | Chưa đủ số ảnh hoặc nhóm độc lập theo khai báo |
| `needs_review` | Đủ lượng mẫu nhưng còn lỗi/margin nhỏ/thiếu SKU đối chứng khó |
| `validation_passed` | Đạt tiêu chí kiểm tra phát triển nội bộ trên ảnh đã cung cấp |

Mọi trạng thái đều có `operating_mode=operator_required`. `validation_passed`
không phải chứng nhận chất lượng và không bật tự động bán.

Tiêu chí mặc định, được chụp lại trong từng hồ sơ:

- Support tối thiểu 3; validation tối thiểu 10; negative tối thiểu 10.
- Mỗi vai trò tối thiểu 2 nhóm; tối đa 40 ảnh/vai trò.
- Accuracy top-1 validation ≥90%; tỷ lệ negative bị gán vào SKU mới ≤5%.
- Ít nhất 90% validation có khoảng cách tới SKU khác trừ khoảng cách tới SKU
  mới ≥0,05; phải có negative khai đúng SKU cho các mặt hàng gần nhất đã liệt kê.
- Ảnh tối đa 15 MB và 24 triệu pixel; cạnh ngắn tối thiểu 96 pixel.
- Các con số là cấu hình thử nghiệm, chưa được tối ưu hoặc xác nhận ngoài
  thực tế. Ít mẫu và nhóm tự khai không cho bảo đảm thống kê về tỷ lệ lỗi.

Few-shot ở đây mô tả **số ảnh dựng prototype**. Ảnh để kiểm định là ngân sách
riêng; không quảng bá rằng chỉ ba ảnh tổng cộng đã chứng minh mô hình tốt.

`negative_assignment_rate` đo ảnh SKU khác có top-1 là SKU mới, không phải FAR
của hệ thống có ngưỡng từ chối. Kiểm tra lặp để sửa support là quy trình phát
triển; cần một tập test độc lập chưa dùng trong quy trình để báo cáo khóa luận.

## Khi xuất bản và khi khởi động lại

- Trước khi ghi bank, lưu bản sao tại `var/thesis/bank_backups/`; vị trí cụ thể
  được lưu trong trường `publication.bank_backup` của hồ sơ.
- Prototype mới chỉ được dựng từ support, không từ validation/negative.
- Ngưỡng cũ bị vô hiệu hóa. **Toàn bộ danh mục chuyển sang yêu cầu nhân viên
  xác nhận cho tới khi có hiệu chỉnh phù hợp**. Mặt hàng đăng ký qua luồng này
  vẫn mang cờ yêu cầu nhân viên kể cả khi nạp một calibration mới.
- Khi khởi động lại, calibration có fingerprint không khớp bank sẽ không được
  sử dụng. CLI/API/demo tiếp tục mở được, thay vì lỗi vì calibration cũ.
- Bank mới lưu format v3 để code cũ không bỏ qua cờ nhân viên xác nhận.
  Code hiện tại vẫn đọc được bank v2 đã train trước đó.
- Ứng dụng kiểm tra phiên bản file bank để nhận thay đổi từ tiến trình khác.
  Khi đăng ký, nếu bank trên đĩa đã đổi ngoài phiên hiện tại, yêu cầu kiểm tra
  lại; nên dùng một tiến trình ghi danh mục trong lúc đăng ký.
- Lệnh `calibrate` hiện dành cho dataset thực nghiệm: dựng lại bank từ dataset.
  **Không dùng lệnh đó để hiệu chỉnh danh mục có mặt hàng đăng ký mới**, vì SKU
  không nằm trong dataset có thể bị bỏ khỏi bank. Tính năng hiện tại chưa có
  quy trình cấp quyền tự động cho SKU mới.
- Sau đăng ký, mở tab nhập hàng, nhập đúng SKU và khai hồ sơ lô. Có thể để
  trống ảnh: với SKU đã đăng ký có kiểm tra, tab này giữ nguyên prototype.
  Sang tab quầy, bấm **Tải lại mặt hàng và lô** để cập nhật lựa chọn.

Các checkpoint và báo cáo thực nghiệm cũ không bị tính lại hay sửa. Luồng
kiểm thử tính năng dùng bản sao bank hoặc danh mục tạm.

## API và CLI

API Swagger tại `/docs`, nhóm `enrollment`:

| Phương thức | Đường dẫn | Công dụng |
|---|---|---|
| POST | `/enrollment` | Tạo hồ sơ: sku_id, label, operator |
| GET | `/enrollment` | Danh sách hồ sơ |
| GET | `/enrollment/{id}` | Chi tiết, lịch sử, phiên bản kiểm tra |
| POST | `/enrollment/{id}/images` | Multipart: role, group_id, files, negative_sku nếu là hàng khác |
| GET | `/enrollment/{id}/images/{sample_id}` | Ảnh đã lưu |
| DELETE | `/enrollment/{id}/support/{sample_id}` | Bỏ support khỏi phiên hiện tại |
| POST | `/enrollment/{id}/assess` | Kiểm tra nhầm lẫn |
| POST | `/enrollment/{id}/publish` | Đăng ký ở chế độ nhân viên xác nhận |

CLI hỗ trợ cùng quy trình:

```bash
python -m rpv.cli enroll create --sku tao_moi --label "Táo mới"
python -m rpv.cli enroll list
python -m rpv.cli enroll add --session ID --role support --group qua_a --images anh1.jpg anh2.jpg
python -m rpv.cli enroll add --session ID --role validation --group qua_b --images kiemtra1.jpg
python -m rpv.cli enroll add --session ID --role negative --group qua_c --negative-sku SKU_KHAC --images hangkhac.jpg
python -m rpv.cli enroll assess --session ID
python -m rpv.cli enroll show --session ID
python -m rpv.cli enroll publish --session ID
```

Thay ID và các đường dẫn bằng giá trị thật. Ảnh upload được lưu dưới
`var/thesis/enrollment/images`; metadata, đặc trưng, cấu hình và lịch sử nằm
trong `var/thesis/enrollment/enrollment.sqlite` (hoặc thư mục enrollment cạnh
bank khi đổi đường dẫn). Không cần giữ file tạm của trình duyệt/API.
