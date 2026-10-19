# Cơ sở pháp lý và nghiệp vụ bán lẻ tại Việt Nam

Đây là phần động lực nghiên cứu, trình bày riêng với thuật toán thị giác
trong luận văn. Phạm vi nghiên cứu là nông sản tươi sống bán trực tiếp,
không có bao bì thương phẩm; tại POS có thể chưa có mã vạch riêng cho từng
vật thể nhưng cửa hàng vẫn quản lý mã SKU và hồ sơ lô nội bộ.

Khoản 2 Điều 1 Nghị định 43/2017/NĐ-CP có trường hợp thực phẩm tươi, sống,
thực phẩm chế biến không bao bì bán trực tiếp cho người tiêu dùng nằm ngoài
phạm vi điều chỉnh về nhãn của nghị định. Nghị định 111/2021/NĐ-CP sửa đổi
Nghị định 43. Trình bày đúng phạm vi này làm động lực nghiên cứu; không suy
ra rằng mọi thực phẩm tươi sống đều được miễn nhãn.
[Nghị định 43, nguồn quốc gia](https://vbpl.vn/nganhangnhanuoc/Pages/vbpq-toanvan.aspx?ItemID=122041),
[Nghị định 111, nguồn quốc gia](https://vbpl.vn/bolaodong/Pages/vbpq-toanvan.aspx?ItemID=160297&Keyword=).

Thông tư 02/2024/TT-BKHCN là nguồn tham khảo về quản lý truy xuất nguồn
gốc sản phẩm, hàng hóa. Thiết kế dữ liệu lưu sản phẩm/SKU, nhà cung cấp,
lô, chứng từ và sự kiện để có thể đối chiếu khi bán và truy ngược hồ sơ.
Danh sách `required_evidence` hiện là cấu hình nghiệp vụ nghiên cứu, chưa
phải bộ luật tự động kết luận tuân thủ cho mọi nhóm hàng.
[Thông tư 02, nguồn quốc gia](https://vbpl.vn/TW/Pages/vbpq-thuoctinh.aspx?ItemID=166271).

## Luồng nghiệp vụ cần khảo sát

1. Nhập hàng: nhân viên tạo/đối chiếu SKU, ghi nhà cung cấp, lô, ngày sản
   xuất/nhận/hết hạn, thông tin xuất xứ và chứng từ. Trạng thái verified
   phải do người có trách nhiệm đối chiếu, không do camera quyết định.
2. Đăng ký thị giác: chụp ít ảnh mẫu cho SKU; kiểm tra ảnh kém chất lượng,
   cặp SKU dễ nhầm và yêu cầu bổ sung ảnh khi chưa đủ phân biệt.
3. POS: camera cung cấp ảnh một mặt hàng; POS cung cấp SKU và lô khai báo;
   model có quyền từ chối; policy đối chiếu với hồ sơ độc lập.
4. Ngoại lệ: sai SKU, chưa nhận diện, thiếu hồ sơ, chứng từ hết hiệu lực,
   lô hết hạn/thu hồi/sai SKU đều có mã lý do và nhân viên xử lý.
5. Truy vết: lưu quyết định, hash ảnh, phiên bản encoder/calibration,
   lô được đối chiếu, đầu vào quyết định và audit liên kết hash.

Ảnh nhận đúng SKU không xác định được lô vật lý. Hai lô cùng SKU có thể có
cùng hình thức và khác vùng trồng/hạn sử dụng. Luận văn cần mô tả cách POS
chọn lô từ quy trình nhận hàng/quầy bán; hệ thống hiện chỉ kiểm chứng bản
ghi lô được khai, chưa chứng minh vật thể thực tế thuộc lô đó.

Khảo sát cửa hàng phải bổ sung bằng chứng về quy trình hiện tại, thời gian
thao tác, lỗi chọn SKU/lô và chi phí bổ sung ảnh. Đánh giá kịch bản mô phỏng
không thay thế khảo sát nhu cầu hoặc đánh giá hiệu quả ngoài quầy.
