"""RPV — Retail Provenance Vision.

Hệ thống nhận diện sản phẩm bán lẻ không mã vạch (few-shot, open-set) GẮN VỚI
lớp kiểm chứng nguồn gốc theo lô hàng.

Nguyên tắc kiến trúc (đọc trước khi sửa bất cứ thứ gì):

    Ảnh trả lời câu hỏi "đây là sản phẩm nào".
    Hồ sơ lô + chứng từ trả lời câu hỏi "nó đến từ đâu".
    Policy engine trả lời câu hỏi "hai câu trả lời trên có khớp nhau không".

Hệ thống KHÔNG BAO GIỜ suy đoán xuất xứ từ pixel. Mọi kết luận về nguồn gốc
đều phải truy về một `Evidence` có thể kiểm chứng, hoặc bị đánh dấu
`PROVENANCE_MISSING`.
"""

__version__ = "2.0.0"
__all__ = ["__version__"]
