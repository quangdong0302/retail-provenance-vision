"""RPV — Retail Provenance Vision.

Nhận diện sản phẩm không mã vạch bằng few-shot, open-set và đối chiếu hồ sơ lô.

Ảnh dùng để nhận diện sản phẩm. Hồ sơ lô và chứng từ cung cấp thông tin
nguồn gốc. Policy đối chiếu hai phần này để đưa ra quyết định.

Xuất xứ không được suy từ ảnh. Kết luận nguồn gốc dựa trên ``Evidence``
có thể kiểm chứng; trường hợp thiếu hồ sơ mang trạng thái ``PROVENANCE_MISSING``.
"""

__version__ = "2.0.0"
__all__ = ["__version__"]
