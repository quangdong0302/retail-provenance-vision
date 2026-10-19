.PHONY: help install db warm train-head calibrate seed evaluate test lint api demo status all baseline

help:
	@echo "make install      - cài đầy đủ (DINOv2 + API + demo)"
	@echo "make db           - tạo lược đồ SQLite"
	@echo "make warm         - nhúng trước toàn bộ ảnh (chạy 1 lần, tăng tốc mọi bước sau)"
	@echo "make train-head   - fine-tune ProjectionHead trên đặc trưng DINOv2 (tùy chọn)"
	@echo "make calibrate    - đăng ký prototype + hiệu chỉnh ngưỡng"
	@echo "make evaluate     - đánh giá đầy đủ + xuất báo cáo"
	@echo "make baseline     - chạy lại đánh giá bằng encoder classical để so sánh"
	@echo "make test         - chạy test tự động (không cần GPU/mạng)"
	@echo "make lint         - kiểm tra mã nguồn bằng Ruff"
	@echo "make api          - chạy FastAPI (http://127.0.0.1:8000/docs)"
	@echo "make demo         - chạy demo Gradio (http://127.0.0.1:7860)"
	@echo "make all          - toàn bộ pipeline từ đầu"

install:
	python -m pip install -r requirements.txt

db:
	python -m rpv.cli init-db

warm:
	python -m rpv.cli warm-cache

train-head:
	python -m rpv.cli train-head

calibrate:
	python -m rpv.cli calibrate

seed:
	python -m rpv.cli seed-scenarios

evaluate:
	python -m rpv.cli evaluate

baseline:
	python -m rpv.cli evaluate --encoder classical --ckpt ""

status:
	python -m rpv.cli status

test:
	python -m unittest discover -s tests -t . -v

lint:
	python -m ruff check rpv api app scripts tests

api:
	python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

demo:
	python -m app.demo

all:
	python scripts/run_all.py --encoder dinov2 --train-head

