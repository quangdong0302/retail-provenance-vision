# Ảnh mặc định KÈM PyTorch CPU vì DINOv2 là encoder chính của hệ thống.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    RPV_ENCODER=dinov2 \
    RPV_DB=/data/rpv.db \
    RPV_BANK=/data/prototypes.json \
    RPV_CALIB=/data/calibration.json \
    RPV_AUDIT=/data/audit.jsonl \
    RPV_REPORTS=/data/reports \
    RPV_CACHE_DIR=/data/embed_cache \
    RPV_HEAD_OUT=/data/projection_head.pt \
    RPV_DATA_DIR=/data/dataset_v2 \
    TORCH_HOME=/data/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo zlib1g curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY rpv/ ./rpv/
COPY api/ ./api/
COPY app/ ./app/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY Makefile ./

# TORCH_HOME nằm trong volume /data: trọng số DINOv2 chỉ tải MỘT LẦN rồi được
# giữ lại qua các lần restart container. Không có bước này, mỗi lần khởi động
# lại là một lượt tải ~90MB và container không chạy nổi khi mất mạng.
RUN mkdir -p /data/torch && useradd -m rpv && chown -R rpv /app /data
USER rpv
VOLUME ["/data"]
EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
