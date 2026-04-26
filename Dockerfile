# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Копируем установленные зависимости из builder
COPY --from=builder /install /usr/local

# Копируем код приложения
COPY app/ ./app/
COPY scripts/ ./scripts/

# Переменные окружения по умолчанию
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
    MAX_NEW_TOKENS=256 \
    HF_TOKEN="" \
    PORT=8000

EXPOSE ${PORT}

# Health check — проверяем /health каждые 30 секунд
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Запуск сервера
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
