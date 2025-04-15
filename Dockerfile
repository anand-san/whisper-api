FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    WHISPER_MODEL="base"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl --fail http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=2", "--timeout=300", "app:app"]
