FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8122

WORKDIR /app

# System deps:
# - git: needed for pip install from GitHub (tribev2)
# - ffmpeg: common runtime dependency for media extraction stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
RUN python -m spacy download en_core_web_lg

COPY app /app/app
COPY data /app/data

EXPOSE 8122

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8122"]
