FROM python:3.12

RUN apt-get update && apt-get install -y \
    libgsl-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app
ENV CORS_ORIGINS="http://localhost:5173,http://localhost:8501"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
