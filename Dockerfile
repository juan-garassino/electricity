# Use a slim Python image
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# App workdir
WORKDIR /app

# Install deps first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your package (code + model)
COPY electricity/ ./electricity/

# (Optional) Document the port
EXPOSE 8080

# Start FastAPI; use Cloud Run's $PORT or 8080 locally
CMD ["sh","-c","uvicorn electricity.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
