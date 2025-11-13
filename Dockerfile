# ===========================
# Blockflow v3.6 – Render Dockerfile
# ===========================

FROM python:3.10-slim

# Create app directory
WORKDIR /app

# Install OS compile deps
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 10000

# Start FastAPI for Render
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
