# ─────────────────────────────────────────────
# Blockflow Exchange - Render Deployment
# Version: 3.5 (Production)
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Upgrade pip and wheel for modern builds
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI via Gunicorn + Uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
