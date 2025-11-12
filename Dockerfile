# Dockerfile
# Blockflow Exchange - FastAPI + PostgreSQL (Render-ready)

FROM python:3.11-slim-bullseye

# Working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

# Upgrade pip + wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
