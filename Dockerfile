FROM python:3.10-slim-bullseye
# Force build: 2026-04-22 17:50

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set working directory to backend
WORKDIR /app/backend

# Create necessary directories and set permissions
RUN mkdir -p db data/photos static && chmod -R 777 db data/photos static

# Use Render's dynamic PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
