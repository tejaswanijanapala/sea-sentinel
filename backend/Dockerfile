# ==========================================
# STAGE 1: BUILDER
# ==========================================
FROM python:3.11-slim as builder

# Set the working directory
WORKDIR /app

# Install system build dependencies (if any are needed for compiling wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies into a separate directory for multi-stage copying
# We explicitly force the CPU-only version of PyTorch to save ~2 GB of CUDA binaries
# in the cloud deployment image. (GPU image requires a different Dockerfile profile).
RUN pip install --no-cache-dir --prefix=/install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==========================================
# STAGE 2: PRODUCTION RUNTIME
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime system dependencies (headless opencv doesn't need libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the application source code
COPY . .

# Ensure persistent_data directory is excluded via .dockerignore
# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV STORAGE_MODE=cloud
ENV OMP_NUM_THREADS=4

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
