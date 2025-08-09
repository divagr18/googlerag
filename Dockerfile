# Dockerfile (GPU Version - Corrected)

# --- Builder Stage ---
# CORRECTED: Use a valid and standard NVIDIA CUDA base image.
# This one includes CUDA 12.1.1 and cuDNN 8 on Ubuntu 22.04.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python 3.11 and its tools.
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    tesseract-ocr \
    aria2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy the uv binary for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.2.9 /uv /usr/local/bin/

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy your requirements file
COPY requirements.txt .

# Install Python dependencies using uv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# --- Final Stage ---
# CORRECTED: Use the matching runtime image.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS final

ENV DEBIAN_FRONTEND=noninteractive

# Install only the runtime system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    aria2 \
    python3.11 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy the virtual environment with installed packages from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY ./api /app/api

# Set the working directory
WORKDIR /app

# Make the virtual environment's binaries available
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
