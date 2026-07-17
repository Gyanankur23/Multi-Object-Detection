FROM python:3.11-slim


WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies including latest ONNX runtime
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY data.yaml .
COPY templates/ ./templates/


# Create necessary directories
RUN mkdir -p runs/train/custom_model/weights static

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
