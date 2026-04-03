# Dockerfile - Optimized for OpenEnv Project
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies (minimal for deployment)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy pydantic pyyaml gradio matplotlib python-dotenv

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/trained_models /app/logs

# Expose port for Gradio
EXPOSE 7860

# Set default environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py"]