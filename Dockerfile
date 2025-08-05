FROM python:3.11-slim

# Install system dependencies for Selenium, ChromeDriver, lxml, and Tesseract
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    tesseract-ocr \
    libpng-dev \
    libjpeg-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp/duckdb /tmp/duckdb/extensions /tmp/matplotlib /tmp/cache/fontconfig && \
    chown -R appuser:appuser /tmp/duckdb /tmp/matplotlib /tmp/cache && \
    chmod -R 777 /tmp/duckdb /tmp/matplotlib /tmp/cache

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Matplotlib, Fontconfig, DuckDB, and Chromium
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/cache
ENV DUCKDB_HOME=/tmp/duckdb
ENV DUCKDB_TEMP_DIR=/tmp/duckdb
ENV DUCKDB_EXTENSION_DIR=/tmp/duckdb/extensions
ENV CHROMIUM_PATH=/usr/bin/chromium
ENV PYTHONUNBUFFERED=1

# Copy all other code
COPY . .

# Switch to non-root user
USER appuser

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run FastAPI app with Uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
