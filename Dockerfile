FROM python:3.10-slim

# Install system dependencies for Selenium, ChromeDriver, and lxml
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set MPLCONFIGDIR and XDG_CACHE_HOME for Matplotlib and Fontconfig
RUN mkdir -p /tmp/matplotlib /tmp/cache/fontconfig && chmod -R 777 /tmp/matplotlib /tmp/cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/cache
ENV DUCKDB_HOME=/tmp/duckdb
ENV DUCKDB_TEMP_DIR=/tmp/duckdb
ENV DUCKDB_EXTENSION_DIR=/tmp/duckdb/extensions

# Copy all other code
COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run FastAPI app with Uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
