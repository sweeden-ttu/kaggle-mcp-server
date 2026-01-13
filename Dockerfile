FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for full system
RUN pip install --no-cache-dir \
    beautifulsoup4 \
    lxml \
    requests \
    sqlalchemy \
    alembic

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage/local \
    /app/storage/network \
    /app/storage/cloud \
    /app/storage/github \
    /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port (if needed for web interface)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.fol_workbench.main"]
