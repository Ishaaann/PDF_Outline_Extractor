# Dockerfile for Adobe Hackathon Round 1A PDF Outline Extractor
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY extract_lines.py .
COPY train_model.py .
COPY generate_outline.py .
COPY label_lines.csv .

# Copy pre-trained model files (if available)
COPY model.pkl* ./
COPY label_encoder.pkl* ./

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command to run the outline generator
CMD ["python", "generate_outline.py"]
