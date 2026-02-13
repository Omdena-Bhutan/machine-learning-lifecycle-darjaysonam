FROM python:3.11-slim
WORKDIR /app

# Copy requirements
COPY app/requirements.txt /app/requirements.txt

# Install lightweight packages first
RUN pip install --no-cache-dir --default-timeout=1000 \
    flask gunicorn numpy sentencepiece safetensors

# Install PyTorch CPU-only (much smaller - ~140MB vs 915MB)
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install transformers separately
RUN pip install --no-cache-dir --default-timeout=1000 transformers

# Copy API code
COPY app/api.py /app/api.py

EXPOSE 8000

CMD ["python", "/app/api.py"]
