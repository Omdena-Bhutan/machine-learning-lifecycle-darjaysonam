FROM python:3.11-slim

WORKDIR /app

# Install API dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY app/api.py /app/api.py

EXPOSE 8000

# IMPORTANT: run Python directly, NOT flask CLI
CMD ["python", "/app/api.py"]
