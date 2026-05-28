FROM python:3.12-slim

# System dependencies for OpenCV + FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD gunicorn --worker-class gevent --workers 1 --timeout 600 --bind "0.0.0.0:$PORT" app:app
