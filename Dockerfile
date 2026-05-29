FROM python:3.12-slim

# System deps: OpenCV, FFmpeg, and a software Vulkan device (lavapipe, from
# mesa-vulkan-drivers) so RIFE's GPU compute shaders can run on the CPU.
# vulkan-tools provides `vulkaninfo` for debugging device detection.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    libgomp1 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download RIFE ncnn Linux binary (cached layer — only rebuilds if URL changes)
RUN mkdir -p /root/.video-repair/rife-ncnn && \
    curl -fsSL -o /tmp/rife.zip \
      "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip" && \
    unzip -q /tmp/rife.zip -d /tmp/rife_tmp && \
    find /tmp/rife_tmp -maxdepth 2 -name "rife-ncnn-vulkan" -not -type d \
      -exec cp {} /root/.video-repair/rife-ncnn/rife-ncnn-vulkan \; && \
    find /tmp/rife_tmp -maxdepth 2 -mindepth 2 -type d -name "rife*" \
      -exec cp -r {} /root/.video-repair/rife-ncnn/ \; && \
    chmod +x /root/.video-repair/rife-ncnn/rife-ncnn-vulkan && \
    rm -rf /tmp/rife.zip /tmp/rife_tmp

COPY . .

EXPOSE $PORT

CMD gunicorn --worker-class gevent --workers 1 --timeout 600 --bind "0.0.0.0:$PORT" app:app
