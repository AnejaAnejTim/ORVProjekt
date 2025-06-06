FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopencv-dev \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY req.txt ./
RUN pip install --upgrade pip && pip install -r req.txt

COPY . .

EXPOSE 5001
CMD ["python", "server.py"]
