# Use an Ubuntu base image with CUDA support.
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Python3, pip, virtualenv, build tools, OpenCV dependencies, etc.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy your project code into the container.
# This assumes that you've already cloned your repository locally.
COPY . /app

# Create and activate a virtual environment, upgrade pip, and install Python dependencies.
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# (Optional) If COLMAP isnt installed on the host and you want to build it in the container,

RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCUDA_ARCH_BIN="86" && \
    make -j$(nproc) && \
    cp ./src/colmap/exe/colmap /usr/local/bin/

# Set environment variable for COLMAP path if needed by your app.
# For example, if you built COLMAP above:
ENV COLMAP_PATH="/usr/local/bin/colmap"

# Expose the port your application listens on (adjust as necessary).
EXPOSE 5000

# Ensure that the container uses the virtual environment's Python.
ENV PATH="/app/venv/bin:$PATH"

# Define the command to run your app.
CMD ["python", "app.py"]
