FROM ubuntu:22.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Accept CUDA architecture as build argument
ARG CUDA_ARCH=86
ENV CUDA_ARCH=${CUDA_ARCH}

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libceres-dev \
    libflann-dev \
    libmetis-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    python3-pip \
    python3-dev \
    python3-opencv \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    wget \
    unzip \
    curl \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install FLANN from source if package doesn't work
RUN git clone https://github.com/flann-lib/flann.git /opt/flann && \
    cd /opt/flann && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_C_BINDINGS=ON -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOC=OFF && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Install COLMAP from source with the specified architecture
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout main && \
    mkdir build && \
    cd build && \
    echo "Building COLMAP with CUDA architecture: ${CUDA_ARCH}" && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DFLANN_INCLUDE_DIR=/usr/local/include \
      -DFLANN_LIBRARY=/usr/local/lib/libflann.so && \
    ninja && \
    ninja install

# Add COLMAP to PATH
ENV PATH="/opt/colmap/build/src/colmap/exe:${PATH}"

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/uploads /app/frames /app/models /app/colmap_workspace

# Copy requirements file
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Expose the port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Run the application
CMD ["python3", "app.py"]
