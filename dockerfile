FROM ubuntu:20.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# Install CUDA toolkit
RUN apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

# Set CUDA environment variables
ENV CUDACXX=/usr/bin/nvcc

# Create a working directory
WORKDIR /app

# Clone your repo
RUN git clone https://github.com/Gilito21/TFGDBA.git /app/TFGDBA

# Set up Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip

# Install Python dependencies from your repo
WORKDIR /app/TFGDBA
RUN pip install -r requirements.txt

# Create a script to detect GPU architecture and build COLMAP
RUN echo '#!/bin/bash \n\
# Get GPU architecture from device \n\
if command -v nvidia-smi &> /dev/null; then \n\
  GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ".") \n\
  if [ -z "$GPU_ARCH" ]; then \n\
    # Fallback to a safe default if detection fails \n\
    echo "GPU architecture detection failed, using default value 75" \n\
    GPU_ARCH=75 \n\
  fi \n\
else \n\
  # If nvidia-smi is not available, use a default value \n\
  echo "nvidia-smi not found, using default GPU architecture value 75" \n\
  GPU_ARCH=75 \n\
fi \n\
echo "Detected GPU architecture: $GPU_ARCH" \n\
\n\
# Clone and build COLMAP \n\
cd /app \n\
git clone https://github.com/colmap/colmap.git \n\
cd /app/colmap \n\
mkdir -p build \n\
cd build \n\
\n\
# Build COLMAP with detected architecture \n\
cmake .. -GNinja \\\n\
  -DCMAKE_BUILD_TYPE=Release \\\n\
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \\\n\
  -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH \n\
\n\
# Compile COLMAP \n\
ninja \n\
\n\
# Add COLMAP to PATH \n\
echo "export PATH=/app/colmap/build/src/colmap/exe:$PATH" >> /app/venv/bin/activate \n\
' > /app/build_colmap.sh

# Make the script executable
RUN chmod +x /app/build_colmap.sh

# Create a startup script
RUN echo '#!/bin/bash \n\
# Run the build script if COLMAP is not already built \n\
if [ ! -d "/app/colmap/build" ]; then \n\
  /app/build_colmap.sh \n\
fi \n\
\n\
# Source the virtual environment \n\
source /app/venv/bin/activate \n\
\n\
# Run the application \n\
cd /app/TFGDBA \n\
python app.py \n\
' > /app/start.sh

# Make the startup script executable
RUN chmod +x /app/start.sh

# Return to your project directory
WORKDIR /app/TFGDBA

# Command to run when container starts
CMD ["/app/start.sh"]
