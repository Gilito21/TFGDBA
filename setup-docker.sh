#!/bin/bash
# Setup script for Docker-based COLMAP and TFGDBA on Lambda Cloud
# This script will:
# 1. Detect the current GPU architecture
# 2. Install Docker and NVIDIA Container Toolkit
# 3. Set up environment for Docker container
# 4. Build and run the Docker container

set -e  # Exit on any error

echo "=== Lambda Cloud TFGDBA Docker Setup ==="
echo "  Starting setup process..."

# Check if running as root, if not, use sudo
if [ "$EUID" -ne 0 ]; then
  USE_SUDO=sudo
else
  USE_SUDO=""
fi

# Create the GPU detection script
echo "=== Creating GPU detection script ==="
cat > detect_cuda_arch.sh << 'EOF'
#!/bin/bash
# Script to detect the CUDA compute capability (architecture) of the installed GPU

# Make sure nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi command not found - defaulting to architecture 86" >&2
    echo "86"
    exit 0
fi

# Try to extract the CUDA compute capability from nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [ -z "$GPU_NAME" ]; then
    echo "No GPU detected or nvidia-smi failed - defaulting to architecture 86" >&2
    echo "86"
    exit 0
fi

echo "Detected GPU: $GPU_NAME" >&2

# Match the GPU model with its architecture
if [[ "$GPU_NAME" == *"A100"* ]]; then
    # A100 - Ampere
    echo "80"
elif [[ "$GPU_NAME" == *"A10"* ]] || [[ "$GPU_NAME" == *"A40"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
    # A10, A40, A6000 - Ampere
    echo "86"
elif [[ "$GPU_NAME" == *"H100"* ]]; then
    # H100 - Hopper
    echo "90"
elif [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"3080"* ]] || [[ "$GPU_NAME" == *"3070"* ]] || 
     [[ "$GPU_NAME" == *"3060"* ]] || [[ "$GPU_NAME" == *"3050"* ]]; then
    # RTX 30 series - Ampere
    echo "86"
elif [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"4080"* ]] || [[ "$GPU_NAME" == *"4070"* ]] || 
     [[ "$GPU_NAME" == *"4060"* ]]; then
    # RTX 40 series - Ada Lovelace
    echo "89"
elif [[ "$GPU_NAME" == *"2080"* ]] || [[ "$GPU_NAME" == *"2070"* ]] || 
     [[ "$GPU_NAME" == *"2060"* ]] || [[ "$GPU_NAME" == *"1660"* ]]; then
    # RTX 20 series, GTX 16 series - Turing
    echo "75"
elif [[ "$GPU_NAME" == *"1080"* ]] || [[ "$GPU_NAME" == *"1070"* ]] || 
     [[ "$GPU_NAME" == *"1060"* ]] || [[ "$GPU_NAME" == *"1050"* ]]; then
    # GTX 10 series - Pascal
    echo "61"
elif [[ "$GPU_NAME" == *"TITAN V"* ]] || [[ "$GPU_NAME" == *"V100"* ]]; then
    # Titan V, V100 - Volta
    echo "70"
else
    # Default to a widely supported architecture if we can't identify the GPU
    echo "Warning: Could not identify GPU architecture for $GPU_NAME - using default (86)" >&2
    echo "86"
fi
EOF
chmod +x detect_cuda_arch.sh

# Run the detection script
echo "=== Detecting GPU architecture ==="
export CUDA_ARCH=$(./detect_cuda_arch.sh)
echo "Detected CUDA architecture: $CUDA_ARCH"

# Install Docker and NVIDIA Container Toolkit if not already installed
echo "=== Installing Docker and NVIDIA Container Toolkit ==="
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    $USE_SUDO apt-get update
    $USE_SUDO apt-get install -y docker.io
else
    echo "Docker already installed."
fi

# Install Docker Compose correctly
echo "=== Installing Docker Compose ==="
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    # Download the Docker Compose binary directly
    COMPOSE_VERSION=v2.18.1
    $USE_SUDO curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    $USE_SUDO chmod +x /usr/local/bin/docker-compose
    $USE_SUDO ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    # Verify docker-compose is working
    docker-compose --version || {
      echo "Docker Compose installation failed. Trying alternative method..."
      # Try alternative installation with pip (fallback)
      $USE_SUDO apt-get install -y python3-pip
      $USE_SUDO pip3 install docker-compose
    }
else
    echo "Docker Compose already installed."
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | $USE_SUDO apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | $USE_SUDO tee /etc/apt/sources.list.d/nvidia-docker.list
    $USE_SUDO apt-get update
    $USE_SUDO apt-get install -y nvidia-container-toolkit
    $USE_SUDO systemctl restart docker
else
    echo "NVIDIA Container Toolkit already installed."
fi

# Create docker-compose.yml file with the detected CUDA architecture
echo "=== Creating docker-compose.yml ==="
cat > docker-compose.yml << EOF
version: '3'

services:
  web:
    build:
      context: .
      args:
        - CUDA_ARCH=$CUDA_ARCH
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
      - ./frames:/app/frames
      - ./models:/app/models
      - ./colmap_workspace:/app/colmap_workspace
    restart: always
    environment:
      - MONGO_URI=mongodb+srv://juanp:myUC0QU4AxAZAGp0@cluster0.iiks7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
      - PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/colmap/build/src/colmap/exe
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

# Create Dockerfile with environment variable for COLMAP path
echo "=== Creating Dockerfile ==="
cat > Dockerfile << EOF
FROM ubuntu:22.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Accept CUDA architecture as build argument
ARG CUDA_ARCH=86
ENV CUDA_ARCH=\${CUDA_ARCH}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    cmake \\
    ninja-build \\
    build-essential \\
    libboost-program-options-dev \\
    libboost-filesystem-dev \\
    libboost-graph-dev \\
    libboost-system-dev \\
    libeigen3-dev \\
    libsuitesparse-dev \\
    libfreeimage-dev \\
    libgoogle-glog-dev \\
    libgflags-dev \\
    libglew-dev \\
    qtbase5-dev \\
    libqt5opengl5-dev \\
    libcgal-dev \\
    libcgal-qt5-dev \\
    libatlas-base-dev \\
    libsuitesparse-dev \\
    libceres-dev \\
    python3-pip \\
    python3-dev \\
    python3-opencv \\
    ffmpeg \\
    libavcodec-dev \\
    libavformat-dev \\
    libswscale-dev \\
    libavutil-dev \\
    wget \\
    unzip \\
    curl \\
    nvidia-cuda-toolkit \\
    nvidia-cuda-toolkit-gcc \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Install COLMAP from source with the specified architecture
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git && \\
    cd colmap && \\
    git checkout dev && \\
    mkdir build && \\
    cd build && \\
    echo "Building COLMAP with CUDA architecture: \${CUDA_ARCH}" && \\
    cmake .. -GNinja \\
      -DCMAKE_BUILD_TYPE=Release \\
      -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \\
      -DCMAKE_CUDA_ARCHITECTURES=\${CUDA_ARCH} && \\
    ninja && \\
    ninja install

# Add COLMAP to PATH
ENV PATH="/opt/colmap/build/src/colmap/exe:\${PATH}"

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
EOF

# Create or update requirements.txt with necessary packages
if [ ! -f requirements.txt ]; then
    echo "=== Creating requirements.txt ==="
    cat > requirements.txt << EOF
flask>=2.0.0
opencv-python>=4.5.0
numpy>=1.20.0
matplotlib>=3.4.0
pymongo>=4.0.0
pycolmap>=0.3.0
torch>=1.10.0
trimesh>=3.9.0
open3d>=0.13.0
scikit-learn>=0.24.0
scipy>=1.7.0
pandas>=1.3.0
Pillow>=8.3.0
EOF
fi

# Create necessary directories
echo "=== Creating directories ==="
mkdir -p uploads frames models colmap_workspace

# Create empty app.py if it doesn't exist (to avoid docker build errors)
if [ ! -f app.py ]; then
    echo "Note: Creating placeholder app.py - you'll need to replace this with your actual app code."
    cat > app.py << EOF
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "TFGDBA is being set up. Replace this with your actual app.py file."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF
fi

# Create .dockerignore
echo "=== Creating .dockerignore ==="
cat > .dockerignore << EOF
# Git
.git
.gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Data directories (will be mounted as volumes)
uploads/
frames/
models/
colmap_workspace/

# Environment
.env
.venv
env/
venv/
ENV/

# Editor configs
.idea/
.vscode/
*.swp
*.swo

# OS specific
.DS_Store
Thumbs.db
EOF

# Function to run docker-compose commands with fallback to docker compose (newer syntax)
docker_compose() {
    if command -v docker-compose &> /dev/null; then
        $USE_SUDO docker-compose "$@"
    else
        echo "docker-compose not found, trying 'docker compose' command..."
        $USE_SUDO docker compose "$@"
    fi
}

# Build the Docker container
echo "=== Building Docker container ==="
echo "This may take a while..."
docker_compose build

# Run the Docker container
echo "=== Starting Docker container ==="
docker_compose up -d

# Show logs
echo "=== Container logs ==="
docker_compose logs

echo "=== Setup complete ==="
echo "Your Docker container is now running."
echo "Access the web application at http://localhost:5000"
echo ""
echo "You can check the logs with:"
echo "  sudo docker-compose logs -f  (or 'sudo docker compose logs -f' if using newer Docker)"
echo ""
echo "To stop the container:"
echo "  sudo docker-compose down  (or 'sudo docker compose down' if using newer Docker)"
echo ""
echo "COLMAP is available at the path /opt/colmap/build/src/colmap/exe inside the container"
echo "and has been added to the PATH environment variable."
