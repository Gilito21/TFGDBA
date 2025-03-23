#!/bin/bash

# Install Docker and NVIDIA Container Toolkit
echo "Installing Docker and NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y docker.io git
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify NVIDIA drivers are installed
echo "Verifying NVIDIA drivers..."
nvidia-smi

# Clone or update the GitHub repository
echo "Getting the latest code from GitHub..."
if [ -d "TFGDBA" ]; then
    cd TFGDBA
    git pull
    cd ..
else
    git clone https://github.com/Gilito21/TFGDBA.git
fi

# Check architecture
ARCH=$(uname -m)
echo "Detected host architecture: $ARCH"

# Stop and remove any existing container with the same name
echo "Cleaning up any existing containers..."
sudo docker stop tfgdba-instance 2>/dev/null || true
sudo docker rm tfgdba-instance 2>/dev/null || true

# We'll use a hybrid approach:
# 1. Pull the official image as a base (for environment setup)
# 2. Mount the local repository as a volume to use the latest code

echo "Pulling base Docker image..."
sudo docker pull tiogilito21/tfgdba-app:latest

# Run the container with mounted code
echo "Starting the application with the latest code..."
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "Running on ARM architecture, using platform emulation..."
    sudo docker run -d --platform linux/amd64 --gpus all -p 0.0.0.0:5000:5000 \
        --name tfgdba-instance \
        -v $(pwd)/TFGDBA:/app/TFGDBA \
        --entrypoint "/bin/bash" \
        tiogilito21/tfgdba-app:latest \
        -c "cd /app/TFGDBA && python3 app.py"
else
    echo "Running on AMD64 architecture..."
    sudo docker run -d --gpus all -p 0.0.0.0:5000:5000 \
        --name tfgdba-instance \
        -v $(pwd)/TFGDBA:/app/TFGDBA \
        --entrypoint "/bin/bash" \
        tiogilito21/tfgdba-app:latest \
        -c "cd /app/TFGDBA && python3 app.py"
fi

# Install dependencies inside the container
echo "Installing required dependencies..."
sudo docker exec tfgdba-instance pip install open3d pymeshlab trimesh

# Restart the container to pick up new dependencies
echo "Restarting container to apply dependency changes..."
sudo docker restart tfgdba-instance

# Wait for container to initialize
echo "Waiting for the application to initialize..."
sleep 5

# Display container status
echo "Container status:"
sudo docker ps | grep tfgdba-instance

# Show application logs
echo "Application logs:"
sudo docker logs tfgdba-instance

# Display follow logs instruction
echo -e "\nYour application is now running at http://$(curl -s ifconfig.me):5000"
echo "To follow the logs in real-time, run: sudo docker logs -f tfgdba-instance"
echo "To stop the application, run: sudo docker stop tfgdba-instance"
echo "To update with the latest code, simply run this script again"

# Create a convenience update script
cat > update-app.sh << 'EOF'
#!/bin/bash
echo "Pulling latest code from GitHub..."
cd TFGDBA
git pull
cd ..

echo "Restarting container to apply changes..."
sudo docker restart tfgdba-instance

echo "Application updated and restarted!"
echo "Your application is running at http://$(curl -s ifconfig.me):5000"
EOF

chmod +x update-app.sh
echo -e "\nA convenience script has been created. To update your app in the future, just run: ./update-app.sh"
