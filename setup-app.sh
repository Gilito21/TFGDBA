#!/bin/bash

# Install Docker and NVIDIA Container Toolkit
echo "Installing Docker and NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y docker.io
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify NVIDIA drivers are installed
echo "Verifying NVIDIA drivers..."
nvidia-smi

# Pull the Docker image
echo "Pulling Docker image..."
sudo docker pull tiogilito21/tfgdba-app:latest

# Stop and remove any existing container with the same name
echo "Cleaning up any existing containers..."
sudo docker stop tfgdba-instance 2>/dev/null || true
sudo docker rm tfgdba-instance 2>/dev/null || true

# Run the container in detached mode
echo "Starting the application in detached mode..."
sudo docker run -d --gpus all -p 5000:5000 --name tfgdba-instance tiogilito21/tfgdba-app:latest

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
