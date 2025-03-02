# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app_working_retrieve_frames_mongodb.py .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app_working_retrieve_frames_mongodb.py"]
