# Set up working directory
WORKDIR /app

# Copy requirements (create this file first)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads frames models

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
