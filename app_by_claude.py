import os
import cv2
import pymongo
import numpy as np
import tempfile
import datetime
import pycolmap
from pycolmap import (
    SiftExtractionOptions,
    CameraMode,
    Device
)
import torch
import shutil
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify, Response, send_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base64 import b64encode

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
MODEL_FOLDER = "models"
COLMAP_WORKSPACE = "colmap_workspace"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(COLMAP_WORKSPACE, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connect to MongoDB using environment variable
MONGO_URI = "mongodb+srv://juanp:myUC0QU4AxAZAGp0@cluster0.iiks7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["video_frames"]
frames_collection = db["frames"]
models_collection = db["models"]

model_creation_progress = {
    "current_step": 0,
    "total_steps": 4,  # Feature extraction, matching, mapping, export
    "step_name": "Initializing",
    "percent_complete": 0,
    "is_complete": False,
    "error": None,
    "model_path": None
}

# Check for GPU availability
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

def extract_frames(video_path, output_folder, frame_interval=5):
    """Extract frames from a video file using CUDA acceleration and save to MongoDB"""
    # Check if CUDA is available
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("CUDA is not available, falling back to CPU processing")
        use_cuda = False
    else:
        print(f"CUDA is available with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        use_cuda = True
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0
    
    # Create CUDA stream if CUDA is available
    if use_cuda:
        stream = cv2.cuda_Stream()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            if use_cuda:
                # Upload frame to GPU memory
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Process frame on GPU (optional processing like resizing or color conversion)
                # For example: gpu_frame = cv2.cuda.resize(gpu_frame, (new_width, new_height))
                
                # Download processed frame back to CPU
                frame = gpu_frame.download()
            
            # Save the frame
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Save frame data in MongoDB
            with open(frame_path, "rb") as f:
                frame_data = f.read()
                frames_collection.insert_one({"filename": frame_filename, "data": frame_data})
            
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    return extracted_count

@app.route('/')
def upload_form():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>3D Reconstruction from Video</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; margin-bottom: 20px; }
            .description { color: #4a5568; margin-bottom: 25px; }
            input[type="file"] { margin: 15px 0; }
            input[type="submit"], button { background: #4299e1; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; margin: 8px; font-weight: 600; transition: all 0.3s ease; }
            input[type="submit"]:hover, button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            select { padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; margin: 10px 0; width: 200px; }
            .options { margin: 20px 0; }
            .gpu-status { display: inline-block; padding: 8px 12px; border-radius: 6px; font-size: 14px; margin-top: 20px; }
            .gpu-active { background-color: #c6f6d5; color: #276749; }
            .gpu-inactive { background-color: #fed7d7; color: #9b2c2c; }
            .buttons { display: flex; justify-content: center; gap: 10px; margin-top: 25px; }
            .model-btn { background: #48bb78; }
            .model-btn:hover { background: #38a169; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>3D Reconstruction from Video</h1>
            <p class="description">Upload a video to extract frames and create a 3D model using COLMAP.</p>
            
            <div class="gpu-status {{ 'gpu-active' if gpu_available else 'gpu-inactive' }}">
                {{ 'GPU Acceleration: Active ✓' if gpu_available else 'GPU Acceleration: Inactive ✗' }}
            </div>
            
            <form action="/process_video" method="post" enctype="multipart/form-data">
                <div class="options">
                    <input type="file" name="video" required><br>
                    <select name="frame_interval">
                        <option value="1">Every frame</option>
                        <option value="5" selected>Every 5th frame</option>
                        <option value="10">Every 10th frame</option>
                        <option value="20">Every 20th frame</option>
                    </select>
                </div>
                <input type="submit" value="Extract Frames">
            </form>
            
            <div class="buttons">
                <a href="/frames"><button>View Extracted Frames</button></a>
                <a href="/models"><button class="model-btn">View 3D Models</button></a>
            </div>
        </div>
    </body>
    </html>
    ''', gpu_available=USE_GPU)

@app.route('/model_progress')
def get_model_progress():
    """Return the current progress of model creation as JSON"""
    global model_creation_progress
    return jsonify(model_creation_progress)



@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected file. Please upload a video file."}), 400
    
    # Get frame interval
    frame_interval = int(request.form.get('frame_interval', 5))
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    try:
        extracted_count = extract_frames(video_path, FRAME_FOLDER, frame_interval)
        
        # List the images in FRAME_FOLDER
        frame_files = os.listdir(FRAME_FOLDER)
        frame_files = sorted([f for f in frame_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        # If more than 15 images, select evenly distributed samples
        preview_frames = []
        if len(frame_files) > 15:
            step = len(frame_files) // 15
            preview_frames = [frame_files[i] for i in range(0, len(frame_files), step)][:15]
        else:
            preview_frames = frame_files
        
        # Generate preview URLs so the template can <img> them
        # We'll define a route /frame_local/<filename> to serve files from disk
        preview_urls = [f"/frame_local/{frame}" for frame in preview_frames]
        
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the video: {str(e)}"}), 500
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Processing Complete</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; }
            .success-icon { font-size: 60px; color: #48bb78; margin: 20px 0; }
            p { color: #4a5568; margin-bottom: 25px; }
            button { display: inline-block; padding: 12px 20px; font-size: 16px; color: white; background: #4299e1; text-decoration: none; border-radius: 8px; margin: 8px; border: none; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
            button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .create-model-btn { background: #48bb78; }
            .create-model-btn:hover { background: #38a169; }
            .delete-btn { background: #e53e3e; }
            .delete-btn:hover { background: #c53030; }
            .preview-container { display: flex; flex-wrap: wrap; justify-content: center; margin: 20px 0; }
            .preview-image { width: 120px; height: 90px; object-fit: cover; margin: 5px; border-radius: 4px; border: 1px solid #e2e8f0; }
            .more-indicator { font-style: italic; color: #718096; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="success-icon">✓</div>
            <h1>Frames Extracted Successfully</h1>
            <p>Extracted {{ count }} frames from your video.</p>
            
            <h3>Preview of Extracted Frames</h3>
            <div class="preview-container">
                {% for url in preview_urls %}
                <img src="{{ url }}" class="preview-image" alt="Frame preview">
                {% endfor %}
            </div>
            {% if count > 15 %}
            <div class="more-indicator">Showing 15 of {{ count }} frames</div>
            {% endif %}
            
            <div>
                <a href="/frames"><button>View All Frames</button></a>
                <a href="/create_model"><button class="create-model-btn">Create 3D Model</button></a>
                <button class="delete-btn" onclick="deleteFrames()">Delete Frames</button>
            </div>
        </div>
        
        <script>
        function deleteFrames() {
            if (confirm('Are you sure you want to delete all extracted frames? This cannot be undone.')) {
                fetch('/delete_mongo_frames', {
                    method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Frames deleted successfully');
                        // Redirect to home page
                        window.location.href = '/';
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while deleting frames');
                });
            }
        }
        </script>
    </body>
    </html>
    ''', count=extracted_count, preview_urls=preview_urls)

#
# Route to serve frames from disk for preview
#
@app.route('/frame_local/<filename>')
def frame_local(filename):
    """
    Serve a frame image from disk (FRAME_FOLDER) so that the browser
    can render it in the <img> tag. 
    """
    path = os.path.join(FRAME_FOLDER, filename)
    # or validate if path exists, etc.
    return send_file(path, mimetype='image/jpeg')

def get_frame_data_from_mongo():
    """Retrieve all frame filenames from MongoDB"""
    frames = list(frames_collection.find({}, {"filename": 1, "_id": 0}))
    return [frame["filename"] for frame in frames]

@app.route('/frames')
def list_frames():
    """List all stored frames with option to create 3D model"""
    try:
        # 1) Get filenames from Mongo
        frame_list = get_frame_data_from_mongo()  # e.g. ['frame_0000.jpg', 'frame_0005.jpg', ...]

        if not frame_list:
            return render_template_string('''
            <!doctype html>
            <html>
            <head>
                <title>No Frames Available</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                    .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                    h1 { color: #2d3748; }
                    .empty-icon { font-size: 60px; color: #a0aec0; margin: 20px 0; }
                    p { color: #4a5568; margin-bottom: 25px; }
                    button { display: inline-block; padding: 12px 20px; font-size: 16px; color: white; background: #4299e1; text-decoration: none; border-radius: 8px; margin: 8px; border: none; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
                    button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="empty-icon">📁</div>
                    <h1>No Frames Available</h1>
                    <p>You need to upload and process a video first.</p>
                    <a href="/"><button>Back to Upload</button></a>
                </div>
            </body>
            </html>
            ''')

        total_frames = len(frame_list)

        # 2) Pick up to 20 frames (for performance)
        display_frames = []
        if total_frames > 20:
            step = total_frames // 20
            selected_filenames = [frame_list[i] for i in range(0, total_frames, step)][:20]
        else:
            selected_filenames = frame_list

        # 3) For each chosen filename, fetch the image from MongoDB and Base64-encode it
        for filename in selected_filenames:
            doc = frames_collection.find_one({"filename": filename})
            if doc and "data" in doc:
                # Encode image data in Base64 so the browser can display it inline
                b64_data = b64encode(doc["data"]).decode('utf-8')
                display_frames.append({
                    "filename": filename,
                    "image_data": b64_data
                })

        # 4) Render them in the HTML
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Extracted Frames</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 900px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #2d3748; }
                p { color: #4a5568; margin-bottom: 25px; }
                .frame-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .frame-item { width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; background-color: #ffffff; }
                .frame-img { width: 100%; height: 150px; object-fit: cover; display: block; }
                .frame-name { font-size: 12px; color: #718096; padding: 8px; text-align: center; background-color: #f8f9fa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                .button-container { margin-top: 30px; }
                button { display: inline-block; padding: 12px 20px; font-size: 16px; color: white; background: #4299e1; text-decoration: none; border-radius: 8px; margin: 8px; border: none; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
                button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .create-model-btn { background: #48bb78; }
                .create-model-btn:hover { background: #38a169; }
                .delete-btn { background: #e53e3e; }
                .delete-btn:hover { background: #c53030; }
                .more-indicator { font-style: italic; color: #718096; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Extracted Frames</h1>
                <p>These frames were extracted from your uploaded video.</p>
                
                <div class="frame-grid">
                    {% for frame in frames %}
                    <div class="frame-item">
                        <!-- Inline base64-encoded image -->
                        <img src="data:image/jpeg;base64,{{ frame.image_data }}" 
                             class="frame-img" alt="{{ frame.filename }}">
                        <div class="frame-name">{{ frame.filename }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                {% if total > 20 %}
                <div class="more-indicator">Showing {{ frames|length }} of {{ total }} frames</div>
                {% endif %}
                
                <div class="button-container">
                    <a href="/"><button>Back to Upload</button></a>
                    <a href="/create_model"><button class="create-model-btn">Create 3D Model</button></a>
                    <button class="delete-btn" onclick="deleteFrames()">Delete All Frames</button>
                </div>
            </div>
            
            <script>
            function deleteFrames() {
                if (confirm('Are you sure you want to delete all frames from the database? This cannot be undone.')) {
                    fetch('/delete_mongo_frames', {
                        method: 'POST',
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('All frames have been deleted from the database');
                            window.location.href = '/';
                        } else {
                            alert('Error: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred while deleting frames');
                    });
                }
            }
            </script>
        </body>
        </html>
        ''', frames=display_frames, total=total_frames)

    except Exception as e:
        app.logger.error(f"Error in list_frames route: {str(e)}")
        return f"Error loading frames: {str(e)}", 500


# Add a new route to handle MongoDB frame deletion
@app.route('/delete_mongo_frames', methods=['POST'])
def delete_mongo_frames():
    try:
        # Connect to MongoDB and delete all frames
        # This assumes you have a MongoDB collection for frames
        db = client["video_frames"]  # Replace with your actual database name
        collection = db["frames"]      # Replace with your actual collection name
        
        # Delete all documents in the collection
        result = collection.delete_many({})
        
        return jsonify({
            "success": True, 
            "deleted_count": result.deleted_count
        })
    except Exception as e:
        app.logger.error(f"Error deleting frames from MongoDB: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

import subprocess
import os
from pathlib import Path
import shutil

def run_colmap_reconstruction(workspace_dir: Path, video_id: str = None):
    """Run COLMAP reconstruction using the custom CUDA-enabled build and update progress."""
    global model_creation_progress
    
    try:
        database_path = workspace_dir / "database.db"
        images_path = workspace_dir / "images"
        sparse_path = workspace_dir / "sparse"
        
        # Clean up old files
        if database_path.exists():
            os.remove(database_path)
        if sparse_path.exists():
            shutil.rmtree(sparse_path)
        sparse_path.mkdir(parents=True, exist_ok=True)
        
        # Path to your custom COLMAP build
        colmap_exe = "/home/ubuntu/TFGDBA/colmap/build/src/colmap/exe/colmap"
        
        # Initialize progress
        model_creation_progress["current_step"] = 1
        model_creation_progress["step_name"] = "Feature Extraction"
        model_creation_progress["percent_complete"] = 25
        model_creation_progress["is_complete"] = False
        
        # 1. Feature extraction (with GPU)
        print("Running feature extraction...")
        cmd = [
            colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.gpu_index", "0"
        ]
        subprocess.run(cmd, check=True)
        
        # Update progress
        model_creation_progress["current_step"] = 2
        model_creation_progress["step_name"] = "Feature Matching"
        model_creation_progress["percent_complete"] = 50
        
        # 2. Exhaustive matching (with GPU)
        print("Running exhaustive feature matching...")
        cmd = [
            colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.gpu_index", "0"
        ]
        subprocess.run(cmd, check=True)
        
        # Update progress
        model_creation_progress["current_step"] = 3
        model_creation_progress["step_name"] = "3D Reconstruction"
        model_creation_progress["percent_complete"] = 75
        
        # 3. Mapper
        print("Running incremental mapping...")
        cmd = [
            colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--output_path", str(sparse_path)
        ]
        subprocess.run(cmd, check=True)
        
        # Update progress
        model_creation_progress["current_step"] = 4
        model_creation_progress["step_name"] = "Exporting Model"
        model_creation_progress["percent_complete"] = 90
        
        # Export to PLY
        model_path = workspace_dir / "model.ply"
        model_folder = list(sparse_path.glob("*"))[0]  # Get the first reconstruction
        cmd = [
            colmap_exe, "model_converter",
            "--input_path", str(model_folder),
            "--output_path", str(model_path),
            "--output_type", "PLY"
        ]
        subprocess.run(cmd, check=True)
        
        # Save model to persistent storage
        timestamp = datetime.datetime.now().isoformat()
        model_file_name = f"model_{timestamp.replace(':', '-')}.ply"
        persistent_model_path = Path(MODEL_FOLDER) / model_file_name
        shutil.copy(model_path, persistent_model_path)
        
        # Save model metadata to MongoDB
        model_document = {
            "filename": model_file_name,
            "path": str(persistent_model_path),
            "created_at": timestamp,
            "video_id": video_id,
            "status": "completed",
            "reconstruction_data": {
                "num_images": len(list(images_path.glob("*.*"))),
                "workspace_path": str(workspace_dir)
            }
        }
        
        # Insert into MongoDB
        model_id = models_collection.insert_one(model_document).inserted_id
        
        # Update progress with model information
        model_creation_progress["percent_complete"] = 100
        model_creation_progress["is_complete"] = True
        model_creation_progress["model_path"] = str(persistent_model_path)
        model_creation_progress["model_id"] = str(model_id)
        
        print(f"Exported model to {persistent_model_path} and saved to database with ID: {model_id}")
        return persistent_model_path, model_id
        
    except Exception as e:
        # Update progress with error
        model_creation_progress["error"] = str(e)
        print(f"Error in reconstruction: {str(e)}")
        raise e

@app.route('/create_model_progress')
def create_model_progress():
    """Show a page with progress bar for model creation"""
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Creating 3D Model</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; margin-bottom: 20px; }
            .progress-container { margin: 40px 0; }
            .progress-bar {
                width: 100%;
                background-color: #e2e8f0;
                border-radius: 10px;
                height: 20px;
                position: relative;
                margin-bottom: 10px;
            }
            .progress-fill {
                height: 100%;
                background-color: #4299e1;
                border-radius: 10px;
                transition: width 0.5s;
                width: 0%;
                position: absolute;
                left: 0;
            }
            .step-name {
                font-size: 16px;
                color: #4a5568;
                margin: 15px 0;
                font-weight: 600;
            }
            .percent {
                font-size: 14px;
                color: #718096;
            }
            .loading-spinner {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                color: #e53e3e;
                background-color: #fed7d7;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                text-align: left;
            }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Creating 3D Model</h1>
            <p>Please wait while we process your frames and create a 3D model with COLMAP.</p>
            
            <div class="loading-spinner" id="spinner"></div>
            
            <div class="progress-container">
                <div class="step-name" id="stepName">Initializing...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="percent" id="percentText">0%</div>
            </div>
            
            <div id="errorContainer" class="error hidden">
                <strong>Error:</strong> <span id="errorText"></span>
            </div>
        </div>
        
        <script>
            // Poll for progress updates every 1 second
            const progressFill = document.getElementById('progressFill');
            const percentText = document.getElementById('percentText');
            const stepName = document.getElementById('stepName');
            const spinner = document.getElementById('spinner');
            const errorContainer = document.getElementById('errorContainer');
            const errorText = document.getElementById('errorText');
            
            let isComplete = false;
            
            function updateProgress() {
                if (isComplete) return;
                
                fetch('/model_progress')
                    .then(response => response.json())
                    .then(data => {
                        // Update the progress bar
                        progressFill.style.width = `${data.percent_complete}%`;
                        percentText.textContent = `${data.percent_complete}%`;
                        stepName.textContent = data.step_name;
                        
                        // Check for completion or error
                        if (data.error) {
                            isComplete = true;
                            spinner.classList.add('hidden');
                            errorContainer.classList.remove('hidden');
                            errorText.textContent = data.error;
                        } else if (data.is_complete) {
                            isComplete = true;
                            spinner.classList.add('hidden');
                            // Redirect to the model view page
                            setTimeout(() => {
                                window.location.href = '/models';
                            }, 1000);
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching progress:', error);
                    });
            }
            
            // Update progress immediately and then every second
            updateProgress();
            const intervalId = setInterval(updateProgress, 1000);
            
            // Clean up interval on page unload
            window.addEventListener('beforeunload', function() {
                clearInterval(intervalId);
            });
        </script>
    </body>
    </html>
    ''')

def prepare_colmap_workspace():
    """Prepare the COLMAP workspace by copying frames from MongoDB."""
    workspace_dir = Path(COLMAP_WORKSPACE)
    images_dir = workspace_dir / "images"
    
    # Clean existing files
    if images_dir.exists():
        shutil.rmtree(images_dir)
    
    images_dir.mkdir(exist_ok=True, parents=True)
    
    # Retrieve frames from MongoDB and save them to the workspace
    frame_list = get_frame_data_from_mongo()
    frame_paths = []
    
    for frame_name in frame_list:
        frame_data = frames_collection.find_one({"filename": frame_name})
        if frame_data:
            frame_path = images_dir / frame_name
            with open(frame_path, "wb") as f:
                f.write(frame_data["data"])
            frame_paths.append(str(frame_path))
    
    return workspace_dir, frame_paths

@app.route('/create_model')
def create_model():
    """Start the 3D model creation process in a background thread and redirect to progress page"""
    global model_creation_progress
    
    frame_list = get_frame_data_from_mongo()
    
    if not frame_list or len(frame_list) < 5:
        return jsonify({"error": "Need at least 5 frames to create a good 3D model"}), 400
    
    # Reset the progress tracker
    model_creation_progress = {
        "current_step": 0,
        "total_steps": 4,
        "step_name": "Initializing",
        "percent_complete": 0,
        "is_complete": False,
        "error": None,
        "model_path": None
    }
    
    # Start the model creation process in a background thread
    import threading
    
    def create_model_thread():
        try:
            # Prepare the workspace
            workspace_dir, frame_paths = prepare_colmap_workspace()
            
            if len(frame_paths) < 5:
                model_creation_progress["error"] = "Failed to load at least 5 valid frames"
                return
            
            # Run COLMAP reconstruction
            model_path = run_colmap_reconstruction(workspace_dir)
            
            # Create a visualization from the resulting PLY
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Load and visualize the point cloud
            point_cloud = np.loadtxt(model_path, skiprows=15, usecols=(0, 1, 2, 3, 4, 5))
            points = point_cloud[:, :3]
            colors = point_cloud[:, 3:] / 255.0  # Normalize RGB values
            
            # Plot points with their colors
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('COLMAP 3D Reconstruction')
            
            # Save the visualization
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            visualization_path = os.path.join(MODEL_FOLDER, f"colmap_model_{timestamp}_viz.png")
            plt.savefig(visualization_path, dpi=200)
            plt.close()
            
            # Generate a unique name for the model
            model_name = f"colmap_model_{timestamp}.ply"
            model_output_path = os.path.join(MODEL_FOLDER, model_name)
            
            # Copy the model PLY to the models folder
            shutil.copy(model_path, model_output_path)
            
            # Read data for MongoDB
            with open(model_output_path, 'rb') as f:
                model_data = f.read()
            
            with open(visualization_path, 'rb') as f:
                viz_data = f.read()
            
            point_count = len(points)
            
            # Insert model record in MongoDB
            models_collection.insert_one({
                "name": model_name,
                "data": model_data,
                "visualization": viz_data,
                "frame_count": len(frame_paths),
                "point_count": point_count,
                "created_at": datetime.datetime.now(),
                "gpu_used": USE_GPU  # Fixed variable name
            })
            
        except Exception as e:
            app.logger.error(f"Error creating 3D model: {str(e)}")
            model_creation_progress["error"] = str(e)
    
    # Start the background thread
    thread = threading.Thread(target=create_model_thread)
    thread.daemon = True
    thread.start()
    
    # Redirect to the progress page
    return redirect('/create_model_progress')

@app.route('/models')
def list_models():
    """List all created 3D models"""
    model_list = list(models_collection.find({}, {"name": 1, "_id": 0, "created_at": 1, "frame_count": 1, "point_count": 1, "gpu_used": 1}))
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>3D Models</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 950px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; margin-bottom: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            th, td { padding: 15px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }
            th { background-color: #4299e1; color: white; font-weight: 600; }
            tr:hover { background-color: #f7fafc; }
            button { display: inline-block; padding: 12px 20px; font-size: 16px; color: white; background: #4299e1; text-decoration: none; border-radius: 8px; margin: 8px; border: none; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
            button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .no-models { padding: 30px; text-align: center; font-size: 18px; color: #718096; background: #edf2f7; border-radius: 8px; margin-top: 20px; }
            .thumbnail { width: 100px; height: 75px; object-fit: cover; border-radius: 4px; transition: all 0.3s ease; }
            .thumbnail:hover { transform: scale(1.2); box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
            .actions a { margin-right: 10px; color: #4299e1; text-decoration: none; font-weight: 500; }
            .actions a:hover { text-decoration: underline; }
            .main-actions { margin: 20px 0; }
            .gpu-badge { display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
            .gpu-active { background-color: #c6f6d5; color: #276749; }
            .gpu-inactive { background-color: #fed7d7; color: #9b2c2c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>3D Models</h1>
            
            <div class="main-actions">
                <a href="/"><button>Back to Upload</button></a>
                <a href="/frames"><button>View Frames</button></a>
            </div>
            
            {% if models %}
                <table>
                    <tr>
                        <th>Preview</th>
                        <th>Model Name</th>
                        <th>Created</th>
                        <th>Points</th>
                        <th>Frames</th>
                        <th>Processing</th>
                        <th>Actions</th>
                    </tr>
                    {% for model in models %}
                        <tr>
                            <td>
                                <a href="/model_viz/{{ model.name.split('.')[0] }}" target="_blank">
                                    <img class="thumbnail" src="/model_viz/{{ model.name.split('.')[0] }}" alt="Preview">
                                </a>
                            </td>
                            <td>{{ model.name }}</td>
                            <td>{{ model.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                            <td>{{ model.point_count }}</td>
                            <td>{{ model.frame_count }}</td>
                            <td>
                                <span class="gpu-badge {{ 'gpu-active' if model.gpu_used else 'gpu-inactive' }}">
                                    {{ 'GPU' if model.gpu_used else 'CPU' }}
                                </span>
                            </td>
                            <td class="actions">
                                <a href="/model/{{ model.name }}" download>Download</a>
                                <a href="/model_view/{{ model.name }}">View</a>
                            </td>
                        </tr>
                    {% endfor %}
                </table>
            {% else %}
                <div class="no-models">
                    <p>No 3D models have been created yet.</p>
                    <p>Go to the frames page to create a 3D model using COLMAP.</p>
                </div>
            {% endif %}
        </div>
    </body>
    </html>
    ''', models=model_list)

@app.route('/frame/<filename>')
def get_frame(filename):
    """Retrieve and display a specific frame"""
    frame_data = frames_collection.find_one({"filename": filename})
    
    if not frame_data:
        return "Frame not found", 404

    image_data = frame_data["data"]
    return Response(image_data, mimetype="image/jpeg")

@app.route('/model/<filename>')
def get_model(filename):
    """Download a specific 3D model"""
    model_data = models_collection.find_one({"name": filename})
    
    if not model_data:
        return "Model not found", 404

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(model_data["data"])
        tmp_file_path = tmp_file.name
        return send_file(tmp_file_path, as_attachment=True, download_name=filename)

@app.route('/model_viz/<filename>')
def get_model_viz(filename):
    """Retrieve and display the visualization of a specific 3D model"""
    model_viz_data = models_collection.find_one({"name": filename + ".ply"})
 
    if not model_viz_data:
        return "Model visualization not found", 404

    visualization_data = model_viz_data["visualization"]
    return Response(visualization_data, mimetype="image/png")
    
@app.template_filter('format_number')
def format_number(value):
    return f"{value:,}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
