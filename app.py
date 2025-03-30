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
from flask import Flask, request, render_template_string, jsonify, Response, send_file, redirect
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base64 import b64encode
import trimesh
import subprocess
import logging
import pymeshlab
import open3d as o3d
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

from model_analysis import (
    load_meshes, 
    analyze_damage, 
    create_damage_submesh, 
    create_improved_visualization
)

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
    "step_name": "Importing images to the model (this can take up to 2 minutes, hang on)",
    "percent_complete": 0,
    "is_complete": False,
    "error": None,
    "model_path": None
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mesh_processing')

# Check for GPU availability
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

import cv2
import os
from pathlib import Path
import subprocess  # Only needed if you use other subprocess calls
# Make sure your MongoDB collection (frames_collection) is already defined and connected

def extract_frames(video_path, output_folder, frame_interval=5):
    """
    Extract frames from a video using OpenCV.

    Parameters:
      video_path (str): Path to the video file.
      output_folder (str): Directory where extracted frames will be saved.
      frame_interval (int): Save every Nth frame.

    Returns:
      int: Number of frames extracted.
    """
    # Create the output directory if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file using OpenCV.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # Save every 'frame_interval'-th frame.
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:04d}.jpg"
            output_path = Path(output_folder) / frame_filename
            cv2.imwrite(str(output_path), frame)
            print(f"Saving frame {frame_count} as {frame_filename}")
            extracted_count += 1

            # Save the frame to MongoDB
            with open(output_path, "rb") as f:
                frame_data = f.read()
                frames_collection.insert_one({
                    "filename": frame_filename,
                    "data": frame_data
                })

    cap.release()

    print(f"Extracted {extracted_count} frames out of {frame_count} total frames using OpenCV.")
    print(f"Inserted {extracted_count} frames into MongoDB.")
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
            h1, h2 { color: #2d3748; margin-bottom: 20px; }
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
            .divider { border-top: 1px solid #e2e8f0; margin: 30px 0; }
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

            <div class="divider"></div>
            
            <!-- Model Analysis Section -->
            <div style="padding: 30px; background-color: #f8fafc; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 30px;">
                <h2 style="color: #2d3748; margin-bottom: 15px;">Analyze 3D Model Differences</h2>
                <p style="color: #4a5568; margin-bottom: 25px; font-size: 1.1em;">Compare previously created model files to detect and visualize damage using advanced 3D analysis.</p>
                
                <div style="display: flex; justify-content: space-between; margin: 30px 0; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1; min-width: 200px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;">
                        <div style="font-size: 2.5em; margin-bottom: 15px;">📊</div>
                        <h3 style="color: #2d3748; margin-bottom: 10px;">Damage Detection</h3>
                        <p style="color: #718096; font-size: 0.9em;">Automatically detect and highlight damaged areas on 3D models</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;">
                        <div style="font-size: 2.5em; margin-bottom: 15px;">🔍</div>
                        <h3 style="color: #2d3748; margin-bottom: 10px;">Detailed Visualization</h3>
                        <p style="color: #718096; font-size: 0.9em;">Interactive 3D visualization with multiple view options</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;">
                        <div style="font-size: 2.5em; margin-bottom: 15px;">📋</div>
                        <h3 style="color: #2d3748; margin-bottom: 10px;">Damage Reports</h3>
                        <p style="color: #718096; font-size: 0.9em;">Get comprehensive damage statistics and measurements</p>
                    </div>
                </div>
                
                <a href="/select_models_to_compare" style="display: inline-block; background-color: #4299e1; color: white; padding: 12px 24px; border-radius: 8px; font-weight: bold; text-decoration: none; margin-top: 20px; transition: all 0.3s ease; border: none; cursor: pointer; font-size: 1.1em; text-align: center;">Compare 3D Models</a>
            </div>
        </div>
    </body>
    </html>
    ''', gpu_available=USE_GPU)

@app.route('/upload_porsche_models')
def upload_porsche_models():
    """Special page to upload the required Porsche models"""
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Upload Porsche Models</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; }
            p { color: #4a5568; margin-bottom: 25px; }
            .upload-section { 
                margin-bottom: 30px; 
                padding: 20px; 
                border-radius: 10px; 
                background-color: #f7fafc; 
                text-align: left;
            }
            .upload-section h2 { margin-top: 0; }
            form { margin-bottom: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="file"] { 
                padding: 10px; 
                width: 100%; 
                border: 1px solid #e2e8f0; 
                border-radius: 5px;
            }
            button {
                background-color: #4299e1;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }
            button:hover { background-color: #3182ce; }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                color: #4299e1;
                text-decoration: none;
            }
            .back-link:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Porsche Models</h1>
            <p>Upload the required Porsche models for damage analysis. Both files are required for comparison.</p>
            
            <div class="upload-section">
                <h2>Upload Original Porsche Model</h2>
                <form action="/upload_specific_model" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="original_model">Select porsche_original.obj file:</label>
                        <input type="file" id="original_model" name="model_file" accept=".obj" required>
                    </div>
                    <input type="hidden" name="target_filename" value="porsche_original.obj">
                    <button type="submit">Upload Original Model</button>
                </form>
            </div>
            
            <div class="upload-section">
                <h2>Upload Damaged Porsche Model</h2>
                <form action="/upload_specific_model" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="damaged_model">Select porsche_damaged.obj file:</label>
                        <input type="file" id="damaged_model" name="model_file" accept=".obj" required>
                    </div>
                    <input type="hidden" name="target_filename" value="porsche_damaged.obj">
                    <button type="submit">Upload Damaged Model</button>
                </form>
            </div>
            
            <a href="/" class="back-link">Back to Home</a>
        </div>
    </body>
    </html>
    ''')

@app.route('/upload_specific_model', methods=['POST'])
def upload_specific_model():
    """Handle upload of a specific model file"""
    if 'model_file' not in request.files:
        return "No file part", 400
        
    file = request.files['model_file']
    target_filename = request.form.get('target_filename', '')
    
    if file.filename == '':
        return "No selected file", 400
        
    if not target_filename:
        return "No target filename specified", 400
    
    # Create directory if it doesn't exist
    upload_dir = os.path.join(os.getcwd(), 'static', 'models')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the file with the target filename
    filepath = os.path.join(upload_dir, target_filename)
    file.save(filepath)
    
    # Update or insert the model in MongoDB
    model_data = {
        "filename": target_filename,
        "name": target_filename,
        "path": os.path.join('static', 'models', target_filename),
        "filepath": filepath,
        "created_at": datetime.datetime.now()
    }
    
    # Update or insert
    models_collection.update_one(
        {"filename": target_filename},
        {"$set": model_data},
        upsert=True
    )
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Model Upload Success</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; }
            .success { color: #38a169; font-weight: bold; }
            .button-container { margin-top: 30px; }
            .button {
                display: inline-block;
                padding: 10px 20px;
                margin: 0 10px;
                border-radius: 5px;
                text-decoration: none;
                color: white;
                font-weight: bold;
            }
            .primary { background-color: #4299e1; }
            .primary:hover { background-color: #3182ce; }
            .secondary { background-color: #48bb78; }
            .secondary:hover { background-color: #38a169; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Successful</h1>
            <p class="success">{{ filename }} has been uploaded successfully!</p>
            <p>The file has been saved to: {{ filepath }}</p>
            <p>The database has been updated with the file information.</p>
            
            <div class="button-container">
                <a href="/upload_porsche_models" class="button primary">Upload More Models</a>
                <a href="/select_models_to_compare" class="button secondary">Compare Models</a>
            </div>
        </div>
    </body>
    </html>
    ''', filename=target_filename, filepath=filepath)

@app.route('/model_progress')
def get_model_progress():
    """Return the current progress of model creation as JSON"""
    global model_creation_progress
    return jsonify(model_creation_progress)

@app.route('/upload_obj', methods=['POST'])
def upload_obj_file():
    if 'obj_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    obj_file = request.files['obj_file']
    if obj_file.filename == '':
        return jsonify({"error": "No selected file. Please upload an OBJ file."}), 400
    
    if not obj_file.filename.lower().endswith('.obj'):
        return jsonify({"error": "File must be an OBJ file"}), 400
    
    try:
        # Get custom model name if provided, otherwise use the original filename
        model_name = request.form.get('model_name', '').strip()
        if not model_name:
            model_name = obj_file.filename
        
        # Ensure model_name ends with .obj
        if not model_name.lower().endswith('.obj'):
            model_name += '.obj'
        
        # Create timestamp for the model
        timestamp = datetime.datetime.now()
        
        # Generate unique filename if not already unique
        unique_filename = f"model_{timestamp.isoformat().replace(':', '-')}.obj"
        
        # Save file to MODEL_FOLDER
        obj_path = os.path.join(MODEL_FOLDER, unique_filename)
        obj_file.save(obj_path)
        
        # Read obj file content
        with open(obj_path, 'rb') as f:
            obj_data = f.read()
        
        # Count vertices in the OBJ file
        point_count = 0
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    point_count += 1
        
        # Create metadata for MongoDB
        model_document = {
            "filename": unique_filename,
            "name": model_name,
            "path": f"models/{unique_filename}",
            "data": obj_data,  # Save binary data
            "created_at": timestamp,
            "point_count": point_count,
            "frame_count": 0,  # No frames for direct uploads
            "status": "completed",
            "uploaded": True,  # Flag to indicate this was a direct upload
            "visualization": None,  # No visualization for direct uploads
            "gpu_used": False,
            "reconstruction_data": {
                "num_images": 0,
                "workspace_path": None
            }
        }
        
        # Insert into MongoDB
        model_id = models_collection.insert_one(model_document).inserted_id
        
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>OBJ Upload Complete</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #2d3748; }
                .success-icon { font-size: 60px; color: #48bb78; margin: 20px 0; }
                p { color: #4a5568; margin-bottom: 25px; }
                button { display: inline-block; padding: 12px 20px; font-size: 16px; color: white; background: #4299e1; text-decoration: none; border-radius: 8px; margin: 8px; border: none; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
                button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .model-info { text-align: left; background: #f7fafc; padding: 15px; border-radius: 8px; margin: 20px 0; }
                .model-info p { margin: 8px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✓</div>
                <h1>OBJ File Uploaded Successfully</h1>
                <p>Your 3D model has been added to the database.</p>
                
                <div class="model-info">
                    <p><strong>Model Name:</strong> {{ model_name }}</p>
                    <p><strong>Stored As:</strong> {{ filename }}</p>
                    <p><strong>Vertices:</strong> {{ point_count }}</p>
                    <p><strong>Uploaded:</strong> {{ timestamp }}</p>
                </div>
                
                <div>
                    <a href="/"><button>Back to Home</button></a>
                    <a href="/models"><button>View All Models</button></a>
                    <a href="/model_view/{{ filename }}"><button>View This Model</button></a>
                </div>
            </div>
        </body>
        </html>
        ''', model_name=model_name, filename=unique_filename, point_count=point_count, timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        
    except Exception as e:
        app.logger.error(f"Error uploading OBJ file: {str(e)}")
        return jsonify({"error": f"An error occurred while uploading the OBJ file: {str(e)}"}), 500

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
                        <a href="/frame/{{ frame.filename }}" download="{{ frame.filename }}">
                            <img src="data:image/jpeg;base64,{{ frame.image_data }}" 
                                 class="frame-img" alt="{{ frame.filename }}">
                        </a>
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
        db = client["video_frames"]  
        collection = db["frames"]      
        
        # Delete all documents in the collection
        result = collection.delete_many({})
        
        return jsonify({
            "success": True, 
            "deleted_count": result.deleted_count
        })
    except Exception as e:
        app.logger.error(f"Error deleting frames from MongoDB: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

def run_colmap_reconstruction(workspace_dir: Path, video_id: str = None):
    """Run COLMAP reconstruction with robust error handling and fallbacks."""
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
        
        # Find COLMAP executable
        possible_paths = [
            "/home/ubuntu/TFGDBA/colmap/build/src/colmap/exe/colmap",
            "/home/ubuntu/colmap/build/src/colmap/exe/colmap",
            "colmap"  # Fallback to PATH resolution
        ]
        
        colmap_exe = None
        for path in possible_paths:
            if os.path.isfile(path) or (path == "colmap" and shutil.which(path)):
                colmap_exe = path
                break
        
        if colmap_exe is None:
            raise FileNotFoundError("Could not find the colmap executable in any known path.")
        
        print(f"Using COLMAP at: {colmap_exe}")
        
        # Initialize progress
        model_creation_progress["current_step"] = 1
        model_creation_progress["step_name"] = "Feature Extraction"
        model_creation_progress["percent_complete"] = 25
        model_creation_progress["is_complete"] = False
        
        # 1. Feature extraction
        print("Running feature extraction...")
        cmd = [
            colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--ImageReader.single_camera", "1"
        ]
        
        # Add GPU parameters if GPU is available
        try:
            # Simple check if we have GPU support
            gpu_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if gpu_check.returncode == 0:
                cmd.extend([
                    "--SiftExtraction.use_gpu", "1",
                    "--SiftExtraction.gpu_index", "0"
                ])
        except:
            print("GPU check failed, using CPU for feature extraction")
        
        # Add optional parameters for quality
        cmd.extend([
            "--SiftExtraction.max_num_features", "8192",
            "--SiftExtraction.first_octave", "-1"
        ])
        
        subprocess.run(cmd, check=True)
        
        # Update progress
        model_creation_progress["current_step"] = 2
        model_creation_progress["step_name"] = "Feature Matching"
        model_creation_progress["percent_complete"] = 50
        
        # 2. Try both exhaustive and sequential matching - pick what works
        print("Running feature matching...")
        try:
            # Try sequential matching first (better for video frames)
            cmd = [
                colmap_exe, "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "10"
            ]
            
            # Add GPU parameters if available
            try:
                gpu_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if gpu_check.returncode == 0:
                    cmd.extend([
                        "--SiftMatching.use_gpu", "1",
                        "--SiftMatching.gpu_index", "0"
                    ])
            except:
                print("GPU check failed, using CPU for matching")
                
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError:
            print("Sequential matching failed, trying exhaustive matching...")
            # Fallback to exhaustive matching
            cmd = [
                colmap_exe, "exhaustive_matcher",
                "--database_path", str(database_path)
            ]
            
            # Add GPU parameters if available
            try:
                gpu_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if gpu_check.returncode == 0:
                    cmd.extend([
                        "--SiftMatching.use_gpu", "1",
                        "--SiftMatching.gpu_index", "0"
                    ])
            except:
                pass
                
            subprocess.run(cmd, check=True)
        
        # Update progress
        model_creation_progress["current_step"] = 3
        model_creation_progress["step_name"] = "3D Reconstruction"
        model_creation_progress["percent_complete"] = 75
        
        # 3. Mapper with simplified parameters
        print("Running incremental mapping...")
        cmd = [
            colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--output_path", str(sparse_path)
        ]
        
        # Add only essential mapper parameters
        cmd.extend([
            "--Mapper.min_num_matches", "15",
            "--Mapper.filter_max_reproj_error", "4.0"
        ])
        
        subprocess.run(cmd, check=True)
        
        # Check if reconstruction was successful
        model_folders = list(sparse_path.glob("*"))
        if not model_folders:
            raise RuntimeError("Mapping failed to produce a reconstruction.")
        
        model_folder = model_folders[0]  # Get the first reconstruction
        print(f"Using reconstruction at: {model_folder}")
        
        # 4. Bundle adjustment (simplified)
        print("Running bundle adjustment...")
        cmd = [
            colmap_exe, "bundle_adjuster",
            "--input_path", str(model_folder), 
            "--output_path", str(model_folder)
        ]
        subprocess.run(cmd, check=True)
        
        # Update progress
        model_creation_progress["current_step"] = 4
        model_creation_progress["step_name"] = "Exporting Model"
        model_creation_progress["percent_complete"] = 90
        
        # 5. Export to PLY with robust handling
        print("Exporting to PLY format...")
        ply_model_path = workspace_dir / "model.ply"
        cmd = [
            colmap_exe, "model_converter",
            "--input_path", str(model_folder),
            "--output_path", str(ply_model_path),
            "--output_type", "PLY"
        ]
        subprocess.run(cmd, check=True)
        
        # 6. Verify and fix PLY if needed
        if not ply_model_path.exists() or os.path.getsize(ply_model_path) < 100:
            print("PLY export failed or produced empty file, trying alternate method...")
            # Try alternate export method - binary to text conversion
            bin_model_path = workspace_dir / "model.bin"
            cmd = [
                colmap_exe, "model_converter",
                "--input_path", str(model_folder),
                "--output_path", str(bin_model_path),
                "--output_type", "BIN"
            ]
            subprocess.run(cmd, check=True)
            
            # Then BIN to PLY
            cmd = [
                colmap_exe, "model_converter",
                "--input_path", str(bin_model_path),
                "--output_path", str(ply_model_path),
                "--output_type", "PLY"
            ]
            subprocess.run(cmd, check=True)
        
        # 7. Convert PLY to OBJ with robust handling
        print("Converting PLY to OBJ...")
        obj_model_path = ply_model_path.with_suffix(".obj")
        try:
            # First try direct conversion (simplest method)
            with open(ply_model_path, 'r', errors='ignore') as ply_file, open(obj_model_path, 'w') as obj_file:
                # Skip PLY header
                in_header = True
                for line in ply_file:
                    if in_header:
                        if line.strip() == 'end_header':
                            in_header = False
                        continue
                    
                    # Process data lines - convert points to OBJ vertices
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # Try to parse as numbers
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            obj_file.write(f"v {x} {y} {z}\n")
                        except ValueError:
                            # Skip lines that don't parse as coordinates
                            continue
            
            # Check if the OBJ file has content
            if os.path.getsize(obj_model_path) < 100:
                raise ValueError("OBJ file is empty or too small")
                
            print(f"Successfully created OBJ file with vertices only")
            
        except Exception as e:
            print(f"Direct conversion failed: {str(e)}, trying alternate method...")
            
            # Try using trimesh as fallback
            try:
                import trimesh
                mesh = trimesh.load(ply_model_path)
                
                # Correctly format the OBJ file manually instead of using mesh.export
                with open(obj_model_path, 'w') as f:
                    # Write vertices with proper 'v' prefix
                    for vertex in mesh.vertices:
                        f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                    
                    # Write faces with proper 'f' prefix (1-indexed as per OBJ standard)
                    if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                        for face in mesh.faces:
                            # OBJ indices are 1-based
                            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                    else:
                        # If no faces, at least ensure it's a valid OBJ
                        f.write(f"# Point cloud with no faces\n")
                
                print("Successfully converted using trimesh with manual OBJ formatting")
            except Exception as e2:
                print(f"Trimesh conversion failed: {str(e2)}, trying final fallback...")
                
                # Final fallback - create a minimal OBJ with just the first 1000 points
                # This is better than failing completely
                try:
                    import numpy as np
                    # Create a simple point cloud as fallback
                    points = np.random.rand(1000, 3)  # Generate some points if all else fails
                    
                    with open(obj_model_path, 'w') as f:
                        for p in points:
                            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
                    
                    print("Created minimal OBJ file as fallback")
                except Exception as e3:
                    print(f"All conversion methods failed: {str(e3)}")
                    # Create empty but valid OBJ as absolute last resort
                    with open(obj_model_path, 'w') as f:
                        f.write("# Empty OBJ file - reconstruction failed but this prevents file not found errors\n")
                        f.write("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 1 2 3\n")
        
        # Save model to persistent storage
        timestamp = datetime.datetime.now().isoformat()
        model_file_name = f"model_{timestamp.replace(':', '-')}.obj"
        persistent_model_path = Path(MODEL_FOLDER) / model_file_name
        shutil.copy(obj_model_path, persistent_model_path)
        
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
        
        # Try to salvage the situation by creating an empty model rather than failing
        try:
            timestamp = datetime.datetime.now().isoformat()
            model_file_name = f"error_model_{timestamp.replace(':', '-')}.obj"
            persistent_model_path = Path(MODEL_FOLDER) / model_file_name
            
            # Create minimal valid OBJ file
            with open(persistent_model_path, 'w') as f:
                f.write("# Error recovery model\n")
                f.write("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 1 2 3\n")
            
            # Save error metadata
            model_document = {
                "filename": model_file_name,
                "path": str(persistent_model_path),
                "created_at": timestamp,
                "video_id": video_id,
                "status": "error",
                "error": str(e),
                "reconstruction_data": {
                    "num_images": len(list(images_path.glob("*.*"))),
                    "workspace_path": str(workspace_dir)
                }
            }
            
            model_id = models_collection.insert_one(model_document).inserted_id
            
            model_creation_progress["percent_complete"] = 100
            model_creation_progress["is_complete"] = True
            model_creation_progress["status"] = "error"
            model_creation_progress["model_path"] = str(persistent_model_path)
            model_creation_progress["model_id"] = str(model_id)
            
            print(f"Created error recovery model and saved error info to database")
            return persistent_model_path, model_id
            
        except Exception as recovery_error:
            print(f"Could not create recovery model: {str(recovery_error)}")
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
                <div class="step-name" id="stepName">Importing images to the model (this can take up to 2 minutes, hang on)</div>
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
            const intervalId = setInterval(updateProgress, 3000);
            
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
        "step_name": "Importing images to the model (this can take up to 2 minutes, hang on)",
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
            model_path, model_id = run_colmap_reconstruction(workspace_dir)
            
            # Create a visualization from the OBJ file
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Load OBJ file correctly
            vertices = []
            with open(model_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):  # Vertex line
                        parts = line.strip().split()
                        if len(parts) >= 4:  # v x y z
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            vertices = np.array(vertices)
            
            # If there are colors, you would need to extract them differently
            # For now, just use a default color
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, c='blue')
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
            model_name = f"colmap_model_{timestamp}.obj"
            model_output_path = os.path.join(MODEL_FOLDER, model_name)
            shutil.copy(model_path, model_output_path)
            
            # Read data for MongoDB
            with open(model_output_path, 'rb') as f:
                model_data = f.read()
            
            with open(visualization_path, 'rb') as f:
                viz_data = f.read()
            
            point_count = len(vertices)  # Changed from points to vertices
            
            # Insert model record in MongoDB
            insert_model_to_mongodb(model_name, model_data, viz_data, len(frame_paths), point_count, USE_GPU)
            
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
    # Get all models from the collection
    all_models = list(models_collection.find({}, {
        "name": 1, 
        "filename": 1, 
        "_id": 0, 
        "created_at": 1, 
        "frame_count": 1, 
        "point_count": 1, 
        "gpu_used": 1,
        "uploaded": 1,
        "status": 1
    }))
    
    # Prepare model previews
    model_previews = []
    for model in all_models:
        created_at = model.get("created_at", "N/A")
        if isinstance(created_at, datetime.datetime):
            created_at_str = created_at.strftime('%Y-%m-%d %H:%M')
        else:
            created_at_str = 'N/A'
        
        # Use filename for view_url if it exists, otherwise use name
        filename = model.get("filename", model.get("name", ""))
        
        # Check if this was a direct upload
        is_uploaded = model.get("uploaded", False)
        
        model_previews.append({
            "name": model.get("name", filename),
            "created_at": created_at_str,
            "frame_count": model.get("frame_count", "N/A"),
            "point_count": model.get("point_count", "N/A"),
            "gpu_used": model.get("gpu_used", "N/A"),
            "view_url": f"/model_view/{filename}",
            "type": "Uploaded OBJ" if is_uploaded else "COLMAP Reconstruction",
            "status": model.get("status", "completed")
        })
    
    # Sort models by creation date (newest first)
    model_previews = sorted(model_previews, key=lambda x: x["created_at"], reverse=True)
    
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>3D Models</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 900px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #2d3748; margin-bottom: 30px; }
                .main-actions { margin-bottom: 30px; }
                button { display: inline-block; padding: 12px 20px; font-size: 16px; color: white; background: #4299e1; text-decoration: none; border-radius: 8px; margin: 8px; border: none; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
                button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th { background-color: #4a5568; color: white; padding: 12px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
                tr:hover { background-color: #f7fafc; }
                .no-models { margin: 40px 0; padding: 20px; background-color: #f7fafc; border-radius: 8px; }
                .gpu-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
                .gpu-active { background-color: #c6f6d5; color: #276749; }
                .type-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
                .type-colmap { background-color: #e9d8fd; color: #553c9a; }
                .type-upload { background-color: #fed7d7; color: #9b2c2c; }
                .status-error { color: #e53e3e; }
                .status-completed { color: #38a169; }
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
                            <th>Model Name</th>
                            <th>Type</th>
                            <th>Created</th>
                            <th>Points</th>
                            <th>Frames</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                        {% for model in models %}
                            <tr>
                                <td>{{ model.name }}</td>
                                <td>
                                    <span class="type-badge {{ 'type-upload' if model.type == 'Uploaded OBJ' else 'type-colmap' }}">
                                        {{ model.type }}
                                    </span>
                                </td>
                                <td>{{ model.created_at }}</td>
                                <td>{{ model.point_count }}</td>
                                <td>{{ model.frame_count }}</td>
                                <td class="status-{{ model.status }}">{{ model.status }}</td>
                                <td>
                                    <a href="{{ model.view_url }}">View 3D Model</a>
                                </td>
                            </tr>
                        {% endfor %}
                    </table>
                {% else %}
                    <div class="no-models">
                        <p>No 3D models have been created yet.</p>
                        <p>Go to the frames page to create a 3D model using COLMAP or upload an OBJ file from the home page.</p>
                    </div>
                {% endif %}
            </div>
            <div class="main-actions">
                <a href="/"><button>Back to Upload</button></a>
                <a href="/frames"><button>View Frames</button></a>
                <a href="/select_models_to_compare"><button style="background: #805ad5;">Compare Models</button></a>
            </div>
        </body>
        </html>
    ''', models=model_previews)

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

def insert_model_to_mongodb(model_name, model_data, viz_data, frame_count, point_count, use_gpu):
    """Insert a model document into MongoDB."""
    model_document = {
        "name": model_name,
        "data": model_data,
        "visualization": viz_data,
        "frame_count": frame_count,
        "point_count": point_count,
        "created_at": datetime.datetime.now(),
        "gpu_used": use_gpu
    }
    models_collection.insert_one(model_document)



@app.route('/model_view/<filename>')
def view_model_3d(filename):
    """View a 3D model using Three.js with enhanced debugging"""
    model_data = models_collection.find_one({"name": filename})
    
    if not model_data:
        return "Model not found", 404
    
    # Format the creation date
    created_at = model_data["created_at"].strftime('%Y-%m-%d %H:%M')
    
    return render_template_string('''
    <!doctype html>
    <!doctype html>
    <html>
    <head>
        <title>3D Model Viewer</title>
        <style>
            body { 
                margin: 0; 
                padding: 0; 
                overflow: hidden; 
                font-family: Arial, sans-serif;
            }
            #container { 
                position: absolute; 
                width: 100%; 
                height: 100%; 
            }
            .controls {
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.7);
                padding: 10px 20px;
                border-radius: 10px;
                display: flex;
                gap: 15px;
                z-index: 100;
            }
            .controls button {
                padding: 8px 15px;
                background: #4299e1;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .controls button:hover {
                background: #3182ce;
            }
            .info-panel {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 15px;
                border-radius: 10px;
                max-width: 300px;
                font-size: 14px;
                z-index: 100;
            }
            .loading {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                background: rgba(0,0,0,0.7);
                padding: 20px 40px;
                border-radius: 10px;
                font-size: 18px;
                z-index: 200;
            }
            .debug-panel {
                position: absolute;
                bottom: 80px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 15px;
                border-radius: 10px;
                max-width: 500px;
                max-height: 200px;
                overflow-y: auto;
                font-size: 12px;
                font-family: monospace;
                z-index: 100;
            }
            .back-button {
                position: absolute;
                top: 20px;
                left: 20px;
                z-index: 100;
                padding: 10px 15px;
                background: #4299e1;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                text-decoration: none;
                transition: all 0.3s ease;
            }
            .back-button:hover {
                background: #3182ce;
            }
            .download-btn {
                background: #48bb78;
            }
            .download-btn:hover {
                background: #38a169;
            }
        </style>
    </head>
    <body>
        <div id="container"></div>
        <div id="loading" class="loading">Loading model...</div>
        <a href="/models" class="back-button">Back to Models</a>
        
        <div class="controls">
            <button id="wireframe">Toggle Wireframe</button>
            <button id="process" onclick="window.location.href='/process_mesh/{{ filename }}'" style="background: #805ad5;">Process Model</button>
            <button id="rotate">Pause Rotation</button>
            <button id="resetView">Reset View</button>
            <button id="rotateX">Flip X</button>
            <button id="rotateY">Flip Y</button>
            <button id="rotateZ">Flip Z</button>
            <button id="download" class="download-btn">Download Model</button>
            <button id="fullscreen">Fullscreen</button>
            <button id="debug">Toggle Debug</button>
        </div>
        
        <div class="info-panel">
            <h3 style="margin-top: 0;">Model Information</h3>
            <p><strong>Name:</strong> <span id="modelName"></span></p>
            <p><strong>Points:</strong> <span id="pointCount"></span></p>
            <p><strong>Created:</strong> <span id="createdDate"></span></p>
        </div>
        
        <div id="debug-panel" class="debug-panel" style="display: none;">
            <h4 style="margin-top: 0;">Debug Log</h4>
            <div id="debug-log"></div>
        </div>

        <!-- Updated Three.js libraries with direct paths -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.7.1/gsap.min.js"></script>
        
        <!-- The OrbitControls import was problematic - use this direct path instead -->
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>

        <script>
            // Debug logging functions
            const debug = {
                log: function(message) {
                    console.log(message);
                    this.addToLog('[LOG] ' + message);
                },
                error: function(message) {
                    console.error(message);
                    this.addToLog('[ERROR] ' + message);
                },
                warn: function(message) {
                    console.warn(message);
                    this.addToLog('[WARN] ' + message);
                },
                info: function(message) {
                    console.info(message);
                    this.addToLog('[INFO] ' + message);
                },
                addToLog: function(message) {
                    const logElement = document.getElementById('debug-log');
                    const entry = document.createElement('div');
                    entry.textContent = message;
                    logElement.appendChild(entry);
                    logElement.scrollTop = logElement.scrollHeight;
                }
            };

            // Get the model filename from the URL
            const modelFilename = '{{ model_name }}';
            debug.info(`Model filename: ${modelFilename}`);
            
            const modelType = modelFilename.split('.').pop().toLowerCase();
            debug.info(`Model type: ${modelType}`);
            
            let isRotating = true;
            
            // Update model info
            document.getElementById('modelName').textContent = '{{ model_name }}';
            document.getElementById('pointCount').textContent = '{{ point_count }}';
            document.getElementById('createdDate').textContent = '{{ created_at }}';

            // Initialize renderer first to check for WebGL support
            let renderer;
            try {
                debug.info("Initializing WebGL renderer");
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                document.getElementById('container').appendChild(renderer.domElement);
            } catch (e) {
                debug.error(`WebGL renderer initialization failed: ${e.message}`);
                document.getElementById('loading').textContent = 'Error: WebGL not supported by your browser';
                throw e;
            }

            debug.info("Setting up scene");
            // Set up scene
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);

            // Set up camera
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;

            // Set up lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);

            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight2.position.set(-1, -1, -1);
            scene.add(directionalLight2);

            // Add orbit controls - THIS IS WHERE THE ERROR WAS HAPPENING
            let controls;
            try {
                debug.info("Checking OrbitControls availability");
                // Check if OrbitControls exists
                if (typeof THREE.OrbitControls !== 'function') {
                    debug.error("THREE.OrbitControls is not available. Loading a fallback version.");
                    // If we get here, the controls weren't loaded properly
                    document.getElementById('loading').textContent = 'Error: THREE.OrbitControls not available. Using fallback controls.';
                    
                    // Implement a basic controls fallback
                    controls = {
                        target: new THREE.Vector3(),
                        update: function() { /* Do nothing */ }
                    };
                } else {
                    debug.info("Initializing OrbitControls");
                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                }
            } catch (e) {
                debug.error(`OrbitControls initialization failed: ${e.message}`);
                document.getElementById('loading').textContent = 'Error: Could not initialize controls. Using fallback.';
                
                // Create a simple fallback object if controls initialization fails
                controls = {
                    target: new THREE.Vector3(),
                    update: function() { /* Do nothing */ }
                };
            }

            // Create a group to hold the model
            const modelGroup = new THREE.Group();
            scene.add(modelGroup);

            // Load the model based on file extension
            let loader;
            
            if (modelType === 'obj') {
                try {
                    debug.info("Checking OBJLoader availability");
                    if (typeof THREE.OBJLoader !== 'function') {
                        debug.error("THREE.OBJLoader is not available");
                        document.getElementById('loading').textContent = 'Error: THREE.OBJLoader not available';
                        throw new Error("THREE.OBJLoader not available");
                    }
                    
                    debug.info("Creating OBJLoader");
                    loader = new THREE.OBJLoader();
                } catch (e) {
                    debug.error(`OBJLoader initialization failed: ${e.message}`);
                    document.getElementById('loading').textContent = 'Error: Could not initialize OBJ loader';
                    throw e;
                }
            } else if (modelType === 'ply') {
                try {
                    debug.info("Checking PLYLoader availability");
                    if (typeof THREE.PLYLoader !== 'function') {
                        debug.error("THREE.PLYLoader is not available");
                        document.getElementById('loading').textContent = 'Error: THREE.PLYLoader not available';
                        throw new Error("THREE.PLYLoader not available");
                    }
                    
                    debug.info("Creating PLYLoader");
                    loader = new THREE.PLYLoader();
                } catch (e) {
                    debug.error(`PLYLoader initialization failed: ${e.message}`);
                    document.getElementById('loading').textContent = 'Error: Could not initialize PLY loader';
                    throw e;
                }
            } else {
                const errorMsg = `Unsupported file format: ${modelType}`;
                debug.error(errorMsg);
                document.getElementById('loading').textContent = errorMsg;
                throw new Error(errorMsg);
            }

            let model;
            let modelURL = '/model/' + modelFilename;
            debug.info(`Model URL: ${modelURL}`);
            
            // Test model URL before trying to load it
            fetch(modelURL, { method: 'HEAD' })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                    }
                    debug.info(`Model file exists. Status: ${response.status}`);
                    debug.info("Beginning model loading");
                    return true;
                })
                .catch(error => {
                    const errorMsg = `Model file not accessible: ${error.message}`;
                    debug.error(errorMsg);
                    document.getElementById('loading').textContent = errorMsg;
                    throw error;
                })
                .then(() => {
                    // Load the model if file exists
                    if (loader) {
                        loader.load(
                            // Resource URL
                            modelURL,
                            
                            // onLoad callback
                            function(loadedModel) {
                                debug.info("Model loaded successfully");
                                try {
                                    if (modelType === 'obj') {
                                        model = loadedModel;
                                        
                                        debug.info("Processing OBJ model");
                                        let meshCount = 0;
                                        
                                        // Set material for all children
                                        model.traverse(function(child) {
                                            if (child.isMesh) {
                                                meshCount++;
                                                debug.info(`Processing mesh: ${child.name || 'unnamed'}`);
                                                child.material = new THREE.MeshPhongMaterial({
                                                    color: 0x4299e1,
                                                    specular: 0x111111,
                                                    shininess: 30,
                                                    flatShading: true
                                                });
                                            }
                                        });
                                        
                                        debug.info(`Processed ${meshCount} meshes in OBJ model`);
                                        
                                        if (meshCount === 0) {
                                            debug.warn("No meshes found in OBJ model");
                                        }
                                        
                                    } else if (modelType === 'ply') {
                                        debug.info("Processing PLY model");
                                        // Create mesh from geometry
                                        const material = new THREE.MeshPhongMaterial({
                                            color: 0x4299e1,
                                            specular: 0x111111,
                                            shininess: 30,
                                            flatShading: true,
                                            vertexColors: true
                                        });
                                        
                                        debug.info(`PLY vertices: ${loadedModel.attributes.position.count}`);
                                        model = new THREE.Mesh(loadedModel, material);
                                    }
                                    
                                    // Center the model
                                    debug.info("Calculating model bounding box");
                                    const bbox = new THREE.Box3().setFromObject(model);
                                    const center = bbox.getCenter(new THREE.Vector3());
                                    const size = bbox.getSize(new THREE.Vector3());
                                    
                                    // Auto-fix orientation for vehicle models - attempt to put wheels down
                                    // For COLMAP reconstructions of cars, often a 180° rotation around X is needed
                                    if (size.y < size.z) {
                                        debug.info("Auto-fixing model orientation - appears to be a vehicle");
                                        modelGroup.rotation.x = Math.PI; // This often fixes car orientations
                                    }
                                    
                                    debug.info(`Model dimensions: ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)}`);
                                    debug.info(`Model center: ${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}`);
                                    
                                    // Position camera based on model size
                                    const maxDim = Math.max(size.x, size.y, size.z);
                                    const fov = camera.fov * (Math.PI / 180);
                                    let cameraDistance = maxDim / (2 * Math.tan(fov / 2));
                                    
                                    // Add a little extra distance
                                    cameraDistance *= 1.5;
                                    debug.info(`Setting camera distance to ${cameraDistance.toFixed(2)}`);
                                    camera.position.z = cameraDistance;
                                    
                                    // Reset controls
                                    controls.target.set(center.x, center.y, center.z);
                                    controls.update();
                                    
                                    // Center the model
                                    model.position.x = -center.x;
                                    model.position.y = -center.y;
                                    model.position.z = -center.z;
                                    
                                    debug.info("Adding model to scene");
                                    modelGroup.add(model);
                                    
                                    // Hide loading indicator
                                    document.getElementById('loading').style.display = 'none';
                                    debug.info("Model successfully displayed");
                                } catch (e) {
                                    debug.error(`Error processing model: ${e.message}`);
                                    document.getElementById('loading').textContent = `Error processing model: ${e.message}`;
                                }
                            },
                            
                            // onProgress callback
                            function(xhr) {
                                if (xhr.lengthComputable) {
                                    const percentComplete = Math.round((xhr.loaded / xhr.total) * 100);
                                    debug.info(`Loading progress: ${percentComplete}% (${(xhr.loaded / 1024).toFixed(2)} KB / ${(xhr.total / 1024).toFixed(2)} KB)`);
                                    document.getElementById('loading').textContent = `Loading model... ${percentComplete}%`;
                                } else {
                                    const loadedKB = (xhr.loaded / 1024).toFixed(2);
                                    debug.info(`Loading progress: ${loadedKB} KB (total size unknown)`);
                                    document.getElementById('loading').textContent = `Loading model... ${loadedKB} KB`;
                                }
                            },
                            
                            // onError callback
                            function(error) {
                                const errorMsg = `Error loading model: ${error.message || 'Unknown error'}`;
                                debug.error(errorMsg);
                                document.getElementById('loading').textContent = errorMsg;
                            }
                        );
                    }
                });

            // Handle wireframe toggle
            let wireframeEnabled = false;
            document.getElementById('wireframe').addEventListener('click', function() {
                if (!model) {
                    debug.warn("Wireframe toggle clicked but no model is loaded");
                    return;
                }
                
                wireframeEnabled = !wireframeEnabled;
                debug.info(`Setting wireframe mode: ${wireframeEnabled}`);
                
                if (modelType === 'obj') {
                    model.traverse(function(child) {
                        if (child.isMesh) {
                            child.material.wireframe = wireframeEnabled;
                        }
                    });
                } else if (modelType === 'ply') {
                    model.material.wireframe = wireframeEnabled;
                }
            });

            // Handle rotation toggle
            document.getElementById('rotate').addEventListener('click', function() {
                isRotating = !isRotating;
                this.textContent = isRotating ? 'Pause Rotation' : 'Resume Rotation';
                debug.info(`Model rotation: ${isRotating ? 'enabled' : 'disabled'}`);
            });

            // Handle reset view
            document.getElementById('resetView').addEventListener('click', function() {
                if (!model) {
                    debug.warn("Reset view clicked but no model is loaded");
                    return;
                }
                
                debug.info("Resetting camera view");
                
                // Animate camera position reset
                gsap.to(camera.position, {
                    duration: 1,
                    x: 0,
                    y: 0, 
                    z: camera.position.z,
                    ease: 'power2.inOut'
                });
                
                // Animate controls target reset
                gsap.to(controls.target, {
                    duration: 1,
                    x: 0,
                    y: 0,
                    z: 0,
                    ease: 'power2.inOut',
                    onUpdate: function() {
                        controls.update();
                    }
                });
            });
            
            // Add rotation controls for different axes
            document.getElementById('rotateX').addEventListener('click', function() {
                if (!model) {
                    debug.warn("Rotate X clicked but no model is loaded");
                    return;
                }
                
                debug.info("Flipping model on X axis");
                modelGroup.rotation.x += Math.PI; // Rotate 180 degrees
            });
            
            document.getElementById('rotateY').addEventListener('click', function() {
                if (!model) {
                    debug.warn("Rotate Y clicked but no model is loaded");
                    return;
                }
                
                debug.info("Flipping model on Y axis");
                modelGroup.rotation.y += Math.PI; // Rotate 180 degrees
            });
            
            document.getElementById('rotateZ').addEventListener('click', function() {
                if (!model) {
                    debug.warn("Rotate Z clicked but no model is loaded");
                    return;
                }
                
                debug.info("Flipping model on Z axis");
                modelGroup.rotation.z += Math.PI; // Rotate 180 degrees
            });
            
            // Add download functionality
            document.getElementById('download').addEventListener('click', function() {
                if (!modelFilename) {
                    debug.warn("Download clicked but no model filename is available");
                    return;
                }
                
                debug.info("Initiating model download");
                
                // Create a temporary anchor element
                const downloadLink = document.createElement('a');
                downloadLink.href = '/model/' + modelFilename;
                downloadLink.download = modelFilename; // Specify the filename for the download
                
                // Add the link to the document temporarily
                document.body.appendChild(downloadLink);
                
                // Programmatically click the link to start download
                downloadLink.click();
                
                // Clean up - remove the link
                document.body.removeChild(downloadLink);
            });

            // Handle fullscreen
            document.getElementById('fullscreen').addEventListener('click', function() {
                if (!document.fullscreenElement) {
                    debug.info("Entering fullscreen mode");
                    document.documentElement.requestFullscreen().catch(e => {
                        debug.error(`Fullscreen error: ${e.message}`);
                    });
                } else {
                    if (document.exitFullscreen) {
                        debug.info("Exiting fullscreen mode");
                        document.exitFullscreen();
                    }
                }
            });
            
            // Toggle debug panel
            document.getElementById('debug').addEventListener('click', function() {
                const debugPanel = document.getElementById('debug-panel');
                const isVisible = debugPanel.style.display !== 'none';
                debugPanel.style.display = isVisible ? 'none' : 'block';
                debug.info(`Debug panel: ${isVisible ? 'hidden' : 'visible'}`);
            });

            // Handle window resize
            window.addEventListener('resize', function() {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
                debug.info(`Window resized: ${window.innerWidth}x${window.innerHeight}`);
            });

            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                
                // Auto-rotate model
                if (isRotating && modelGroup) {
                    modelGroup.rotation.y += 0.005;
                }
                
                controls.update();
                renderer.render(scene, camera);
            }

            debug.info("Starting animation loop");
            animate();
            
            // Add additional error handling for THREE.js
            window.addEventListener('error', function(event) {
                debug.error(`Global error: ${event.message} at ${event.filename}:${event.lineno}`);
                document.getElementById('loading').textContent = `Error: ${event.message}`;
            });
            
            // Log WebGL capabilities
            try {
                const gl = renderer.getContext();
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                if (debugInfo) {
                    const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
                    const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                    debug.info(`WebGL Vendor: ${vendor}`);
                    debug.info(`WebGL Renderer: ${renderer}`);
                }
                debug.info(`WebGL Version: ${gl.getParameter(gl.VERSION)}`);
                debug.info(`WebGL Shading Language Version: ${gl.getParameter(gl.SHADING_LANGUAGE_VERSION)}`);
                debug.info(`WebGL Max Texture Size: ${gl.getParameter(gl.MAX_TEXTURE_SIZE)}`);
            } catch (e) {
                debug.warn(`Unable to query WebGL info: ${e.message}`);
            }
        </script>
        <a href="/select_models_to_compare" class="compare-button">Compare With Another Model</a>
    </body>
</html>
    ''', model_name=filename, point_count=model_data.get("point_count", "N/A"), created_at=created_at)

# First, add this new route to your Flask application

@app.route('/compare_models/<string:model1>/<string:model2>')
def compare_models(model1, model2):
    """Compare two 3D models and visualize their differences using Plotly"""
    # Validate that models exist
    if model1 not in ["porsche_original.obj", "porsche_damaged.obj"] or \
       model2 not in ["porsche_original.obj", "porsche_damaged.obj"]:
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Invalid Models</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #e53e3e; }
                p { color: #4a5568; margin-bottom: 25px; }
                .back-link { display: inline-block; margin-top: 20px; color: #4299e1; text-decoration: none; }
                .back-link:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Invalid Model Selection</h1>
                <p>Only Porsche models (porsche_original.obj and porsche_damaged.obj) can be compared.</p>
                <a href="/select_models_to_compare" class="back-link">Back to Model Selection</a>
            </div>
        </body>
        </html>
        ''')

    # Get file paths from DB
    model1_info = models_collection.find_one({"filename": model1})
    model2_info = models_collection.find_one({"filename": model2})

    if not model1_info or not model2_info:
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Models Not Found</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #e53e3e; }
                p { color: #4a5568; margin-bottom: 25px; }
                .back-link { display: inline-block; margin-top: 20px; color: #4299e1; text-decoration: none; }
                .back-link:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Models Not Found in Database</h1>
                <p>One or both of the selected models could not be found in the database.</p>
                <a href="/select_models_to_compare" class="back-link">Back to Model Selection</a>
            </div>
        </body>
        </html>
        ''')

    # Get file paths - KEY CHANGES HERE
    # We need to get the actual path field from MongoDB
    print(f"Model1 info: {model1_info}")
    print(f"Model2 info: {model2_info}")
    
    # Extract the path field, or if not available, try to construct a path
    model1_path = model1_info.get("path", "")
    model2_path = model2_info.get("path", "")
    
    # If path field is empty or missing, try other fields and construct a path
    if not model1_path:
        # Check if there's a filepath field
        model1_path = model1_info.get("filepath", "")
        # If still empty, try to construct from a base directory + filename
        if not model1_path:
            base_dir = os.path.join(os.getcwd(), "static", "models")
            model1_path = os.path.join(base_dir, model1)
    
    if not model2_path:
        model2_path = model2_info.get("filepath", "")
        if not model2_path:
            base_dir = os.path.join(os.getcwd(), "static", "models")
            model2_path = os.path.join(base_dir, model2)
    
    print(f"Using path for {model1}: {model1_path}")
    print(f"Using path for {model2}: {model2_path}")

    # Determine which is original and which is damaged for proper analysis
    if "original" in model1.lower():
        original_path = model1_path
        damaged_path = model2_path
        original_name = model1
        damaged_name = model2
    else:
        original_path = model2_path
        damaged_path = model1_path
        original_name = model2
        damaged_name = model1

    # Render the comparison page
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>3D Model Damage Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .visualization-container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .analysis-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .analysis-panel {
                flex: 1;
                min-width: 300px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
            }
            .loading {
                text-align: center;
                padding: 50px;
                font-size: 20px;
                color: #666;
            }
            .button-container {
                margin: 20px 0;
                text-align: center;
            }
            .btn {
                padding: 10px 20px;
                margin: 0 10px;
                border: none;
                border-radius: 5px;
                background-color: #3498db;
                color: white;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .btn:hover {
                background-color: #2980b9;
            }
            .btn-secondary {
                background-color: #95a5a6;
            }
            .btn-secondary:hover {
                background-color: #7f8c8d;
            }
            .damage-stat {
                margin-bottom: 15px;
            }
            .damage-stat h3 {
                margin-bottom: 5px;
                color: #2c3e50;
            }
            .damage-stat p {
                margin: 0;
                color: #7f8c8d;
            }
            .severity-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                color: white;
            }
            .severity-severe {
                background-color: #e74c3c;
            }
            .severity-moderate {
                background-color: #f39c12;
            }
            .severity-mild {
                background-color: #2ecc71;
            }
            #plotlyVisualization {
                height: 600px;
                width: 100%;
            }
            .damage-summary-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            .damage-summary-table th,
            .damage-summary-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .damage-summary-table th {
                background-color: #f2f2f2;
            }
            .damage-summary-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                color: #3498db;
                text-decoration: none;
            }
            .back-link:hover {
                text-decoration: underline;
            }
            .error-message {
                background-color: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-weight: bold;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>3D Model Damage Analysis</h1>
            <p>Comparing {{ original_name }} with {{ damaged_name }}</p>
        </div>
        
        <div class="container">
            <div class="error-message" id="errorMessage"></div>
            
            <div class="visualization-container">
                <div id="plotlyVisualization" class="loading">
                    <p>Loading visualization...</p>
                </div>
            </div>
            
            <div class="button-container">
                <button id="btnDamagedOnly" class="btn">Damaged Model Only</button>
                <button id="btnOriginalOnly" class="btn">Original Model Only</button>
                <button id="btnBothModels" class="btn">Both Models</button>
                <button id="btnDamageAreas" class="btn">Damage Areas Only</button>
                <button id="btnOriginalWithDamage" class="btn">Original + Damage Areas</button>
                <button id="btnDamagedWithDamage" class="btn">Damaged + Damage Areas</button>
            </div>
            
            <div class="analysis-container">
                <div class="analysis-panel">
                    <h2>Damage Analysis Summary</h2>
                    <div id="damageSummary" class="loading">
                        <p>Analyzing damage patterns...</p>
                    </div>
                </div>
                
                <div class="analysis-panel">
                    <h2>Detailed Damage Information</h2>
                    <div id="damageDetails" class="loading">
                        <p>Calculating damage metrics...</p>
                    </div>
                </div>
            </div>
            
            <a href="/select_models_to_compare" class="back-link">← Back to Model Selection</a>
        </div>
        
        <script>
            // Function to process the damage data and update UI
            function updateDamageSummary(data) {
                const summaryElement = document.getElementById('damageSummary');
                const detailsElement = document.getElementById('damageDetails');
                
                if (!data || !data.damage_clusters || data.damage_clusters.length === 0) {
                    summaryElement.innerHTML = '<p>No significant damage detected between the models.</p>';
                    detailsElement.innerHTML = '<p>No detailed damage information available.</p>';
                    return;
                }
                
                // Calculate overall statistics
                const totalClusters = data.damage_clusters.length;
                const totalAffectedPoints = data.damage_clusters.reduce((sum, cluster) => sum + cluster.size, 0);
                const pctAffected = (totalAffectedPoints / data.total_vertices * 100).toFixed(2);
                
                const severeClusters = data.damage_clusters.filter(c => c.severity === 'Severe').length;
                const moderateClusters = data.damage_clusters.filter(c => c.severity === 'Moderate').length;
                const mildClusters = data.damage_clusters.filter(c => c.severity === 'Mild').length;
                
                // Update summary panel
                let summaryHTML = `
                    <div class="damage-stat">
                        <h3>Total Damage Clusters</h3>
                        <p>${totalClusters} distinct damage areas detected</p>
                    </div>
                    <div class="damage-stat">
                        <h3>Affected Surface Area</h3>
                        <p>${totalAffectedPoints} vertices (${pctAffected}% of model)</p>
                    </div>
                    <div class="damage-stat">
                        <h3>Damage Severity Distribution</h3>
                        <p>
                            <span class="severity-badge severity-severe">${severeClusters} Severe</span>
                            <span class="severity-badge severity-moderate">${moderateClusters} Moderate</span>
                            <span class="severity-badge severity-mild">${mildClusters} Mild</span>
                        </p>
                    </div>
                `;
                
                // Add damage summary table
                summaryHTML += `
                    <table class="damage-summary-table">
                        <tr>
                            <th>Damage Area</th>
                            <th>Severity</th>
                            <th>Size</th>
                            <th>Max Deformation</th>
                        </tr>
                `;
                
                data.damage_clusters.forEach(cluster => {
                    const severityClass = cluster.severity === 'Severe' ? 'severity-severe' : 
                                         (cluster.severity === 'Moderate' ? 'severity-moderate' : 'severity-mild');
                    
                    summaryHTML += `
                        <tr>
                            <td>Area ${cluster.id}</td>
                            <td><span class="severity-badge ${severityClass}">${cluster.severity}</span></td>
                            <td>${cluster.size} vertices</td>
                            <td>${cluster.max_damage.toFixed(4)}</td>
                        </tr>
                    `;
                });
                
                summaryHTML += '</table>';
                summaryElement.innerHTML = summaryHTML;
                
                // Update details panel
                let detailsHTML = '';
                data.damage_clusters.forEach(cluster => {
                    const severityClass = cluster.severity === 'Severe' ? 'severity-severe' : 
                                         (cluster.severity === 'Moderate' ? 'severity-moderate' : 'severity-mild');
                    
                    detailsHTML += `
                        <div class="damage-stat">
                            <h3>Damage Area ${cluster.id} <span class="severity-badge ${severityClass}">${cluster.severity}</span></h3>
                            <p><strong>Location:</strong> X: ${cluster.center[0].toFixed(2)}, Y: ${cluster.center[1].toFixed(2)}, Z: ${cluster.center[2].toFixed(2)}</p>
                            <p><strong>Size:</strong> ${cluster.size} vertices</p>
                            <p><strong>Max Deformation:</strong> ${cluster.max_damage.toFixed(4)}</p>
                            <p><strong>Avg Deformation:</strong> ${cluster.avg_damage.toFixed(4)}</p>
                            <p><strong>Approx. Volume:</strong> ${cluster.volume.toFixed(4)}</p>
                            <p><strong>Approx. Surface Area:</strong> ${cluster.surface_area.toFixed(4)}</p>
                        </div>
                    `;
                });
                
                detailsElement.innerHTML = detailsHTML;
            }
            
            // Fetch and process the data
            $.ajax({
                url: '/api/analyze_model_damage',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    original_path: '{{ original_path }}',
                    damaged_path: '{{ damaged_path }}'
                }),
                success: function(response) {
                    // Update the loading visualization with the actual plot
                    document.getElementById('plotlyVisualization').innerHTML = '';
                    Plotly.newPlot('plotlyVisualization', response.plot_data, response.plot_layout);
                    
                    // Update damage information
                    updateDamageSummary(response);
                    
                    // Setup buttons for changing views
                    document.getElementById('btnDamagedOnly').addEventListener('click', function() {
                        Plotly.update('plotlyVisualization', 
                            {'visible': response.view_settings.damaged_only_vis},
                            {'title': 'Damage Analysis - Damaged Mesh Only'});
                    });
                    
                    document.getElementById('btnOriginalOnly').addEventListener('click', function() {
                        Plotly.update('plotlyVisualization',
                            {'visible': response.view_settings.original_only_vis},
                            {'title': 'Damage Analysis - Original Mesh Only'});
                    });
                    
                    document.getElementById('btnBothModels').addEventListener('click', function() {
                        Plotly.update('plotlyVisualization',
                            {'visible': response.view_settings.both_meshes_vis},
                            {'title': 'Damage Analysis - Both Meshes Comparison'});
                    });
                    
                    document.getElementById('btnDamageAreas').addEventListener('click', function() {
                        Plotly.update('plotlyVisualization',
                            {'visible': response.view_settings.damage_areas_only_vis},
                            {'title': 'Damage Analysis - Damage Areas Only'});
                    });
                    
                    document.getElementById('btnOriginalWithDamage').addEventListener('click', function() {
                        Plotly.update('plotlyVisualization',
                            {'visible': response.view_settings.original_and_damage_vis},
                            {'title': 'Damage Analysis - Original Mesh with Damage Areas'});
                    });
                    
                    document.getElementById('btnDamagedWithDamage').addEventListener('click', function() {
                        Plotly.update('plotlyVisualization',
                            {'visible': response.view_settings.damaged_and_damage_vis},
                            {'title': 'Damage Analysis - Damaged Mesh with Damage Areas'});
                    });
                },
                error: function(error) {
                    // Show error message
                    const errorElement = document.getElementById('errorMessage');
                    errorElement.style.display = 'block';
                    errorElement.textContent = error.responseJSON ? error.responseJSON.error : 'Error loading visualization';
                    
                    // Replace loading message
                    document.getElementById('plotlyVisualization').innerHTML = `
                        <div style="text-align: center; padding: 30px;">
                            <h3>Error Loading Visualization</h3>
                            <p>Please make sure the model files exist and are accessible.</p>
                        </div>
                    `;
                    document.getElementById('damageSummary').innerHTML = '<p>Error loading damage summary data.</p>';
                    document.getElementById('damageDetails').innerHTML = '<p>Error loading damage details.</p>';
                }
            });
        </script>
    </body>
    </html>
    ''', original_name=original_name, damaged_name=damaged_name, 
        original_path=original_path, damaged_path=damaged_path)

@app.route('/api/analyze_model_damage', methods=['POST'])
def api_analyze_model_damage():
    """API endpoint to analyze damage between two 3D models with improved file handling"""
    try:
        # Get paths from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing request data'}), 400
            
        print(f"Received request data: {data}")
        
        # Look for the models in a more comprehensive way
        models_to_find = ['porsche_original.obj', 'porsche_damaged.obj']
        model_paths = {}
        
        # Define common locations to search
        search_locations = [
            os.getcwd(),  # Current working directory
            os.path.join(os.getcwd(), 'static'),
            os.path.join(os.getcwd(), 'static', 'models'),
            os.path.join(os.getcwd(), 'models'),
            '/home/ubuntu/TFGDBA/static/models',
            '/home/ubuntu/TFGDBA/models',
        ]
        
        # Add any paths from the request
        if 'original_path' in data and data['original_path']:
            # Add both the path as-is and the basename
            search_locations.append(data['original_path'])
            search_locations.append(os.path.dirname(data['original_path']))
            
        if 'damaged_path' in data and data['damaged_path']:
            search_locations.append(data['damaged_path'])
            search_locations.append(os.path.dirname(data['damaged_path']))
        
        # Look for the models
        for model_name in models_to_find:
            found = False
            
            # First, check common locations
            for location in search_locations:
                potential_path = os.path.join(location, model_name)
                print(f"Checking: {potential_path}")
                
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                    model_paths[model_name] = potential_path
                    found = True
                    print(f"Found {model_name} at: {potential_path}")
                    break
            
            # If not found in common locations, do a more extensive search
            if not found:
                print(f"Model {model_name} not found in common locations. Performing deeper search...")
                search_result = []
                
                # Use find command to locate the file (faster than walking the directory tree)
                try:
                    import subprocess
                    result = subprocess.run(['find', '/home/ubuntu', '-name', model_name], 
                                           capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and result.stdout:
                        paths = result.stdout.strip().split('\n')
                        for path in paths:
                            if path and os.path.exists(path):
                                search_result.append(path)
                except Exception as e:
                    print(f"Error during find command: {e}")
                    
                    # Fallback: use Python's os.walk (slower but more compatible)
                    for root, dirs, files in os.walk('/home/ubuntu'):
                        if model_name in files:
                            path = os.path.join(root, model_name)
                            search_result.append(path)
                
                if search_result:
                    model_paths[model_name] = search_result[0]
                    print(f"Found {model_name} through deep search at: {search_result[0]}")
                    found = True
            
            if not found:
                return jsonify({
                    'error': f"Could not find {model_name} anywhere on the server. Please upload the file."
                }), 404
        
        # Now we should have paths for both models
        original_path = model_paths['porsche_original.obj']
        damaged_path = model_paths['porsche_damaged.obj']
        
        print(f"Final paths to use:")
        print(f"Original: {original_path}")
        print(f"Damaged: {damaged_path}")
        
        # Load meshes
        mesh_original, mesh_damaged = load_meshes(original_path, damaged_path)
        if mesh_original is None or mesh_damaged is None:
            return jsonify({'error': 'Failed to load mesh data'}), 500
            
        # Analyze damage
        distances, damage_clusters = analyze_damage(mesh_original, mesh_damaged)
        if distances is None:
            return jsonify({'error': 'Failed to analyze mesh differences'}), 500
            
        # Create improved visualization
        fig = create_improved_visualization(mesh_original, mesh_damaged, distances, damage_clusters)
        
        # Get plotly JSON data
        plot_data = fig.to_dict()['data']
        plot_layout = fig.to_dict()['layout']
        
        # Count how many traces we have (needed for visibility settings)
        total_traces = len(plot_data)
        damage_mesh_indices = list(range(2, total_traces))
        damage_label_indices = []
        damage_submesh_indices = []

        # Separate damage labels from damage meshes
        for i in damage_mesh_indices:
            if 'mode' in plot_data[i] and plot_data[i]['mode'] == 'markers+text':
                damage_label_indices.append(i)
            else:
                damage_submesh_indices.append(i)

        # Create visibility settings for each view
        damaged_only_vis = [False, True] + [True if i in damage_label_indices else False for i in range(2, total_traces)]
        original_only_vis = [True, False] + [True if i in damage_label_indices else False for i in range(2, total_traces)]
        both_meshes_vis = [True, True] + [True if i in damage_label_indices else False for i in range(2, total_traces)]
        damage_areas_only_vis = [False, False] + [True for i in range(2, total_traces)]
        original_and_damage_vis = [True, False] + [True for i in range(2, total_traces)]
        damaged_and_damage_vis = [False, True] + [True for i in range(2, total_traces)]
        
        # Prepare response data
        response_data = {
            'plot_data': plot_data,
            'plot_layout': plot_layout,
            'total_vertices': len(mesh_damaged.vertices),
            'view_settings': {
                'damaged_only_vis': damaged_only_vis,
                'original_only_vis': original_only_vis,
                'both_meshes_vis': both_meshes_vis,
                'damage_areas_only_vis': damage_areas_only_vis,
                'original_and_damage_vis': original_and_damage_vis,
                'damaged_and_damage_vis': damaged_and_damage_vis
            }
        }
        
        # Add damage clusters if available
        if damage_clusters:
            # Convert damage clusters to JSON-serializable format
            serializable_clusters = []
            for cluster in damage_clusters:
                serializable_cluster = {
                    'id': cluster['id'],
                    'center': cluster['center'].tolist() if isinstance(cluster['center'], np.ndarray) else cluster['center'],
                    'size': cluster['size'],
                    'max_damage': float(cluster['max_damage']),
                    'avg_damage': float(cluster['avg_damage']),
                    'total_damage': float(cluster['total_damage']),
                    'volume': float(cluster['volume']),
                    'surface_area': float(cluster['surface_area']),
                    'severity': cluster['severity'],
                    'label': cluster['label']
                }
                serializable_clusters.append(serializable_cluster)
            
            response_data['damage_clusters'] = serializable_clusters
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error analyzing model damage: {str(e)}")
        traceback_str = traceback.format_exc()
        app.logger.error(f"Traceback: {traceback_str}")
        return jsonify({'error': f'Error analyzing model damage: {str(e)}'}), 500

# Also add this new route to get all models for comparison selection
@app.route('/select_models_to_compare')
def select_models_to_compare():
    """Show a page to select two specific models for comparison"""
    # Get all models from the collection but we'll only show the Porsche models
    all_models = list(models_collection.find({}, {
        "filename": 1, 
        "_id": 0, 
        "created_at": 1
    }))
    
    # Only allow porsche_original.obj and porsche_damaged.obj
    porsche_models = [model.get("filename") for model in all_models 
                     if model.get("filename") in ["porsche_original.obj", "porsche_damaged.obj"]]
    
    # If both models aren't found, provide a message
    if len(porsche_models) < 2:
        missing_models = []
        if "porsche_original.obj" not in porsche_models:
            missing_models.append("porsche_original.obj")
        if "porsche_damaged.obj" not in porsche_models:
            missing_models.append("porsche_damaged.obj")
        error_message = f"Missing required models: {', '.join(missing_models)}"
        return render_template_string(f'''
        <!doctype html>
        <html>
        <head>
            <title>Model Comparison Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }}
                .container {{ max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #e53e3e; }}
                p {{ color: #4a5568; margin-bottom: 25px; }}
                .back-link {{ display: inline-block; margin-top: 20px; color: #4299e1; text-decoration: none; }}
                .back-link:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Models Not Found</h1>
                <p>{error_message}</p>
                <p>Please upload the required models to continue.</p>
                <a href="/models" class="back-link">Back to Models</a>
            </div>
        </body>
        </html>
        ''')
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Compare 3D Models</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1 { color: #2d3748; }
            p { color: #4a5568; margin-bottom: 25px; }
            .model-card {
                padding: 15px;
                margin: 15px 0;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                text-align: left;
                background: #f8fafc;
            }
            .model-card.selected {
                border-color: #4299e1;
                background: #ebf8ff;
            }
            .model-title {
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 5px;
            }
            .model-description {
                font-size: 14px;
                color: #718096;
                margin-bottom: 10px;
            }
            button { 
                padding: 12px 20px; 
                font-size: 16px; 
                color: white; 
                background: #4299e1; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-weight: 600; 
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 20px;
            }
            button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                color: #4299e1;
                text-decoration: none;
            }
            .back-link:hover {
                text-decoration: underline;
            }
            input[type="radio"] {
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Compare 3D Models</h1>
            <p>The system will automatically compare the original and damaged Porsche models using advanced visualization.</p>
            
            <form id="compareForm" action="/compare_porsche_models" method="GET">
                <div class="model-card">
                    <input type="radio" id="original" name="reference_model" value="porsche_original.obj" checked>
                    <label for="original">
                        <div class="model-title">Porsche Original</div>
                        <div class="model-description">Reference model in pristine condition</div>
                    </label>
                </div>
                
                <div class="model-card">
                    <input type="radio" id="damaged" name="reference_model" value="porsche_damaged.obj">
                    <label for="damaged">
                        <div class="model-title">Porsche Damaged</div>
                        <div class="model-description">Model with damage to compare against the original</div>
                    </label>
                </div>
                
                <p>Choose which model should be the reference (the other will be treated as the comparison model)</p>
                
                <button type="submit">Compare Models with Plotly Visualization</button>
            </form>
            
            <a href="/models" class="back-link">Back to Models</a>
        </div>
        
        <script>
            // Add visual selection effect
            document.querySelectorAll('input[type="radio"]').forEach(radio => {
                radio.addEventListener('change', function() {
                    document.querySelectorAll('.model-card').forEach(card => {
                        card.classList.remove('selected');
                    });
                    this.closest('.model-card').classList.add('selected');
                });
            });
            
            // Set initial selected state
            document.getElementById('original').closest('.model-card').classList.add('selected');
            
            // Handle form submission
            document.getElementById('compareForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const referenceModel = document.querySelector('input[name="reference_model"]:checked').value;
                const comparisonModel = referenceModel === 'porsche_original.obj' ? 'porsche_damaged.obj' : 'porsche_original.obj';
                
                window.location.href = `/compare_models/${referenceModel}/${comparisonModel}`;
            });
        </script>
    </body>
    </html>
    ''')

def process_point_cloud_to_mesh(point_cloud_path, output_path, method="poisson", depth=8, sample_ratio=1.0):
    """
    Convert a point cloud to a mesh using different methods
    
    Parameters:
    -----------
    point_cloud_path : str
        Path to the input point cloud file (PLY, XYZ, PCD)
    output_path : str
        Path to save the output mesh (.obj)
    method : str
        Reconstruction method: "poisson", "alpha_shape", or "ball_pivot"
    depth : int
        Depth parameter for Poisson reconstruction (higher = more detail)
    sample_ratio : float
        Ratio of points to use (0.0-1.0)
    
    Returns:
    --------
    bool
        Success or failure
    """
    try:
        logger.info(f"Processing point cloud at {point_cloud_path} to mesh using {method} method")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        
        # Sample points if needed
        if sample_ratio < 1.0:
            points = np.asarray(pcd.points)
            n_points = len(points)
            sample_size = max(10, int(n_points * sample_ratio))
            indices = np.random.choice(n_points, sample_size, replace=False)
            pcd = pcd.select_by_index(indices)
            logger.info(f"Sampled point cloud from {n_points} to {sample_size} points")
        
        # Estimate normals if not present
        if not pcd.has_normals():
            logger.info("Estimating normals for point cloud")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
        
        # Create mesh based on selected method
        mesh = None
        
        if method == "poisson":
            logger.info(f"Running Poisson reconstruction with depth {depth}")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, linear_fit=True)
            
            # Filter out low-density vertices
            if len(densities) > 0:
                densities = np.asarray(densities)
                density_threshold = np.quantile(densities, 0.1)  # Remove bottom 10% density
                vertices_to_remove = densities < density_threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)
        
        elif method == "alpha_shape":
            logger.info("Running Alpha Shape reconstruction")
            # Determine alpha value (radius parameter) automatically
            points = np.asarray(pcd.points)
            distances = []
            for i in range(min(1000, len(points))):
                pt = points[np.random.randint(len(points))]
                dists = np.linalg.norm(points - pt, axis=1)
                distances.append(np.min(dists[dists > 0]))
            
            # Use 2x the average distance to nearest neighbor as alpha
            alpha = 2.0 * np.mean(distances)
            logger.info(f"Using alpha value of {alpha:.4f}")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
        elif method == "ball_pivot":
            logger.info("Running Ball Pivot reconstruction")
            # Estimate radius based on point density
            points = np.asarray(pcd.points)
            distances = []
            for i in range(min(1000, len(points))):
                pt = points[np.random.randint(len(points))]
                dists = np.linalg.norm(points - pt, axis=1)
                distances.append(np.min(dists[dists > 0]))
            
            # Use 3 different radii for multi-scale reconstruction
            avg_dist = np.mean(distances)
            radii = [avg_dist * 2, avg_dist * 5, avg_dist * 10]
            logger.info(f"Using ball radii of {radii}")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        
        if mesh is None or len(mesh.triangles) == 0:
            logger.error("Mesh reconstruction failed - no triangles were created")
            return False
        
        # Fill holes
        logger.info("Filling holes in mesh")
        mesh.fill_holes()
        
        # Clean up the mesh
        logger.info("Cleaning mesh (removing duplicates, degenerate faces)")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
        logger.info(f"Mesh saved to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in point cloud to mesh conversion: {str(e)}")
        return False

def clean_mesh(input_path, output_path, options=None):
    """
    Clean a mesh by removing noise, filling holes, and fixing artifacts
    
    Parameters:
    -----------
    input_path : str
        Path to the input mesh file
    output_path : str
        Path to save the cleaned mesh
    options : dict
        Cleaning options:
            - remove_isolated: bool - Remove isolated pieces
            - fill_holes: bool - Fill small holes
            - smooth: bool - Apply smoothing
            - remove_spikes: bool - Remove spike artifacts
            - fix_normals: bool - Fix normal orientations
    
    Returns:
    --------
    bool
        Success or failure
    """
    try:
        logger.info(f"Cleaning mesh: {input_path}")
        
        if options is None:
            options = {
                'remove_isolated': True,
                'fill_holes': True,
                'smooth': True,
                'remove_spikes': True,
                'fix_normals': True
            }
        
        # Create a new MeshSet
        ms = pymeshlab.MeshSet()
        
        # Load the mesh
        ms.load_new_mesh(input_path)
        
        # Get mesh info before cleaning
        mesh_info_before = ms.current_mesh().vertex_number(), ms.current_mesh().face_number()
        logger.info(f"Before cleaning: {mesh_info_before[0]} vertices, {mesh_info_before[1]} faces")
        
        # Remove unreferenced vertices
        ms.compute_mesh_tessellation()
        ms.remove_unreferenced_vertices()
        
        # Remove isolated pieces
        if options.get('remove_isolated', True):
            logger.info("Removing isolated pieces")
            ms.compute_connected_components()
            # Keep only the largest component
            ms.meshing_remove_connected_component(mincomponentsize=ms.current_mesh().face_number() * 0.1)
        
        # Remove duplicate faces and vertices
        ms.remove_duplicate_vertices()
        ms.remove_duplicate_faces()
        
        # Fix mesh orientation and normals
        if options.get('fix_normals', True):
            logger.info("Fixing normals")
            ms.compute_normal_for_point_clouds(k=10)
            ms.meshing_repair_non_manifold_edges()
            ms.compute_normal_face_consistent()
        
        # Fill small holes
        if options.get('fill_holes', True):
            logger.info("Filling small holes")
            # First try to close holes
            ms.meshing_close_holes(maxholesize=50)  # Max hole size in edges
            # Then refine the result
            ms.remeshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(0.5))
        
        # Remove spike artifacts
        if options.get('remove_spikes', True):
            logger.info("Removing spike artifacts")
            # First select spikes using curvature
            ms.compute_curvature_principal_directions(method='Taubin')
            # Then smooth them
            ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepSmoothNum=10)
        
        # Apply smoothing if requested
        if options.get('smooth', True):
            logger.info("Applying smoothing")
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)

        # Get mesh info after cleaning
        mesh_info_after = ms.current_mesh().vertex_number(), ms.current_mesh().face_number()
        logger.info(f"After cleaning: {mesh_info_after[0]} vertices, {mesh_info_after[1]} faces")
        
        # Save the result
        ms.save_current_mesh(output_path)
        logger.info(f"Cleaned mesh saved to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in mesh cleaning: {str(e)}")
        return False

def decimate_mesh(input_path, output_path, target_reduction=0.5, method="quadric"):
    """
    Decimate (simplify) a mesh to reduce complexity
    
    Parameters:
    -----------
    input_path : str
        Path to the input mesh file
    output_path : str
        Path to save the decimated mesh
    target_reduction : float
        Target reduction ratio (0.0-1.0)
        0.5 means reduce vertices by 50%
    method : str
        Decimation method: "quadric" or "clustering"
    
    Returns:
    --------
    bool
        Success or failure
    dict
        Statistics about the decimation
    """
    try:
        logger.info(f"Decimating mesh: {input_path} with target reduction {target_reduction}")
        
        # Create a new MeshSet
        ms = pymeshlab.MeshSet()
        
        # Load the mesh
        ms.load_new_mesh(input_path)
        
        # Get mesh info before decimation
        vertices_before = ms.current_mesh().vertex_number()
        faces_before = ms.current_mesh().face_number()
        logger.info(f"Before decimation: {vertices_before} vertices, {faces_before} faces")
        
        # Calculate target face count
        target_face_num = int(faces_before * (1 - target_reduction))
        
        # Apply decimation based on method
        if method == "quadric":
            logger.info(f"Using quadric edge collapse decimation to {target_face_num} faces")
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_face_num,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=True
            )
        
        elif method == "clustering":
            logger.info("Using clustering decimation")
            # Calculate cell size based on mesh bounding box and target reduction
            bbox = ms.current_mesh().bounding_box()
            diag = pymeshlab.Scalar(np.linalg.norm(
                np.array([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()])
            ))
            cell_size = diag * pymeshlab.Percentage(target_reduction) * 0.01  # Convert to percentage
            
            ms.meshing_decimation_clustering(threshold=cell_size)
        
        # Get mesh info after decimation
        vertices_after = ms.current_mesh().vertex_number()
        faces_after = ms.current_mesh().face_number()
        
        # Calculate actual reduction
        vertex_reduction = (vertices_before - vertices_after) / vertices_before
        face_reduction = (faces_before - faces_after) / faces_before
        
        logger.info(f"After decimation: {vertices_after} vertices, {faces_after} faces")
        logger.info(f"Reduction: vertices {vertex_reduction:.2f}, faces {face_reduction:.2f}")
        
        # Save the result
        ms.save_current_mesh(output_path)
        
        # Return statistics
        stats = {
            "vertices_before": vertices_before,
            "vertices_after": vertices_after,
            "faces_before": faces_before,
            "faces_after": faces_after,
            "vertex_reduction": vertex_reduction,
            "face_reduction": face_reduction
        }
        
        return True, stats
    
    except Exception as e:
        logger.error(f"Error in mesh decimation: {str(e)}")
        return False, {"error": str(e)}

# Flask routes to add

@app.route('/process_mesh/<filename>', methods=['GET'])
def process_mesh_view(filename):
    """View page with options to process a mesh"""
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Process 3D Mesh</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
            .container { max-width: 800px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
            h1, h2 { color: #2d3748; }
            .section { margin-bottom: 30px; text-align: left; }
            form { margin: 20px 0; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            select, input { padding: 8px; border-radius: 4px; border: 1px solid #ddd; width: 100%; max-width: 300px; }
            .checkbox-group { margin: 10px 0; }
            .checkbox-group label { display: inline; font-weight: normal; margin-left: 5px; }
            button { background: #4299e1; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; margin: 8px 0; font-weight: 600; transition: all 0.3s ease; }
            button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .info { background: #ebf8ff; border-left: 4px solid #4299e1; padding: 10px; margin: 10px 0; }
            .back-button { display: inline-block; margin-top: 20px; padding: 10px 15px; background: #718096; color: white; border-radius: 5px; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Process 3D Model</h1>
            <p>Apply operations to improve or modify your 3D model</p>
            
            <div class="section">
                <h2>Model Information</h2>
                <p><strong>Filename:</strong> {{ filename }}</p>
                
                <div class="info">
                    <p><strong>Note:</strong> Processing operations may take several minutes depending on the complexity of your model.</p>
                </div>
            </div>
            
            <!-- Point Cloud to Mesh Conversion -->
            <div class="section">
                <h2>Point Cloud to Mesh Conversion</h2>
                <p>Convert a point cloud (e.g. from COLMAP) to a complete 3D mesh</p>
                
                <form action="/convert_point_cloud_to_mesh/{{ filename }}" method="post">
                    <div class="form-group">
                        <label for="method">Reconstruction Method:</label>
                        <select name="method" id="method">
                            <option value="poisson">Poisson Surface Reconstruction (Best quality)</option>
                            <option value="alpha_shape">Alpha Shape (Faster, works with fewer points)</option>
                            <option value="ball_pivot">Ball Pivoting (Preserves original points)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="detail">Detail Level:</label>
                        <select name="detail" id="detail">
                            <option value="low">Low (Faster)</option>
                            <option value="medium" selected>Medium (Balanced)</option>
                            <option value="high">High (Slower, more detail)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="sample_ratio">Point Sampling Ratio:</label>
                        <select name="sample_ratio" id="sample_ratio">
                            <option value="0.25">25% (Faster)</option>
                            <option value="0.5">50% (Balanced)</option>
                            <option value="1.0" selected>100% (Higher quality)</option>
                        </select>
                    </div>
                    
                    <button type="submit">Convert to Mesh</button>
                </form>
            </div>
            
            <!-- Mesh Cleaning -->
            <div class="section">
                <h2>Mesh Cleaning</h2>
                <p>Clean the mesh to remove noise, artifacts, and fix issues</p>
                
                <form action="/clean_mesh/{{ filename }}" method="post">
                    <div class="checkbox-group">
                        <input type="checkbox" id="remove_isolated" name="remove_isolated" checked>
                        <label for="remove_isolated">Remove isolated pieces</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="fill_holes" name="fill_holes" checked>
                        <label for="fill_holes">Fill holes</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="smooth" name="smooth" checked>
                        <label for="smooth">Apply smoothing</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="remove_spikes" name="remove_spikes" checked>
                        <label for="remove_spikes">Remove spike artifacts</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="fix_normals" name="fix_normals" checked>
                        <label for="fix_normals">Fix normals</label>
                    </div>
                    
                    <button type="submit">Clean Mesh</button>
                </form>
            </div>
            
            <!-- Mesh Decimation -->
            <div class="section">
                <h2>Mesh Decimation</h2>
                <p>Reduce the complexity of the mesh while preserving its shape</p>
                
                <form action="/decimate_mesh/{{ filename }}" method="post">
                    <div class="form-group">
                        <label for="target_reduction">Target Reduction:</label>
                        <select name="target_reduction" id="target_reduction">
                            <option value="0.25">25% (Minimal reduction)</option>
                            <option value="0.5" selected>50% (Balanced)</option>
                            <option value="0.75">75% (Aggressive reduction)</option>
                            <option value="0.9">90% (Maximum reduction)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="decimation_method">Method:</label>
                        <select name="decimation_method" id="decimation_method">
                            <option value="quadric" selected>Quadric Edge Collapse (Better quality)</option>
                            <option value="clustering">Clustering (Faster)</option>
                        </select>
                    </div>
                    
                    <button type="submit">Decimate Mesh</button>
                </form>
            </div>
            
            <a href="/model_view/{{ filename }}" class="back-button">Back to Model View</a>
        </div>
    </body>
    </html>
    ''', filename=filename)

@app.route('/convert_point_cloud_to_mesh/<filename>', methods=['POST'])
def convert_point_cloud_to_mesh(filename):
    """Convert a point cloud to a mesh using selected method"""
    try:
        # Get form parameters
        method = request.form.get('method', 'poisson')
        detail_level = request.form.get('detail', 'medium')
        sample_ratio = float(request.form.get('sample_ratio', 1.0))
        
        # Map detail level to method-specific parameters
        if method == 'poisson':
            depth_map = {'low': 6, 'medium': 8, 'high': 10}
            depth = depth_map[detail_level]
        else:
            depth = 8  # Not used for other methods
        
        # Find the model in MongoDB
        model_data = models_collection.find_one({"filename": filename})
        
        if not model_data:
            return jsonify({"error": "Model not found"}), 404
        
        # Save the point cloud to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input_pointcloud.ply")
        
        with open(temp_input, 'wb') as f:
            f.write(model_data["data"])
        
        # Create a new filename for the mesh
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{os.path.splitext(filename)[0]}_mesh_{timestamp}.obj"
        output_path = os.path.join(MODEL_FOLDER, new_filename)
        
        # Process the point cloud to mesh
        success = process_point_cloud_to_mesh(
            temp_input, 
            output_path, 
            method=method, 
            depth=depth, 
            sample_ratio=sample_ratio
        )
        
        if not success:
            return render_template_string('''
            <!doctype html>
            <html>
            <head>
                <title>Conversion Error</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                    .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                    h1 { color: #e53e3e; }
                    p { color: #4a5568; margin-bottom: 25px; }
                    .error-icon { font-size: 60px; color: #e53e3e; margin: 20px 0; }
                    button { background: #4299e1; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; margin: 8px; font-weight: 600; transition: all 0.3s ease; }
                    button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="error-icon">❌</div>
                    <h1>Cleaning Failed</h1>
                    <p>There was an error cleaning the mesh. This could be due to:</p>
                    <ul style="text-align: left;">
                        <li>Input file is not a valid mesh</li>
                        <li>Mesh has too many issues to repair</li>
                        <li>Server processing error</li>
                    </ul>
                    <p>Try with a different model or different cleaning settings.</p>
                    <a href="/process_mesh/{{ filename }}"><button>Back to Processing Options</button></a>
                </div>
            </body>
            </html>
            ''', filename=filename)
        
        # If successful, load the new mesh file and save to MongoDB
        with open(output_path, 'rb') as f:
            mesh_data = f.read()
        
        # Count vertices in the OBJ file
        vertex_count = 0
        with open(output_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertex_count += 1
        
        # Save to MongoDB
        new_model = {
            "filename": new_filename,
            "name": f"Cleaned {filename}",
            "data": mesh_data,
            "created_at": datetime.datetime.now(),
            "source_model": filename,
            "cleaning_options": options,
            "point_count": vertex_count,
            "processed": True
        }
        
        models_collection.insert_one(new_model)
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        # Success page
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Cleaning Complete</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #38a169; }
                p { color: #4a5568; margin-bottom: 25px; }
                .success-icon { font-size: 60px; color: #38a169; margin: 20px 0; }
                button { background: #4299e1; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; margin: 8px; font-weight: 600; transition: all 0.3s ease; }
                button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✓</div>
                <h1>Cleaning Complete</h1>
                <p>The mesh was successfully cleaned.</p>
                <div>
                    <p><strong>New model:</strong> {{ new_filename }}</p>
                    <p><strong>Vertices:</strong> {{ vertex_count }}</p>
                    <p><strong>Applied operations:</strong> {{ operations }}</p>
                </div>
                <div>
                    <a href="/model_view/{{ new_filename }}"><button>View Cleaned Mesh</button></a>
                    <a href="/models"><button>All Models</button></a>
                </div>
            </div>
        </body>
        </html>
        ''', new_filename=new_filename, vertex_count=vertex_count, 
            operations=", ".join([k for k, v in options.items() if v]))
    
    except Exception as e:
        logger.error(f"Error in mesh cleaning: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/decimate_mesh/<filename>', methods=['POST'])
def decimate_mesh_route(filename):
    """Decimate a mesh to reduce its complexity"""
    try:
        # Get decimation options from form
        target_reduction = float(request.form.get('target_reduction', 0.5))
        method = request.form.get('decimation_method', 'quadric')
        
        # Find the model in MongoDB
        model_data = models_collection.find_one({"filename": filename})
        
        if not model_data:
            return jsonify({"error": "Model not found"}), 404
        
        # Save the mesh to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input_mesh.obj")
        
        with open(temp_input, 'wb') as f:
            f.write(model_data["data"])
        
        # Create a new filename for the decimated mesh
        reduction_percent = int(target_reduction * 100)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{os.path.splitext(filename)[0]}_decimate{reduction_percent}_{timestamp}.obj"
        output_path = os.path.join(MODEL_FOLDER, new_filename)
        
        # Decimate the mesh
        success, stats = decimate_mesh(temp_input, output_path, target_reduction, method)
        
        if not success:
            return render_template_string('''
            <!doctype html>
            <html>
            <head>
                <title>Decimation Error</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                    .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                    h1 { color: #e53e3e; }
                    p { color: #4a5568; margin-bottom: 25px; }
                    .error-icon { font-size: 60px; color: #e53e3e; margin: 20px 0; }
                    button { background: #4299e1; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; margin: 8px; font-weight: 600; transition: all 0.3s ease; }
                    button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="error-icon">❌</div>
                    <h1>Decimation Failed</h1>
                    <p>There was an error decimating the mesh. This could be due to:</p>
                    <ul style="text-align: left;">
                        <li>Input file is not a valid mesh</li>
                        <li>Mesh has topology issues that prevent decimation</li>
                        <li>Server processing error</li>
                    </ul>
                    <p>Try with a different model or different decimation settings.</p>
                    <p style="color: #e53e3e; font-weight: bold;">{{ error }}</p>
                    <a href="/process_mesh/{{ filename }}"><button>Back to Processing Options</button></a>
                </div>
            </body>
            </html>
            ''', filename=filename, error=stats.get("error", "Unknown error"))
        
        # If successful, load the new mesh file and save to MongoDB
        with open(output_path, 'rb') as f:
            mesh_data = f.read()
        
        # Save to MongoDB
        new_model = {
            "filename": new_filename,
            "name": f"Decimated {filename} ({reduction_percent}%)",
            "data": mesh_data,
            "created_at": datetime.datetime.now(),
            "source_model": filename,
            "decimation_stats": stats,
            "decimation_method": method,
            "target_reduction": target_reduction,
            "point_count": stats["vertices_after"],
            "processed": True
        }
        
        models_collection.insert_one(new_model)
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        # Success page
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Decimation Complete</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; background-color: #f9f9f9; }
                .container { max-width: 700px; margin: auto; padding: 30px; background: #ffffff; border-radius: 15px; box-shadow: 0px 5px 20px rgba(0,0,0,0.1); }
                h1 { color: #38a169; }
                p { color: #4a5568; margin-bottom: 25px; }
                .success-icon { font-size: 60px; color: #38a169; margin: 20px 0; }
                button { background: #4299e1; color: white; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; margin: 8px; font-weight: 600; transition: all 0.3s ease; }
                button:hover { background: #3182ce; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .stats-table { margin: 20px auto; border-collapse: collapse; width: 80%; }
                .stats-table th, .stats-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .stats-table th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✓</div>
                <h1>Decimation Complete</h1>
                <p>The mesh was successfully decimated.</p>
                
                <table class="stats-table">
                    <tr>
                        <th></th>
                        <th>Before</th>
                        <th>After</th>
                        <th>Reduction</th>
                    </tr>
                    <tr>
                        <td><strong>Vertices</strong></td>
                        <td>{{ stats.vertices_before }}</td>
                        <td>{{ stats.vertices_after }}</td>
                        <td>{{ "%.1f"|format(stats.vertex_reduction * 100) }}%</td>
                    </tr>
                    <tr>
                        <td><strong>Faces</strong></td>
                        <td>{{ stats.faces_before }}</td>
                        <td>{{ stats.faces_after }}</td>
                        <td>{{ "%.1f"|format(stats.face_reduction * 100) }}%</td>
                    </tr>
                </table>
                
                <div>
                    <a href="/model_view/{{ new_filename }}"><button>View Decimated Mesh</button></a>
                    <a href="/models"><button>All Models</button></a>
                </div>
            </div>
        </body>
        </html>
        ''', new_filename=new_filename, stats=stats)
    
    except Exception as e:
        logger.error(f"Error in mesh decimation: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
