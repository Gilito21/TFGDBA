from flask import Flask, request, jsonify
import boto3  # AWS S3 for storage
import subprocess
import os
import cv2
import numpy as np
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# AWS S3 Config
S3_BUCKET = "your-s3-bucket"
S3_ACCESS_KEY = "your-access-key"
S3_SECRET_KEY = "your-secret-key"
UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

s3 = boto3.client("s3", aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY)

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    video = request.files["video"]
    filename = secure_filename(video.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(file_path)
    
    # Upload to S3
    s3.upload_file(file_path, S3_BUCKET, filename)
    video_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{filename}"
    
    return jsonify({"message": "Video uploaded", "video_url": video_url})

@app.route("/process", methods=["POST"])
def process_frames():
    frame_folder = os.path.join(UPLOAD_FOLDER, "frames")
    os.makedirs(frame_folder, exist_ok=True)
    
    # Extract frames using OpenCV
    video_path = os.path.join(UPLOAD_FOLDER, request.json["video_filename"])
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(frame_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    
    return jsonify({"message": "Frames extracted", "total_frames": frame_count})

@app.route("/generate_model", methods=["POST"])
def generate_3d_model():
    frame_folder = os.path.join(UPLOAD_FOLDER, "frames")
    output_model = os.path.join(MODEL_FOLDER, "output_model.ply")
    
    # Run COLMAP SfM pipeline
    subprocess.run(["colmap", "feature_extractor", "--database_path", "database.db", "--image_path", frame_folder])
    subprocess.run(["colmap", "mapper", "--database_path", "database.db", "--image_path", frame_folder, "--output_path", "sparse"])
    
    # Convert to dense model
    subprocess.run(["colmap", "image_undistorter", "--image_path", frame_folder, "--input_path", "sparse", "--output_path", "dense"])
    subprocess.run(["colmap", "patch_match_stereo", "--workspace_path", "dense", "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"])
    subprocess.run(["colmap", "stereo_fusion", "--workspace_path", "dense", "--workspace_format", "COLMAP", "--output_path", output_model])
    
    return jsonify({"message": "3D model generated", "model": output_model})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
