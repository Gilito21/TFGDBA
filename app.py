import os
import cv2
import pymongo
import numpy as np
import tempfile
import datetime
from flask import Flask, request, render_template_string, jsonify, Response, send_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connect to MongoDB using environment variable
MONGO_URI = "mongodb+srv://juanp:iGy1RQfwvSKuVlHh@cluster0.iiks7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["video_frames"]
frames_collection = db["frames"]
models_collection = db["models"]

def extract_frames(video_path, output_folder, frame_interval=5):
    """Extract frames from a video file and save to MongoDB"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
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
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 600px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            input[type="file"] { margin: 10px 0; }
            input[type="submit"], button { background: #007BFF; color: white; border: none; padding: 10px; cursor: pointer; border-radius: 5px; margin: 5px; }
            input[type="submit"]:hover, button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="color: #007BFF;">3D Reconstruction from Video</h1>
            <p>Upload a video to extract frames and create a 3D model.</p>
            
            <form action="/process_video" method="post" enctype="multipart/form-data">
                <input type="file" name="video" required><br>
                <select name="frame_interval">
                    <option value="1">Every frame</option>
                    <option value="5" selected>Every 5th frame</option>
                    <option value="10">Every 10th frame</option>
                    <option value="20">Every 20th frame</option>
                </select><br><br>
                <input type="submit" value="Extract Frames">
            </form>
            
            <div style="margin-top: 20px;">
                <a href="/frames"><button>View Extracted Frames</button></a>
                <a href="/models"><button style="background: #28a745;">View 3D Models</button></a>
            </div>
        </div>
    </body>
    </html>
    ''')

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
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the video: {str(e)}"}), 500
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Processing Complete</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 600px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            button { display: inline-block; padding: 10px 20px; font-size: 16px; color: white; background: #007BFF; text-decoration: none; border-radius: 5px; margin: 5px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Frames Extracted Successfully</h1>
            <p>Extracted {{ count }} frames from your video.</p>
            
            <div>
                <a href="/frames"><button>View Frames</button></a>
                <a href="/create_model"><button style="background: #28a745;">Create 3D Model</button></a>
            </div>
        </div>
    </body>
    </html>
    ''', count=extracted_count)

def get_frame_data_from_mongo():
    """Retrieve all frame filenames from MongoDB"""
    frames = list(frames_collection.find({}, {"filename": 1, "_id": 0}))
    return [frame["filename"] for frame in frames]

@app.route('/frames')
def list_frames():
    """List all stored frames with option to create 3D model"""
    frame_list = get_frame_data_from_mongo()
    
    if not frame_list:
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>No Frames Available</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
                .container { max-width: 600px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                button { display: inline-block; padding: 10px 20px; font-size: 16px; color: white; background: #007BFF; text-decoration: none; border-radius: 5px; margin: 5px; border: none; cursor: pointer; }
                button:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>No Frames Available</h1>
                <p>You need to upload and process a video first.</p>
                <a href="/"><button>Back to Upload</button></a>
            </div>
        </body>
        </html>
        ''')
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Extracted Frames</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 800px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .frame-container { display: flex; flex-wrap: wrap; justify-content: center; }
            .frame-item { margin: 10px; }
            img { width: 150px; height: auto; border: 1px solid #ddd; border-radius: 5px; }
            button { display: inline-block; padding: 10px 20px; font-size: 16px; color: white; background: #007BFF; text-decoration: none; border-radius: 5px; margin: 5px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .create-btn { background: #28a745; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Extracted Frames</h1>
            <p>{{ frames|length }} frames available</p>
            
            <div>
                <a href="/"><button>Back to Upload</button></a>
                <a href="/create_model"><button class="create-btn">Create 3D Model</button></a>
            </div>
            
            <div class="frame-container">
                {% for frame in frames %}
                    <div class="frame-item">
                        <a href="/frame/{{ frame }}" target="_blank">
                            <img src="/frame/{{ frame }}" alt="{{ frame }}">
                        </a>
                        <div>{{ frame }}</div>
                    </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    ''', frames=frame_list)

def detect_features(image):
    """Detect features in an image using SIFT"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two images"""
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches

def find_camera_matrices(kp1, kp2, matches, K):
    """Find camera matrices from matching points"""
    # Convert keypoints to numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    # Find essential matrix
    E = K.T @ F @ K
    
    # Recover pose
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    # Create projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    
    return P1, P2, pts1, pts2

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate 3D points from 2D correspondences"""
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d_homogeneous / points_4d_homogeneous[3]
    return points_3d[:3].T

@app.route('/create_model')
def create_model():
    """Create a 3D model from the extracted frames using structure from motion"""
    frame_list = get_frame_data_from_mongo()
    
    if not frame_list or len(frame_list) < 2:
        return jsonify({"error": "Need at least two frames to create a 3D model"}), 400
    
    try:
        # Load frames
        frames = []
        for frame_name in frame_list[:10]:  # Limit to first 10 frames for simplicity
            frame_data = frames_collection.find_one({"filename": frame_name})
            if frame_data:
                image_array = np.frombuffer(frame_data["data"], np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                frames.append(image)
        
        if len(frames) < 2:
            return jsonify({"error": "Failed to load at least two valid frames"}), 400
        
        # Initialize camera matrix (estimate)
        height, width = frames[0].shape[:2]
        focal_length = width
        camera_matrix = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])
        
        # Detect features in all frames
        all_keypoints = []
        all_descriptors = []
        for frame in frames:
            keypoints, descriptors = detect_features(frame)
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)
        
        # Initialize 3D points list
        all_points_3d = []
        
        # Process sequential pairs of frames
        for i in range(len(frames) - 1):
            # Match features between consecutive frames
            matches = match_features(all_descriptors[i], all_descriptors[i+1])
            
            if len(matches) < 8:
                continue  # Not enough matches
            
            # Find camera matrices and triangulate points
            P1, P2, pts1, pts2 = find_camera_matrices(all_keypoints[i], all_keypoints[i+1], matches, camera_matrix)
            points_3d = triangulate_points(P1, P2, pts1, pts2)
            
            # Filter points by z-value (depth)
            points_3d = points_3d[points_3d[:, 2] > 0]
            all_points_3d.append(points_3d)
        
        if not all_points_3d:
            return jsonify({"error": "Failed to reconstruct 3D points from the frames"}), 400
        
        # Combine all 3D points
        combined_points = np.vstack(all_points_3d)
        
        # Generate a unique name for the model
        model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
        model_path = os.path.join(MODEL_FOLDER, model_name)
        
        # Save 3D points as OBJ file
        with open(model_path, 'w') as f:
            f.write("# OBJ file created by Flask app\n")
            for point in combined_points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
        # Create a visualization of the 3D points
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2], s=1, c=combined_points[:, 2], cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Reconstruction')
        
        # Save the visualization
        visualization_path = os.path.join(MODEL_FOLDER, f"{model_name.split('.')[0]}_viz.png")
        plt.savefig(visualization_path)
        plt.close()
        
        # Store model in MongoDB
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        with open(visualization_path, 'rb') as f:
            viz_data = f.read()
        
        models_collection.insert_one({
            "name": model_name,
            "data": model_data,
            "visualization": viz_data,
            "frame_count": len(frames),
            "point_count": len(combined_points),
            "created_at": datetime.datetime.now()
        })
        
        return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>3D Model Created</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
                .container { max-width: 700px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                img { max-width: 600px; max-height: 500px; margin: 20px auto; border: 1px solid #ddd; border-radius: 5px; }
                button { display: inline-block; padding: 10px 20px; font-size: 16px; color: white; background: #007BFF; text-decoration: none; border-radius: 5px; margin: 5px; border: none; cursor: pointer; }
                button:hover { background: #0056b3; }
                .info { text-align: left; margin: 20px auto; max-width: 500px; }
                .info p { margin: 5px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>3D Model Created Successfully</h1>
                
                <div class="info">
                    <p><strong>Model name:</strong> {{ model_name }}</p>
                    <p><strong>Points:</strong> {{ point_count }}</p>
                    <p><strong>Frames used:</strong> {{ frame_count }}</p>
                </div>
                
                <img src="/model_viz/{{ model_name.split('.')[0] }}" alt="3D Model Visualization">
                
                <div>
                    <a href="/model/{{ model_name }}" download><button>Download OBJ File</button></a>
                    <a href="/models"><button>View All Models</button></a>
                    <a href="/"><button>Back to Home</button></a>
                </div>
            </div>
        </body>
        </html>
        ''', model_name=model_name, point_count=len(combined_points), frame_count=len(frames))
        
    except Exception as e:
        app.logger.error(f"Error creating 3D model: {str(e)}")
        return jsonify({"error": f"Failed to create 3D model: {str(e)}"}), 500

@app.route('/models')
def list_models():
    """List all created 3D models"""
    model_list = list(models_collection.find({}, {"name": 1, "_id": 0, "created_at": 1, "frame_count": 1, "point_count": 1}))
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>3D Models</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 800px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            button { display: inline-block; padding: 10px 20px; font-size: 16px; color: white; background: #007BFF; text-decoration: none; border-radius: 5px; margin: 5px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .no-models { padding: 20px; text-align: center; }
            .thumbnail { width: 80px; height: 60px; object-fit: cover; }
            .actions a { margin-right: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>3D Models</h1>
            
            <div>
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
                    <p>Go to the frames page to create a 3D model.</p>
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
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
    temp_file.write(model_data["data"])
    temp_file.close()
    
    return send_file(temp_file.name, as_attachment=True, 
                    download_name=filename, 
                    mimetype='application/x-tgif')

@app.route('/model_viz/<model_name>')
def get_model_viz(model_name):
    """Retrieve and display a model visualization"""
    # Add .obj extension to search in the database
    model_data = models_collection.find_one({"name": f"{model_name}.obj"})
    
    if not model_data or "visualization" not in model_data:
        return "Model visualization not found", 404

    viz_data = model_data["visualization"]
    return Response(viz_data, mimetype="image/png")

@app.route('/model_view/<filename>')
def view_model(filename):
    """View a 3D model in a simple viewer"""
    model_data = models_collection.find_one({"name": filename})
    
    if not model_data:
        return "Model not found", 404
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>3D Model Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .header { background-color: #333; color: white; padding: 10px; text-align: center; }
            .container { display: flex; height: calc(100vh - 40px); }
            .sidebar { width: 250px; padding: 20px; background-color: #f4f4f4; overflow: auto; }
            .viewer { flex-grow: 1; }
            button { display: inline-block; padding: 10px; width: 100%; margin: 5px 0; background: #007BFF; color: white; border: none; cursor: pointer; border-radius: 5px; }
            button:hover { background: #0056b3; }
            .model-info p { margin: 5px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>3D Model Viewer: {{ model.name }}</h1>
        </div>
        <div class="container">
            <div class="sidebar">
                <div class="model-info">
                    <h3>Model Information</h3>
                    <p><strong>Created:</strong> {{ model.created_at.strftime('%Y-%m-%d %H:%M') }}</p>
                    <p><strong>Points:</strong> {{ model.point_count }}</p>
                    <p><strong>Frames used:</strong> {{ model.frame_count }}</p>
                </div>
                <a href="/model/{{ model.name }}" download><button>Download OBJ</button></a>
                <a href="/models"><button>Back to Models</button></a>
                <a href="/"><button>Back to Home</button></a>
            </div>
            <div class="viewer">
                <img src="/model_viz/{{ model.name.split('.')[0] }}" alt="3D Model Visualization" style="max-width: 100%; max-height: 100%; object-fit: contain;">
            </div>
        </div>
    </body>
    </html>
    ''', model=model_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)