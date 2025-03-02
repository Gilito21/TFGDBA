import os
import cv2
import pymongo
import base64
from flask import Flask, request, render_template_string, jsonify, Response

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connect to MongoDB using environment variable
MONGO_URI = "mongodb+srv://juanp:iGy1RQfwvSKuVlHh@cluster0.iiks7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["video_frames"]
frames_collection = db["frames"]

def extract_frames(video_path, output_folder, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
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
        frame_count += 1
    cap.release()

@app.route('/')
def upload_form():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Upload Video</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 600px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            input[type="file"] { margin: 10px 0; }
            input[type="submit"] { background: #007BFF; color: white; border: none; padding: 10px; cursor: pointer; border-radius: 5px; }
            input[type="submit"]:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="color: #007BFF;">Upload a Video to Extract Frames</h1>
            <p style="color: red;">Please ensure the video is in a supported format (e.g., MP4, AVI, MOV).</p>
            <form action="/process_video" method="post" enctype="multipart/form-data">
                <input type="file" name="video" required><br><br>
                <input type="submit" value="Extract Frames">
            </form>
                <form action="/create_model" method="post">
                <input type="submit" value="Create 3D Model">
            </form>
            <div>
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
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    try:
        extract_frames(video_path, FRAME_FOLDER)
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({"error": "An error occurred while processing the video. Please try again."}), 500
    
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Processing Complete</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 600px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            a.button { display: inline-block; padding: 10px 20px; font-size: 16px; color: white; background: #007BFF; text-decoration: none; border-radius: 5px; }
            a.button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Frames Extracted Successfully</h1>
            <p>Your video has been processed. Click below to view extracted frames.</p>
            <a href="/frames" class="button">View Frames</a>
        </div>
    </body>
    </html>
    ''')

@app.route('/create_model', methods=['POST'])
@app.route('/create_model', methods=['POST'])
def create_model():
    """ Create a 3D model from the extracted frames """
    frames = retrieve_frames()
    if not frames:
        return jsonify({"error": "No frames found"}), 404
    
    # Placeholder for the actual implementation
    # Here you would add the photogrammetry processing logic
    # For example, using Open3D or another library to create the 3D model
    # This is a placeholder for the actual implementation
    obj_file_path = "path/to/generated_model.obj"
    
    return jsonify({"message": "3D model created successfully", "model_path": obj_file_path}), 200

@app.route('/frames')
def retrieve_frames():
    """ Retrieve all stored frames from MongoDB """
    frames = frames_collection.find({}, {"filename": 1, "_id": 0})
    return [frame["filename"] for frame in frames]

def list_frames():
    """ List all stored frames """
    frames = frames_collection.find({}, {"filename": 1, "_id": 0})
    frame_list = retrieve_frames()
    return render_template_string(''' 
    <h1>Extracted Frames</h1>
    <form action="/create_model" method="post">
        <input type="submit" value="Create 3D Model">
    </form>
    <form action="/create_model" method="post">
        <input type="submit" value="Create 3D Model">
    </form>
    <!doctype html>
    <html>
    <head>
        <title>Saved Frames</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 800px; margin: auto; padding: 20px; background: #f4f4f4; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            img { width: 100px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Extracted Frames</h1>
            {% for frame in frames %}
                <a href="/frame/{{ frame }}" download>
                    <img src="/frame/{{ frame }}" alt="{{ frame }}">
                </a>
            {% endfor %}
        </div>
    </body>
    </html>
    ''', frames=frame_list)

@app.route('/frame/<filename>')
def get_frame(filename):
    """ Retrieve and display a specific frame """
    frame_data = frames_collection.find_one({"filename": filename})
    
    if not frame_data:
        return "Frame not found", 404

    image_data = frame_data["data"]
    return Response(image_data, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
