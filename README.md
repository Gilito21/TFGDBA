# Video to 3D model Platform

Video to 3D Model Platform is a Python-based application that converts video files into 3D models by extracting frames, running COLMAP for 3D reconstruction, and storing results in MongoDB.


## Features

- Frame Extraction: Capture frames from any input video at specified intervals (via OpenCV)
- 3D Reconstruction: Run COLMAP to generate point clouds and meshes from extracted frames
- Store frames and 3D models (PLY or OBJ) in MongoDB (using GridFS or references)
- Multiple Reconstruction Options: Potential integration with Meshroom or NeRF for advanced 3D reconstruction
- Model Comparison: Compare two 3D models and highlight differences


## Installation

1. Install TFGDBA with git

```bash
  git clone https://github.com/Gilito21/TFGDBA
  cd TFGDBA
```
2. Install dependencies
```bash
  pip install -r requirements.txt
``` 
3. Set up MongoDB (if not already running)
```bash
sudo service mongodb start
```
4. Ensure COLMAP is installed
- Download latest version. Connect to pycolmap
- https://github.com/colmap/colmap
## Usage/Examples

```python
python app.py
```
- Open on port 5000
- Use web interface to interact with the app

## Roadmap
- Web cloud deployment
- Things to add in the future


## Tech Stack

**Client:** HTML / CSS / Javascript. Three.js

**Server:** Python / Flask / OpenCV / COLMAP / Lambda Cloud GPU

**DataBase:** MongoDB


## Authors

- Juan Pelaez Echaniz [@Gilito21](https://www.github.com/Gilito21) - Main Developer


## Demo

Insert demo when created
