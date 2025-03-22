<div align="center">

#  TFGDBA Video to 3D Model 

### *Transform Videos into Stunning 3D Models*


[![Last Commit](https://img.shields.io/github/last-commit/gilito21/TFGDBA?style=for-the-badge)](https://github.com/gilito21/TFGDBA/commits)
[![Docker Pulls](https://img.shields.io/docker/pulls/tiogilito21/tfgdba-app?style=for-the-badge&logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/tiogilito21/tfgdba-app)
[![License](https://img.shields.io/github/license/gilito21/TFGDBA?style=for-the-badge&color=green)](LICENSE)

</div>

<div align="center">

[![Express](https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express&logoColor=white)](https://expressjs.com/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![npm](https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white)](https://www.npmjs.com/)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://gunicorn.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Three.js](https://img.shields.io/badge/Three.js-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)

</div>

## 📋 Overview

TFGDBA is a powerful application that transforms ordinary videos into detailed 3D models. Using advanced computer vision algorithms and GPU acceleration, it extracts frames, analyzes spatial information, and generates high-quality 3D representations for a variety of use cases.

<div align="center">
<table>
<tr>
<td width="50%">
<img src="https://raw.githubusercontent.com/gilito21/TFGDBA/main/assets/example1.gif" alt="Example 1"/>
</td>
<td width="50%">
<img src="https://raw.githubusercontent.com/gilito21/TFGDBA/main/assets/example2.gif" alt="Example 2"/>
</td>
</tr>
</table>
</div>

## ✨ Features

<div align="center">
<table>
<tr>
<td>
<h3>🎮 3D Model Generation</h3>
Convert videos to detailed, manipulable 3D models
</td>
<td>
<h3>📊 Progress Tracking</h3>
Real-time tracking of processing stages
</td>
</tr>
<tr>
<td>
<h3>💾 Database Integration</h3>
MongoDB storage for models and processing data
</td>
<td>
<h3>🐳 Docker Support</h3>
Containerized deployment for consistent environments
</td>
</tr>
<tr>
<td colspan="2" align="center">
<h3>🌐 Flask-Based Web Application</h3>
Intuitive interface for uploading videos and viewing models
</td>
</tr>
</table>
</div>

## 🚀 Installation

### Prerequisites
- GPU-enabled environment (Lambda Cloud recommended)
- Docker and Docker Compose
- SSH access to your server

### Setup Steps

1. **Connect to Lambda Cloud GPU**
   ```bash
   ssh -i your-key.pem ubuntu@new-lambda-ip
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/Gilito21/TFGDBA.git
   ```

3. **Navigate to Project Directory**
   ```bash
   cd TFGDBA
   ```

4. **Make Setup Script Executable**
   ```bash
   chmod +x setup-app.sh
   ```

5. **Run Setup Script (Pulls from Docker Hub)**
   ```bash
   ./setup-app.sh
   ```

## 🏗️ Architecture

<div align="center">
<img src="data_flow_diagram.drawio.png" alt="Architecture Diagram" width="800"/>
</div>

## 💻 Technology Stack

<table>
<tr>
<th>Layer</th>
<th>Technologies</th>
</tr>
<tr>
<td>Frontend</td>
<td>
  <ul>
    <li>HTML5 / CSS3 / JavaScript</li>
    <li>Three.js for 3D rendering</li>
    <li>Responsive design</li>
  </ul>
</td>
</tr>
<tr>
<td>Backend</td>
<td>
  <ul>
    <li>Python / Flask</li>
    <li>OpenCV for video processing</li>
    <li>COLMAP for 3D reconstruction</li>
    <li>Lambda Cloud GPU for processing</li>
  </ul>
</td>
</tr>
<tr>
<td>Database</td>
<td>
  <ul>
    <li>MongoDB for model and metadata storage</li>
  </ul>
</td>
</tr>
<tr>
<td>Deployment</td>
<td>
  <ul>
    <li>Docker containers</li>
    <li>Docker Compose for orchestration</li>
  </ul>
</td>
</tr>
</table>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

<div align="center">
<a href="https://github.com/gilito21/TFGDBA/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=gilito21/TFGDBA" />
</a>
</div>

<div align="center">

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gilito21/TFGDBA&type=Date)](https://star-history.com/#gilito21/TFGDBA&Date)

</div>

---

<div align="center">
</div>
