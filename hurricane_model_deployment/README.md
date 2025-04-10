# README: Hurricane Damage Assessment Model Deployment

## Project Overview
This project deploys a trained **Alternate LeNet-5 (Alt LeNet-5)** convolutional neural network (CNN) model that classifies aerial images of buildings as either **"damage"** or **"no_damage"**. The model is served using a Flask-based HTTP server and packaged into a Docker container for reproducible deployment.

---

## Files Included
- `Dockerfile`: Defines the container environment
- `docker-compose.yml`: Configuration to start the model server container
- `hurricane_damage_model.h5`: Trained model file (in native Keras HDF5 format)
- `app.py`: Python Flask application to serve the model
- `requirements.txt`: Python dependencies

---

## How to Build and Run the Server

### 1. Clone the Repository and Navigate to Project
```bash
cd ~/hurricane_model_deployment
```

### 2. Build Docker Image
```bash
docker-compose build
```

### 3. Run the Container
```bash
docker-compose up -d
```
The model server will be available at `http://127.0.0.1:5000` (or within Docker network at `http://172.17.0.1:5000` from host).

---

## Using the Prebuilt Docker Image
If you do not want to build the image yourself, you can pull the prebuilt image from Docker Hub:

### Pull the Image
```bash
docker pull ssspro/hurricane-damage-model:latest
```

### Run the Container
```bash
docker run -d -p 5000:5000 ssspro/hurricane-damage-model:latest
```

---

## Endpoints

### `GET /summary`
Returns model metadata in JSON format.
```bash
curl http://localhost:5000/summary
```
**Example Response:**
```json
{
  "model_name": "Hurricane Damage Assessment — Alternate LeNet-5",
  "model_version": "1.0",
  "input_shape": [128, 128, 3],
  "output_shape": [2],
  "framework": "TensorFlow 2.13.0",
  "description": "Binary classification model predicting whether a building has hurricane damage",
  "architecture": "Alternate LeNet-5 with 4 Conv layers, MaxPooling, Dropout and Dense layers",
  "test_accuracy": "97.2%",
  "format": "Keras HDF5 (.h5)"
}
```

### `POST /inference`
Accepts a binary image (JPEG/PNG) as multipart form-data and returns classification.
```bash
curl -X POST http://localhost:5000/inference \
  -F image=@path_to_image.jpg
```
**Example Response:**
```json
{
  "prediction": "damage"
}
```

---

