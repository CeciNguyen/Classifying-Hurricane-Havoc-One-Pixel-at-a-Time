import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = None

# Image preprocessing function
def preprocess_image(image):
    # Resize image to match the model input size (128x128)
    image = cv2.resize(image, (128, 128))
    
    # Ensure image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    image = image.astype("float32") / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/summary', methods=['GET'])
def get_summary():
    global model
    
    # Ensure model is loaded
    if model is None:
        load_saved_model()
    
    # Collect model metadata
    metadata = {
        "model_name": "Hurricane Damage Assessment - Alternate LeNet-5",
        "model_version": "1.0",
        "input_shape": list(model.input_shape[1:]),  # Convert tuple to list for JSON serialization
        "output_shape": list(model.output_shape[1:]),
        "framework": "TensorFlow " + tf.__version__,
        "description": "Binary classification model predicting whether a building has hurricane damage",
        "architecture": "Alternate LeNet-5 with 4 Conv layers, MaxPooling, Dropout and Dense layers",
        "test_accuracy": "97.2%",
        "format": "Native Keras (.keras)"
    }
    
    return jsonify(metadata)

@app.route('/inference', methods=['POST'])
def inference():
    global model
    
    # Ensure model is loaded
    if model is None:
        load_saved_model()
    
    # Check if image data is received
    if 'file' not in request.files and request.data:
        # Process binary data instead of form file
        np_arr = np.frombuffer(request.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif 'file' in request.files:
        # Process form file
        file = request.files['file']
        np_arr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        return jsonify({"error": "No image provided"}), 400
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    
    if prediction.shape[1] == 2:
        class_idx = np.argmax(prediction[0])
        prediction_label = "damage" if class_idx == 0 else "no_damage"
    else:
        prediction_label = "damage" if prediction[0][0] > 0.5 else "no_damage"
    
    # Return prediction result
    return jsonify({"prediction": prediction_label})

def load_saved_model():
    global model
    
    model_path = os.getenv("MODEL_PATH", "hurricane_damage_model.keras")
    print(f"Loading model from {model_path}")
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == '__main__':
    load_saved_model()
    
    # Start Flask server
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
