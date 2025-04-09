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
    
    image = cv2.resize(image, (128, 128))
    
    # Ensure image is in RGB format
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
    """Endpoint that provides model metadata"""
    global model
    
    # Ensure model is loaded
    if model is None:
        load_saved_model()
    
    # Collect model metadata
    metadata = {
        "model_name": "Hurricane Damage Assessment - Alternate LeNet-5",
        "model_version": "1.0",
        "input_shape": list(model.input_shape[1:]),  
        "output_shape": list(model.output_shape[1:]),
        "framework": "TensorFlow " + tf.__version__,
        "description": "Binary classification model predicting whether a building has hurricane damage",
        "architecture": "Alternate LeNet-5 with 4 Conv layers, MaxPooling, Dropout and Dense layers",
        "test_accuracy": "97.5%"
    }
    
    return jsonify(metadata)

@app.route('/inference', methods=['POST'])
def inference():
    """Endpoint to perform model inference"""
    global model
    
    # Ensure model is loaded
    if model is None:
        load_saved_model()
    
    # Check
    if 'file' not in request.files and request.data:
       
        np_arr = np.frombuffer(request.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif 'file' in request.files:
  
        file = request.files['file']
        np_arr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        return jsonify({"error": "No image provided"}), 400
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Perform prediction
    prediction = model.predict(processed_image)
    
    # For binary classification with 2 output neurons (softmax)
    if prediction.shape[1] == 2:
        class_idx = np.argmax(prediction[0])
        
        # Assuming index 0 is "damage" and index 1 is "no_damage"
        prediction_label = "damage" if class_idx == 0 else "no_damage"
    
    else:
        prediction_label = "damage" if prediction[0][0] > 0.5 else "no_damage"
    
    # Return prediction result in the required format
    return jsonify({"prediction": prediction_label})

def load_saved_model():
    """Load the saved model"""
    global model
    
    model_path = os.getenv("MODEL_PATH", "hurricane_damage_model.h5")
    print(f"Loading model from {model_path}")
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == '__main__':
  
    load_saved_model()
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)