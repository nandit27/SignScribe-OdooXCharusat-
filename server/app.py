from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import base64
import cv2
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load TensorFlow Lite model
interpreter = None

def load_model():
    global interpreter
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'TheLastDance.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Model loaded successfully")

# Process image function
def process_image(image_data):
    # Decode base64 image
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
    
    # Preprocess image for model
    img = cv2.resize(img, (224, 224))  # Adjust size to match your model's input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process results (adjust based on your model's output format)
    prediction_idx = np.argmax(output_data[0])
    confidence = float(output_data[0][prediction_idx])
    
    # Map index to sign language label (replace with your labels)
    labels = ["Hello", "Thank you", "Yes", "No", "Please", "Sorry"]
    predicted_sign = labels[prediction_idx] if prediction_idx < len(labels) else "Unknown"
    
    return {
        "sign": predicted_sign,
        "confidence": confidence
    }

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    try:
        result = process_image(data['image'])
        emit('prediction', result)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

# HTTP routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    load_model()
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)