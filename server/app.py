from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import base64
import cv2
import tensorflow as tf
import os
import mediapipe as mp

# Initialize Flask app with comprehensive CORS settings
app = Flask(__name__)

# Update CORS configuration to be more specific
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],  # Specify exact origin
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})

# Configure SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    enable_segmentation=True,  # Enable segmentation if needed
    refine_face_landmarks=True  # Get detailed face landmarks
)

# Load TensorFlow Lite model
interpreter = None

print("Checking MediaPipe installation...")
try:
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_results = holistic.process(test_image_rgb)
    print("MediaPipe test successful")
except Exception as e:
    print(f"MediaPipe test failed: {str(e)}")
    import traceback
    traceback.print_exc()

def load_model():
    global interpreter
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'TheLastDance.tflite')
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def mediapipe_detection(image, model):
    """Process image with MediaPipe and return results"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    
    return image_rgb, results

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results to match exactly 1662 features"""
    
    # Pose: 33 landmarks × 4 coordinates (x,y,z,visibility) = 132 features
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                    results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Face: 468 landmarks × 3 coordinates (x,y,z) = 1404 features
    face = np.array([[res.x, res.y, res.z] for res in 
                    results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Left hand: 21 landmarks × 3 coordinates (x,y,z) = 63 features
    lh = np.array([[res.x, res.y, res.z] for res in 
                  results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right hand: 21 landmarks × 3 coordinates (x,y,z) = 63 features
    rh = np.array([[res.x, res.y, res.z] for res in 
                  results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate all features: 132 + 1404 + 63 + 63 = 1662 features
    keypoints = np.concatenate([pose, face, lh, rh])
    
    # Verify the dimension
    if keypoints.shape[0] != 1662:
        print(f"Warning: Unexpected keypoints dimension. Got {keypoints.shape[0]}, expected 1662")
        # Pad or truncate to ensure correct dimension
        if keypoints.shape[0] < 1662:
            keypoints = np.pad(keypoints, (0, 1662 - keypoints.shape[0]))
        else:
            keypoints = keypoints[:1662]
    
    return keypoints

def predict_keypoints(keypoints):
    """Predict sign from keypoints using TensorFlow Lite model"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print expected shape for debugging
    expected_shape = input_details[0]['shape']
    print(f"Model expects input shape: {expected_shape}")
    print(f"Current keypoints shape: {keypoints.shape}")
    
    # Reshape for model input - add batch dimension and possibly time dimension
    if len(expected_shape) == 3:  # If model expects [batch, time, features]
        # For single frame prediction, we need to add a time dimension
        keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension
        keypoints = np.expand_dims(keypoints, axis=1)  # Add time dimension
    else:
        # Just add batch dimension
        keypoints = np.expand_dims(keypoints, axis=0)
    
    keypoints = keypoints.astype(np.float32)
    print(f"Reshaped keypoints shape: {keypoints.shape}")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], keypoints)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process results
    prediction_idx = np.argmax(output_data[0])
    confidence = float(output_data[0][prediction_idx])
    
    # Map index to sign language label
    labels = ['Alright', 'Hello', 'Indian', 'Namaste', 'Sign']
    predicted_sign = labels[prediction_idx] if prediction_idx < len(labels) else "Unknown"
    
    return predicted_sign, confidence

def process_image(image_data):
    """Process base64 image data and return prediction"""
    # Decode base64 image
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
    
    # Process with MediaPipe
    _, results = mediapipe_detection(img, holistic)
    
    # Extract keypoints
    keypoints = extract_keypoints(results)
    
    # Get prediction
    predicted_sign, confidence = predict_keypoints(keypoints)
    
    return {
        "sign": predicted_sign,
        "confidence": confidence
    }

def process_image_sequence(frames_data):
    """Process a sequence of frames and return prediction"""
    sequence = []
    
    for frame_data in frames_data:
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
        
        # Process with MediaPipe
        _, results = mediapipe_detection(frame, holistic)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
    
    # Convert sequence to numpy array and reshape for model input
    sequence = np.array(sequence)
    
    # Ensure we have exactly 30 frames (pad or truncate if necessary)
    if len(sequence) < 30:
        # Pad with zeros if we have fewer than 30 frames
        padding = np.zeros((30 - len(sequence), sequence.shape[1]))
        sequence = np.vstack([sequence, padding])
    elif len(sequence) > 30:
        # Take only the last 30 frames if we have more
        sequence = sequence[-30:]
    
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get predictions from model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], sequence.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process results
    prediction_idx = np.argmax(output_data[0])
    confidence = float(output_data[0][prediction_idx])
    
    # Map index to sign language label
    labels = ['Alright', 'Hello', 'Indian', 'Namaste', 'Sign']
    predicted_sign = labels[prediction_idx] if prediction_idx < len(labels) else "Unknown"
    
    return {
        "sign": predicted_sign,
        "confidence": confidence
    }

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('connect_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    try:
        if not interpreter:
            load_model()
            
        if not data.get('image'):
            raise ValueError("No image data received")
            
        result = process_image(data['image'])
        emit('prediction', result)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('frame_sequence')
def handle_frame_sequence(data):
    try:
        print("\n----- RECEIVED FRAME SEQUENCE -----")
        if not interpreter:
            print("Loading model...")
            load_model()
            
        if not data.get('frames'):
            print("ERROR: No frames received in data")
            raise ValueError("No frames received")
            
        frames_count = len(data['frames'])
        print(f"Processing sequence of {frames_count} frames")
        
        # Process the frames
        result = process_image_sequence(data['frames'])
        print(f"Prediction result: {result}")
        
        # Emit the prediction
        emit('prediction', result)
        print("----- FRAME SEQUENCE PROCESSED -----\n")
    except Exception as e:
        print(f"ERROR processing frame sequence: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})

@socketio.on('test_connection')
def handle_test_connection(data):
    print("\n----- TEST CONNECTION RECEIVED -----")
    print(f"Message: {data.get('message', 'No message')}")
    print("Sending test response back to client")
    emit('test_response', {'message': 'Hello from server'})
    print("----- TEST CONNECTION COMPLETE -----\n")
    return {'status': 'success', 'message': 'Test connection successful'}

# HTTP routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Update the predict endpoint to also include CORS headers
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return make_response('OK', 200)

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'Empty file'}), 400

    try:
        # Read image from request
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Process with Mediapipe
        _, results = mediapipe_detection(image, holistic)
        keypoints = extract_keypoints(results)

        # Debug print
        print(f"Extracted keypoints shape: {keypoints.shape}")  # Should be (1662,)

        # Create sequence
        sequence = np.array([keypoints] * 30)  # Should be shape (30, 1662)
        print(f"Sequence shape before expand_dims: {sequence.shape}")
        
        sequence = np.expand_dims(sequence, axis=0)  # Should be shape (1, 30, 1662)
        print(f"Final sequence shape: {sequence.shape}")

        # Verify dimensions match model expectations
        input_details = interpreter.get_input_details()
        expected_shape = input_details[0]['shape']
        print(f"Model expected shape: {expected_shape}")

        if sequence.shape != tuple(expected_shape):
            raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {sequence.shape}")

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sequence.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        prediction_idx = np.argmax(output_data[0])
        confidence = float(output_data[0][prediction_idx])
        
        # Map index to sign language label
        labels = ['Alright', 'Hello', 'Indian', 'Namaste', 'Sign']
        predicted_sign = labels[prediction_idx] if prediction_idx < len(labels) else "Unknown"

        response = jsonify({
            'prediction': predicted_sign,
            'confidence': confidence
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)