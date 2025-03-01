import tensorflow as tf
import os

def convert_model():
    # Set paths
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    h5_path = os.path.join(model_dir, 'TheLastDance.h5')
    tflite_path = os.path.join(model_dir, 'TheLastDance.tflite')
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load your .h5 model
    print("Loading model...")
    model = tf.keras.models.load_model(h5_path)
    
    # Convert to TensorFlow Lite
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Add configurations needed for LSTM model
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = True
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    print(f"Saving TFLite model to {tflite_path}")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print("Conversion complete!")

if __name__ == '__main__':
    convert_model()