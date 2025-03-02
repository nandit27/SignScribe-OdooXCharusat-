import 'webrtc-adapter';
import { useState, useRef, useEffect } from 'react';
import { ArrowLeft, Camera, RefreshCw, Wifi, WifiOff, Bug, Upload } from 'lucide-react';
import Webcam from 'react-webcam';
import Button from '../ui/Button';
import { Link } from 'react-router-dom';
import TensorFlowService from '../../services/TensorFlowService';
import io from 'socket.io-client';

const TestFeature = () => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [realtimeTranslation, setRealtimeTranslation] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [serverConnected, setServerConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const [lastPredictionTime, setLastPredictionTime] = useState(null);
  const [debugInfo, setDebugInfo] = useState('Translation stopped');
  const [uploadResult, setUploadResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [captureStatus, setCaptureStatus] = useState('idle'); // 'idle', 'capturing', 'processing', 'waiting'
  const [captureProgress, setCaptureProgress] = useState(0);
  const [waitCountdown, setWaitCountdown] = useState(0);
  
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const frameCountRef = useRef(0);
  const animationRef = useRef(null);
  const socketRef = useRef(null);
  const fileInputRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const framesBuffer = useRef([]);

  // New canvas-specific refs
  const captureCanvasRef = useRef(null);
  const captureCtxRef = useRef(null);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  const [stream, setStream] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [captureInterval, setCaptureInterval] = useState(null);
  
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  
  // WebRTC configuration
  const constraints = {
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30 }
    }
  };

  // Initialize TensorFlow service connection
  useEffect(() => {
    console.log("TestFeature component mounted");
    
    // Initialize connection when component mounts
    const success = TensorFlowService.connect();
    setServerConnected(success);
    
    // Set up event handlers
    TensorFlowService.onConnect(() => {
      console.log("Connection established in component");
      setServerConnected(true);
      setConnectionError('');
      setDebugInfo('Connected to server');
    });

    TensorFlowService.onDisconnect((reason) => {
      console.log("Disconnection detected in component:", reason);
      setServerConnected(false);
      setConnectionError(`Disconnected: ${reason}`);
      setDebugInfo(`Disconnected: ${reason}`);
      
      // If we were translating, stop it
      if (isTranslating) {
        stopTranslation();
      }
    });

    TensorFlowService.onPrediction((data) => {
      console.log('Received prediction in component:', data);
      if (data.sign) {
        setRealtimeTranslation(data.sign);
        setConfidence(data.confidence || 0);
        setLastPredictionTime(new Date());
        setDebugInfo(`Prediction received: ${data.sign} (${Math.round(data.confidence * 100)}%)`);
      }
    });

    TensorFlowService.onError((error) => {
      console.error('TensorFlow error:', error);
      setConnectionError(error.message || 'Error processing sign');
      setDebugInfo(`Error: ${error.message || 'Unknown error'}`);
    });

    // Cleanup on unmount
    return () => {
      console.log("TestFeature component unmounting");
      stopTranslation();
      TensorFlowService.disconnect();
    };
  }, []);

  // Connect to WebSocket server
  useEffect(() => {
    socketRef.current = io('http://localhost:5001');
    
    socketRef.current.on('connect', () => {
      console.log('Connected to server');
      setServerConnected(true);
      
      // Test connection
      socketRef.current.emit('test_connection', { message: 'Hello from client' });
    });
    
    socketRef.current.on('test_response', (data) => {
      console.log('Test response:', data);
    });
    
    socketRef.current.on('prediction', (data) => {
      console.log('Received prediction:', data);
      setPrediction(data);
    });
    
    socketRef.current.on('error', (data) => {
      console.error('Server error:', data.message);
    });
    
    socketRef.current.on('disconnect', () => {
      console.log('Disconnected from server');
      setServerConnected(false);
    });
    
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // Initialize canvas context
  useEffect(() => {
    if (captureCanvasRef.current) {
      captureCtxRef.current = captureCanvasRef.current.getContext('2d');
    }
  }, []);

  // Initialize WebRTC stream
  useEffect(() => {
    const initializeWebRTC = async () => {
      try {
        setDebugInfo('Initializing WebRTC...');
        const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          streamRef.current = mediaStream;
          setStream(mediaStream);
          setIsInitialized(true);
          setDebugInfo('WebRTC initialized successfully');
        }
      } catch (error) {
        console.error('Error initializing WebRTC:', error);
        setDebugInfo(`WebRTC initialization failed: ${error.message}`);
      }
    };

    initializeWebRTC();

    // Cleanup
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Frame capture and processing using WebRTC
  const captureFrame = () => {
    if (!videoRef.current || !captureCanvasRef.current) return null;

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get frame data
    return canvas.toDataURL('image/jpeg', 0.95);
  };

  // Frame processing function
  const processFrame = (frameData) => {
    if (!frameData) return;

    const formData = new FormData();
    
    // Convert base64 to blob
    const byteString = atob(frameData.split(',')[1]);
    const mimeString = frameData.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    
    const blob = new Blob([ab], { type: mimeString });
    formData.append('file', blob, 'frame.jpg');

    return formData;
  };

  // Start frame capture
  const startTranslation = () => {
    if (!isInitialized) {
      setDebugInfo('WebRTC not initialized');
      return;
    }

    setIsTranslating(true);
    setDebugInfo('Starting frame capture...');

    const interval = setInterval(() => {
      const frameData = captureFrame();
      if (frameData) {
        const formData = processFrame(frameData);
        if (formData) {
          // Update the fetch URL to use the proxy
          fetch('/predict', {  
          method: 'POST',
          body: formData,
          credentials: 'include',
          headers: {
            'Accept': 'application/json'
          }
        })
          .then(async response => {
            if (!response.ok) {
              const errorText = await response.text();
              throw new Error(`Server error: ${errorText}`);
            }
            return response.json();
          })
          .then(result => {
            if (result.error) {
              throw new Error(result.error);
            }
            setRealtimeTranslation(result.prediction);
            setConfidence(result.confidence);
            setLastPredictionTime(new Date());
          })
          .catch(error => {
            console.error('Error processing frame:', error);
            setDebugInfo(`Error: ${error.message}`);
            if (error.message.includes('CORS')) {
              stopTranslation();
            }
          });
        }
      }
    }, 100); // Capture every 100ms

    setCaptureInterval(interval);
  };

  // Stop frame capture
  const stopTranslation = () => {
    if (captureInterval) {
      clearInterval(captureInterval);
      setCaptureInterval(null);
    }
    setIsTranslating(false);
    setDebugInfo('Translation stopped');
    setRealtimeTranslation('');
    setConfidence(0);
  };

  // Manually reconnect to server
  const reconnectToServer = () => {
    setConnectionError('Reconnecting...');
    setDebugInfo('Attempting to reconnect...');
    TensorFlowService.disconnect();
    setTimeout(() => {
      const success = TensorFlowService.connect();
      setServerConnected(success);
      if (!success) {
        setConnectionError('Failed to reconnect. Please try again.');
        setDebugInfo('Reconnection failed');
      }
    }, 1000);
  };

  // Send test frames to debug
  const handleTestFrame = () => {
    const frame = captureFrame();
    if (frame) {
      const formData = processFrame(frame);
      if (formData) {
        setDebugInfo('Sending test frame...');
        fetch('http://localhost:5001/predict', {
          method: 'POST',
          body: formData,
          credentials: 'include',
          headers: {
            'Accept': 'application/json',
            'Origin': 'http://localhost:5173'
          }
        })
        .then(async response => {
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${errorText}`);
          }
          return response.json();
        })
        .then(result => {
          setDebugInfo(`Test result: ${result.prediction} (${Math.round(result.confidence * 100)}%)`);
        })
        .catch(error => {
          setDebugInfo(`Test error: ${error.message}`);
        });
      }
    }
  };

  // Check if prediction is stale (no update in 5 seconds)
  const isPredictionStale = () => {
    if (!lastPredictionTime) return false;
    const now = new Date();
    return (now - lastPredictionTime) > 5000; // 5 seconds
  };

  // Handle file upload for single image prediction
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadResult(null);
    setDebugInfo('Uploading image for prediction...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        body: formData,
        mode: 'cors',
        headers: {
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const result = await response.json();
      console.log('Prediction result:', result);
      
      setUploadResult({
        prediction: result.prediction,
        confidence: result.confidence
      });
      
      setDebugInfo(`Image prediction: ${result.prediction} (${Math.round(result.confidence * 100)}%)`);
    } catch (error) {
      console.error('Error uploading image:', error);
      setDebugInfo(`Error uploading image: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Trigger file input click
  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  // Capture and send a single frame for testing
  const captureSingleFrame = () => {
    if (!webcamRef.current || !captureCanvasRef.current) return;
    
    const imageData = captureFrame();
    if (imageData) {
      setDebugInfo('Captured single frame, sending to server...');
      
      try {
        // Convert base64 to blob
        const byteString = atob(imageData.split(',')[1]);
        const mimeString = imageData.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        
        for (let i = 0; i < byteString.length; i++) {
          ia[i] = byteString.charCodeAt(i);
        }
        
        const blob = new Blob([ab], { type: mimeString });
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        
        // Send the form data to the server with explicit CORS settings
        fetch('http://localhost:5001/predict', {
          method: 'POST',
          body: formData,
          mode: 'cors',
          headers: {
            'Accept': 'application/json',
          }
        })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
          }
          return response.json();
        })
        .then(result => {
          console.log('Single frame prediction:', result);
          setUploadResult({
            prediction: result.prediction,
            confidence: result.confidence
          });
          setDebugInfo(`Single frame prediction: ${result.prediction} (${Math.round(result.confidence * 100)}%)`);
        })
        .catch(error => {
          console.error('Error sending single frame:', error);
          setDebugInfo(`Error: ${error.message}`);
        });
      } catch (error) {
        console.error('Error processing image:', error);
        setDebugInfo(`Error processing image: ${error.message}`);
      }
    }
  };

  // Alternative capture single frame method
  const captureSingleFrameAlternative = () => {
    if (!webcamRef.current || !captureCanvasRef.current) return;
    
    try {
      setDebugInfo('Capturing frame...');
      
      // Get the canvas and draw the current video frame
      const canvas = captureCanvasRef.current;
      const ctx = canvas.getContext('2d');
      const video = webcamRef.current.video;
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to blob
      canvas.toBlob((blob) => {
        if (!blob) {
          setDebugInfo('Failed to create blob from canvas');
          return;
        }
        
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        
        setDebugInfo('Sending frame to server...');
        
        // Send request with explicit CORS headers
        fetch('http://localhost:5001/predict', {
          method: 'POST',
          body: formData,
          mode: 'cors',
          headers: {
            'Accept': 'application/json'
          }
        })
        .then(async response => {
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${errorText}`);
          }
          return response.json();
        })
        .then(result => {
          console.log('Prediction result:', result);
          if (result.error) {
            throw new Error(result.error);
          }
          setUploadResult({
            prediction: result.prediction,
            confidence: result.confidence
          });
          setDebugInfo(`Prediction: ${result.prediction} (${Math.round(result.confidence * 100)}%)`);
        })
        .catch(error => {
          console.error('Error:', error);
          setDebugInfo(`Error: ${error.message}`);
        });
      }, 'image/jpeg', 0.95);  // Increased quality to 0.95
      
    } catch (error) {
      console.error('Error capturing frame:', error);
      setDebugInfo(`Error capturing frame: ${error.message}`);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (captureInterval) {
        clearInterval(captureInterval);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 pt-24">
      <div className="container mx-auto px-4 py-12">
        <Link to="/" className="inline-flex items-center text-teal hover:underline mb-8">
          <ArrowLeft size={20} className="mr-2" />
          Back to Home
        </Link>
        
        <div className="max-w-3xl mx-auto">
          <h1 className="text-3xl font-bold mb-6 text-center">
            Test SynaEra Translation
          </h1>
          <p className="text-gray-600 mb-6 text-center">
            Experience how SynaEra translates sign language in real-time. Allow camera access and show a sign to see it translated.
          </p>
          
          {/* Server connection status */}
          <div className={`mb-6 p-3 rounded-lg text-center ${serverConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            <div className="flex items-center justify-center gap-2">
              {serverConnected ? (
                <>
                  <Wifi size={20} />
                  <span>Connected to translation service</span>
                </>
              ) : (
                <>
                  <WifiOff size={20} />
                  <span>Not connected to translation service</span>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={reconnectToServer}
                    className="ml-2"
                  >
                    <RefreshCw size={16} className="mr-1" />
                    Reconnect
                  </Button>
                </>
              )}
            </div>
            {connectionError && (
              <p className="text-red-600 text-sm mt-1">{connectionError}</p>
            )}
          </div>
          
          {/* Debug info */}
          <div className="mb-6 p-3 bg-gray-100 rounded-lg text-sm font-mono">
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold">Debug Info:</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleTestFrame}
                disabled={!serverConnected}
              >
                <Bug size={16} className="mr-1" />
                Send Test Frame
              </Button>
            </div>
            <p>{debugInfo || 'No debug info available'}</p>
          </div>
          
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <div className="p-6">
              <div className="flex flex-col items-center">
                {/* Video display area */}
                <div className="bg-gray-100 rounded-xl w-full aspect-video mb-6 flex items-center justify-center overflow-hidden relative">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  <canvas 
                    ref={captureCanvasRef}
                    className="hidden"
                  />
                  
                  {isTranslating && realtimeTranslation && (
                    <div className={`absolute bottom-4 left-0 right-0 bg-black/70 text-white p-3 mx-4 rounded-lg text-center ${isPredictionStale() ? 'border-2 border-yellow-400' : ''}`}>
                      <p className="font-medium text-xl">{realtimeTranslation}</p>
                      {confidence > 0 && (
                        <div className="mt-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                          <div 
                            className="bg-green-400 h-full" 
                            style={{ width: `${confidence * 100}%` }}
                          ></div>
                        </div>
                      )}
                      {isPredictionStale() && (
                        <p className="text-yellow-400 text-xs mt-1">No recent updates</p>
                      )}
                    </div>
                  )}
                </div>
                
                {/* Control buttons */}
                <div className="flex gap-4 mt-4 justify-center">
                  <Button
                    variant={isTranslating ? "secondary" : "primary"}
                    onClick={isTranslating ? stopTranslation : startTranslation}
                    disabled={!isInitialized}
                    className="w-full md:w-auto"
                  >
                    <span>{isTranslating ? 'Stop Translation' : 'Start Translation'}</span>
                  </Button>
                  
                  {/* File upload button */}
                  <Button
                    variant="outline"
                    onClick={triggerFileUpload}
                    className="w-full md:w-auto"
                    disabled={isUploading}
                  >
                    <Upload size={16} className="mr-1" />
                    <span>{isUploading ? 'Uploading...' : 'Upload Image'}</span>
                  </Button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept="image/*"
                    className="hidden"
                  />
                  
                  {/* Single frame capture button */}
                    <Button 
                    variant="outline"
                    onClick={captureSingleFrameAlternative}
                      className="w-full md:w-auto"
                      disabled={!serverConnected}
                    >
                    <Camera size={16} className="mr-1" />
                    <span>Capture Single Frame</span>
                    </Button>
                </div>
                
                {/* Upload result display */}
                {uploadResult && (
                  <div className="mt-4 p-4 bg-gray-100 rounded-lg w-full">
                    <h3 className="font-semibold text-lg">Prediction Result:</h3>
                    <p className="text-xl mt-2">{uploadResult.prediction}</p>
                    <div className="mt-2 bg-gray-200 rounded-full h-2 overflow-hidden">
                      <div 
                        className="bg-green-400 h-full" 
                        style={{ width: `${uploadResult.confidence * 100}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Confidence: {Math.round(uploadResult.confidence * 100)}%
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="mt-8 bg-gray-50 rounded-lg p-6 border border-gray-200">
            <h3 className="font-semibold mb-2">How it works:</h3>
            <ol className="list-decimal pl-5 space-y-2">
              <li>The app connects to our sign language translation service</li>
              <li>Click "Start Translation" to begin real-time sign language detection</li>
              <li>Show sign language gestures to the camera</li>
              <li>Our AI model processes each frame and returns the translation</li>
              <li>Translations appear in real-time at the bottom of the video</li>
              <li>You can also upload an image file for single-frame prediction</li>
              <li>Or use "Capture Single Frame" for immediate analysis of the current view</li>
              <li>If you encounter issues, try the "Send Test Frame" debug button</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestFeature;
