import { useState, useRef, useEffect } from 'react';
import { ArrowLeft, Camera, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import Button from '../ui/Button';
import { Link } from 'react-router-dom';
import TensorFlowService from '../../services/TensorFlowService';

const TestFeature = () => {
  const [cameraActive, setCameraActive] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [realtimeTranslation, setRealtimeTranslation] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [permissionError, setPermissionError] = useState(false);
  const [serverConnected, setServerConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);

  // Initialize TensorFlow service connection
  useEffect(() => {
    // Set up event handlers
    TensorFlowService.onConnect(() => {
      setServerConnected(true);
      setConnectionError('');
    });

    TensorFlowService.onDisconnect(() => {
      setServerConnected(false);
      if (isTranslating) {
        setIsTranslating(false);
        setRealtimeTranslation('Server disconnected');
      }
    });

    TensorFlowService.onPrediction((data) => {
      setRealtimeTranslation(data.sign);
      setConfidence(data.confidence);
    });

    TensorFlowService.onError((error) => {
      console.error('TensorFlow error:', error);
      setConnectionError(error.message || 'Error processing sign');
    });

    // Connect to the server
    try {
      TensorFlowService.connect();
    } catch (err) {
      console.error('Failed to connect to TensorFlow service:', err);
      setConnectionError('Failed to connect to translation service');
    }

    // Clean up on unmount
    return () => {
      TensorFlowService.disconnect();
      stopCamera();
    };
  }, []);

  // Function to start camera
  const startCamera = async () => {
    try {
      setPermissionError(false);
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support camera access');
      }

      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });
      
      // Store stream reference for cleanup
      streamRef.current = stream;
      
      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready before setting active state
        videoRef.current.onloadedmetadata = () => {
          setCameraActive(true);
          console.log("Camera is now active");
        };
      } else {
        throw new Error("Video element not available");
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setPermissionError(true);
      alert(`Camera error: ${err.message}`);
    }
  };

  // Function to stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    
    setCameraActive(false);
    setIsTranslating(false);
  };

  // Function to capture frames and send to backend
  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current || !isTranslating) return;
    
    const context = canvasRef.current.getContext('2d');
    const { videoWidth, videoHeight } = videoRef.current;
    
    // Set canvas dimensions to match video
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);
    
    // Convert canvas to base64 image
    const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8);
    
    // Send to backend via WebSocket
    if (serverConnected) {
      TensorFlowService.sendFrame(imageData);
    }
    
    // Continue capturing frames (at a reasonable rate to not overload the server)
    animationRef.current = setTimeout(() => {
      requestAnimationFrame(captureFrame);
    }, 100); // Adjust this value based on your server's processing capacity
  };

  // Start real-time translation
  const startTranslation = () => {
    if (!serverConnected) {
      alert('Cannot start translation: not connected to server');
      return;
    }
    
    setIsTranslating(true);
    setRealtimeTranslation('Detecting...');
    
    // Start capturing frames
    captureFrame();
  };

  // Stop real-time translation
  const stopTranslation = () => {
    setIsTranslating(false);
    setRealtimeTranslation('');
    
    if (animationRef.current) {
      clearTimeout(animationRef.current);
      animationRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 pt-24">
      <div className="container mx-auto px-4 py-12">
        <Link to="/" className="inline-flex items-center text-teal hover:underline mb-8">
          <ArrowLeft size={20} className="mr-2" />
          Back to Home
        </Link>
        
        <div className="max-w-3xl mx-auto">
          <h1 className="text-3xl font-bold mb-6 text-center">
            Test SignScribe Translation
          </h1>
          <p className="text-gray-600 mb-6 text-center">
            Experience how SignScribe translates sign language in real-time. Allow camera access and show a sign to see it translated.
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
                </>
              )}
            </div>
            {connectionError && (
              <p className="text-red-600 text-sm mt-1">{connectionError}</p>
            )}
          </div>
          
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <div className="p-6">
              <div className="flex flex-col items-center">
                {/* Video display area */}
                <div className="bg-gray-100 rounded-xl w-full aspect-video mb-6 flex items-center justify-center overflow-hidden relative">
                  {!cameraActive ? (
                    <div className="text-gray-400 flex flex-col items-center">
                      <Camera size={48} className="mb-2" />
                      <p>Camera preview will appear here</p>
                      {permissionError && (
                        <p className="text-red-500 text-sm mt-2">
                          Camera permission denied. Please check your browser settings.
                        </p>
                      )}
                    </div>
                  ) : (
                    <>
                      <video 
                        ref={videoRef} 
                        autoPlay 
                        playsInline 
                        muted 
                        className="w-full h-full object-cover"
                      />
                      {/* Hidden canvas for processing */}
                      <canvas 
                        ref={canvasRef} 
                        className="hidden"
                      />
                      {/* Real-time translation overlay */}
                      {isTranslating && realtimeTranslation && (
                        <div className="absolute bottom-4 left-0 right-0 bg-black/70 text-white p-3 mx-4 rounded-lg text-center">
                          <p className="font-medium text-xl">{realtimeTranslation}</p>
                          {confidence > 0 && (
                            <div className="mt-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                              <div 
                                className="bg-green-400 h-full" 
                                style={{ width: `${confidence * 100}%` }}
                              ></div>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
                
                {/* Control buttons */}
                <div className="flex gap-4 mt-4">
                  {!cameraActive ? (
                    <Button 
                      variant="primary" 
                      onClick={startCamera}
                      className="w-full md:w-auto"
                    >
                      <Camera size={20} />
                      <span>Access Camera</span>
                    </Button>
                  ) : (
                    <>
                      {!isTranslating ? (
                        <Button 
                          variant="primary" 
                          onClick={startTranslation}
                          className="w-full md:w-auto"
                          disabled={!serverConnected}
                        >
                          <span>Start Translation</span>
                        </Button>
                      ) : (
                        <Button 
                          variant="secondary" 
                          onClick={stopTranslation}
                          className="w-full md:w-auto"
                        >
                          <span>Pause Translation</span>
                        </Button>
                      )}
                      <Button 
                        variant="outline" 
                        onClick={stopCamera}
                        className="w-full md:w-auto"
                      >
                        <span>Stop Camera</span>
                      </Button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-8 bg-gray-50 rounded-lg p-6 border border-gray-200">
            <h3 className="font-semibold mb-2">How it works:</h3>
            <ol className="list-decimal pl-5 space-y-2">
              <li>The app connects to our sign language translation service</li>
              <li>Click "Access Camera" to enable your device's camera</li>
              <li>Click "Start Translation" to begin real-time sign language detection</li>
              <li>Show sign language gestures to the camera</li>
              <li>Our AI model processes each frame and returns the translation</li>
              <li>Translations appear in real-time at the bottom of the video</li>
            </ol>
            <p className="mt-4 text-sm text-gray-500">
              Note: This is a demonstration. The full app provides additional features and improved accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestFeature;