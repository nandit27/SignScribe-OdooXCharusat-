import io from 'socket.io-client';

class TensorFlowService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.onPredictionCallback = null;
    this.onErrorCallback = null;
    this.onConnectCallback = null;
    this.onDisconnectCallback = null;
    this.frameBuffer = [];
    this.FRAMES_NEEDED = 30;
    this.isProcessing = false;
    this.reconnectAttempts = 0;
    this.MAX_RECONNECT_ATTEMPTS = 5;
    this.pingInterval = null;
    this.debugMode = true;
  }

  log(message) {
    if (this.debugMode) {
      console.log(`[TensorFlowService] ${message}`);
    }
  }

  error(message) {
    console.error(`[TensorFlowService] ${message}`);
  }

  connect() {
    try {
      // Close any existing connection first
      if (this.socket) {
        this.log('Closing existing socket connection');
        this.socket.close();
        this.socket = null;
      }

      // Reset state
      this.isConnected = false;
      this.frameBuffer = [];
      this.isProcessing = false;

      this.log('Creating new socket connection to http://localhost:5001');
      
      // Configure Socket.IO with explicit transport preference
      this.socket = io('http://localhost:5001', {
        transports: ['websocket'],
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        timeout: 20000,
        forceNew: true
      });
      
      this.socket.on('connect', () => {
        this.log('Connected to TensorFlow service');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        if (this.onConnectCallback) this.onConnectCallback();
      });

      this.socket.on('prediction', (data) => {
        this.log(`Received prediction: ${JSON.stringify(data)}`);
        this.isProcessing = false;
        
        if (this.onPredictionCallback) this.onPredictionCallback(data);
      });

      this.socket.on('error', (error) => {
        this.error(`Server error: ${JSON.stringify(error)}`);
        if (this.onErrorCallback) this.onErrorCallback(error);
      });

      this.socket.on('disconnect', (reason) => {
        this.error(`Disconnected: ${reason}`);
        this.isConnected = false;
        this.isProcessing = false;
        
        if (this.onDisconnectCallback) this.onDisconnectCallback(reason);
        
        // Attempt to reconnect
        this.attemptReconnect();
      });

      return true;
    } catch (error) {
      this.error(`Error connecting to TensorFlow service: ${error.message}`);
      return false;
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS) {
      this.reconnectAttempts++;
      this.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.MAX_RECONNECT_ATTEMPTS})...`);
      
      setTimeout(() => {
        this.connect();
      }, 2000);
    } else {
      this.error('Max reconnection attempts reached. Please check server status.');
    }
  }

  disconnect() {
    if (this.socket) {
      this.frameBuffer = [];
      this.isProcessing = false;
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.log('Disconnected from TensorFlow service');
    }
  }

  // Add frame to buffer
  addFrame(imageData) {
    if (!this.isConnected) {
      this.error('Cannot add frame: not connected');
      return false;
    }
    
    this.frameBuffer.push(imageData);
    return true;
  }

  // Process and send frames in batches
  processFrames() {
    if (!this.isConnected || this.isProcessing) {
      return false;
    }
    
    if (this.frameBuffer.length >= this.FRAMES_NEEDED) {
      this.log(`Sending batch of ${this.FRAMES_NEEDED} frames`);
      this.isProcessing = true;
      
      // Take exactly 30 frames
      const framesToSend = this.frameBuffer.slice(0, this.FRAMES_NEEDED);
      this.frameBuffer = this.frameBuffer.slice(this.FRAMES_NEEDED);
      
      try {
        this.socket.emit('frame_sequence', { frames: framesToSend });
        this.log('Frame sequence sent successfully');
        return true;
      } catch (error) {
        this.error(`Error sending frames: ${error.message}`);
        this.isProcessing = false;
        return false;
      }
    }
    
    return false;
  }

  // Send test frames
  sendTestFrames() {
    if (!this.isConnected) {
      this.error('Cannot send test frames: not connected');
      return false;
    }
    
    this.log('Sending test frame sequence');
    
    // Create a dummy frame
    const dummyFrame = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACv/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVIP/2Q==';
    
    // Create an array of 30 dummy frames
    const dummyFrames = Array(30).fill(dummyFrame);
    
    try {
      this.isProcessing = true;
      this.socket.emit('frame_sequence', { frames: dummyFrames });
      this.log('Test frames sent');
      return true;
    } catch (error) {
      this.error(`Error sending test frames: ${error.message}`);
      this.isProcessing = false;
      return false;
    }
  }

  onPrediction(callback) {
    this.onPredictionCallback = callback;
  }

  onError(callback) {
    this.onErrorCallback = callback;
  }

  onConnect(callback) {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback) {
    this.onDisconnectCallback = callback;
  }

  addFrames(frames) {
    if (!this.isConnected) {
      this.error('Cannot add frames: not connected');
      return false;
    }
    
    // Add all frames to buffer
    this.frameBuffer = [...this.frameBuffer, ...frames];
    this.log(`Added ${frames.length} frames to buffer. Total: ${this.frameBuffer.length}`);
    return true;
  }
}

// Create a singleton instance
const tensorflowService = new TensorFlowService();
export default tensorflowService;

