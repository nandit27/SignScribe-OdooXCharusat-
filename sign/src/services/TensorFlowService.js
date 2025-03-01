import io from 'socket.io-client';

class TensorFlowService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.onPredictionCallback = null;
    this.onErrorCallback = null;
    this.onConnectCallback = null;
    this.onDisconnectCallback = null;
  }

  connect(serverUrl = 'http://localhost:5001') {
    if (this.socket) {
      this.socket.disconnect();
    }

    this.socket = io(serverUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    this.socket.on('connect', () => {
      console.log('Connected to TensorFlow service');
      this.isConnected = true;
      if (this.onConnectCallback) this.onConnectCallback();
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from TensorFlow service');
      this.isConnected = false;
      if (this.onDisconnectCallback) this.onDisconnectCallback();
    });

    this.socket.on('prediction', (data) => {
      if (this.onPredictionCallback) this.onPredictionCallback(data);
    });

    this.socket.on('error', (error) => {
      console.error('TensorFlow service error:', error);
      if (this.onErrorCallback) this.onErrorCallback(error);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  sendFrame(imageData) {
    if (!this.socket || !this.isConnected) {
      console.warn('Cannot send frame: not connected to TensorFlow service');
      return false;
    }

    this.socket.emit('frame', { image: imageData });
    return true;
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
}

export default new TensorFlowService();
