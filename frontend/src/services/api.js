import axios from 'axios';
import io from 'socket.io-client';

const API_BASE_URL = 'http://localhost:5000/api';
const socket = io('http://localhost:5000');

// API calls
export const api = {
  // Health check
  checkHealth: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },

  // Start training
  startTraining: async (params) => {
    const response = await axios.post(`${API_BASE_URL}/train`, params);
    return response.data;
  },

  // Stop training
  stopTraining: async () => {
    const response = await axios.post(`${API_BASE_URL}/train/stop`);
    return response.data;
  },

  // Generate images
  generateImages: async (numImages, checkpoint) => {
    const response = await axios.post(`${API_BASE_URL}/generate`, {
      numImages,
      checkpoint
    });
    return response.data;
  },

  // List checkpoints
  listCheckpoints: async () => {
    const response = await axios.get(`${API_BASE_URL}/checkpoints`);
    return response.data;
  }
};

// WebSocket events
export const setupWebSocket = (callbacks) => {
  socket.on('training_progress', callbacks.onProgress);
  socket.on('sample_images', callbacks.onSampleImages);
  socket.on('training_complete', callbacks.onComplete);
  
  return () => {
    socket.off('training_progress');
    socket.off('sample_images');
    socket.off('training_complete');
  };
};

export default api;