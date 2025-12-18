import React, { useState, useEffect } from 'react';
import {io} from 'socket.io-client'; 
import { Play, Square, Download, RefreshCw, Zap, Brain, Eye, Settings, Info } from 'lucide-react';

const MNISTGANInterface = () => {
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(50);
  const [generatorLoss, setGeneratorLoss] = useState(0);
  const [discriminatorLoss, setDiscriminatorLoss] = useState(0);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [selectedModel, setSelectedModel] = useState('epoch_50');
  const [numImagesToGenerate, setNumImagesToGenerate] = useState(64);
  const [showSettings, setShowSettings] = useState(false);
  const [socket, setSocket] = useState(null);
  const [learningRate, setLearningRate] = useState(0.0002);
  const [batchSize, setBatchSize] = useState(64);
  const [latentDim, setLatentDim] = useState(64);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  useEffect(() => {
    // Note: Socket.IO requires installation: npm install socket.io-client
    // For demo purposes, this will be commented out
    
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('✅ WebSocket connected!');
      setConnectionStatus('connected');
    });

    newSocket.on('disconnect', () => {
      console.log('❌ WebSocket disconnected!');
      setConnectionStatus('disconnected');
    });

    newSocket.on('training_update', (data) => {
      setCurrentEpoch(data.current_epoch);
      setGeneratorLoss(data.generator_loss);
      setDiscriminatorLoss(data.discriminator_loss);
    });

    newSocket.on('sample_images', (data) => {
      console.log('Received sample images for epoch', data.epoch);
      setGeneratedImages(prev => [...prev, { epoch: data.epoch, images: data.images }]);
    });

    newSocket.on('error', (error) => {
      console.error('Socket error:', error);
    });

    return () => {
      console.log('Cleaning up socket connection');
      newSocket.close();
    };
    
    
    // Demo mode: simulate connection
    //console.log('Running in demo mode - WebSocket disabled');
    setConnectionStatus('demo');
  }, []);

  const startTraining = async () => {
    setTrainingStatus('training');
    setCurrentEpoch(0);
    setGeneratedImages([]);
    
    // API call (commented for demo)
    
    try {
      const response = await fetch('http://localhost:5000/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          learning_rate: learningRate,
          batch_size: batchSize,
          latent_dim: latentDim,
          total_epochs: totalEpochs
        })
      });
      
      if (!response.ok) {
        throw new Error('Training request failed');
      }
      
      const data = await response.json();
      console.log('Training started:', data);
    } catch (error) {
      console.error('Error starting training:', error);
      setTrainingStatus('idle');
      alert('Failed to start training. Make sure the backend server is running.');
      return;
    }
    
    /*
    // Simulate training for demo
    console.log('Starting training with params:', {
      learningRate,
      batchSize,
      latentDim,
      totalEpochs 
    });*/
    
    simulateTraining();
  };

  const simulateTraining = async () => {
    for (let i = 1; i <= totalEpochs; i++) {
      if (trainingStatus === 'paused') break;
      
      await new Promise(resolve => setTimeout(resolve, 100));
      setCurrentEpoch(i);
      
      const dLoss = 0.6 + Math.random() * 0.2;
      const gLoss = 0.8 + Math.random() * 0.4;
      setDiscriminatorLoss(dLoss);
      setGeneratorLoss(gLoss);
      
      if (i % 5 === 0) {
        const samples = generateSampleImages(8);
        setGeneratedImages(prev => [...prev, { epoch: i, images: samples }]);
      }
    }
    
    setTrainingStatus('completed');
  };

  const pauseTraining = () => {
    setTrainingStatus('paused');
  };

  const generateImages = () => {
    const gridSize = Math.sqrt(numImagesToGenerate);
    const images = generateSampleImages(gridSize);
    setGeneratedImages([{ epoch: 'generated', images }]);
  };

  const generateSampleImages = (gridSize) => {
    const images = [];
    for (let i = 0; i < gridSize * gridSize; i++) {
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      const ctx = canvas.getContext('2d');
      
      const gradient = ctx.createRadialGradient(14, 14, 2, 14, 14, 14);
      gradient.addColorStop(0, `hsl(${Math.random() * 60}, 20%, ${30 + Math.random() * 40}%)`);
      gradient.addColorStop(1, `hsl(${Math.random() * 60}, 10%, ${10 + Math.random() * 20}%)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 28, 28);
      
      images.push(canvas.toDataURL());
    }
    return images;
  };

  const downloadResults = () => {
    console.log('Downloading results...');
    alert('In production, this would download all generated images and model checkpoints');
  };

  const progress = (currentEpoch / totalEpochs) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-12 h-12 text-purple-400" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              MNIST GAN Studio
            </h1>
          </div>
          <p className="text-gray-400 text-lg">
            Train and generate handwritten digits using Generative Adversarial Networks
          </p>
          
          {/* Connection Status */}
          <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700">
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'disconnected' ? 'bg-red-500' :
              'bg-yellow-500'
            }`} />
            <span className="text-xs text-gray-400">
              {connectionStatus === 'connected' ? 'Backend Connected' :
               connectionStatus === 'disconnected' ? 'Backend Disconnected' :
               'Demo Mode'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="w-5 h-5 text-yellow-400" />
                <h2 className="text-xl font-bold">Training Control</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Epochs</label>
                  <input
                    type="number"
                    value={totalEpochs}
                    onChange={(e) => setTotalEpochs(parseInt(e.target.value) || 50)}
                    disabled={trainingStatus === 'training'}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg transition-colors"
                >
                  <Settings className="w-4 h-4" />
                  {showSettings ? 'Hide' : 'Show'} Advanced Settings
                </button>

                {showSettings && (
                  <div className="space-y-3 pt-2 border-t border-slate-700">
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        step="0.0001"
                        value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0002)}
                        disabled={trainingStatus === 'training'}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                      <input
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(parseInt(e.target.value) || 64)}
                        disabled={trainingStatus === 'training'}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Latent Dimension</label>
                      <input
                        type="number"
                        value={latentDim}
                        onChange={(e) => setLatentDim(parseInt(e.target.value) || 64)}
                        disabled={trainingStatus === 'training'}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                  </div>
                )}

                <div className="flex gap-2">
                  {trainingStatus !== 'training' ? (
                    <button
                      onClick={startTraining}
                      className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 px-6 py-3 rounded-lg font-semibold transition-all shadow-lg hover:shadow-purple-500/50"
                    >
                      <Play className="w-5 h-5" />
                      Start Training
                    </button>
                  ) : (
                    <button
                      onClick={pauseTraining}
                      className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-semibold transition-colors"
                    >
                      <Square className="w-5 h-5" />
                      Pause
                    </button>
                  )}
                </div>

                {trainingStatus === 'training' && (
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Epoch {currentEpoch}/{totalEpochs}</span>
                      <span>{progress.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <Eye className="w-5 h-5 text-blue-400" />
                <h2 className="text-xl font-bold">Loss Metrics</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-400">Generator Loss</span>
                    <span className="font-mono text-green-400">{generatorLoss.toFixed(4)}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-green-500 h-full rounded-full transition-all"
                      style={{ width: `${Math.min(generatorLoss * 50, 100)}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-400">Discriminator Loss</span>
                    <span className="font-mono text-blue-400">{discriminatorLoss.toFixed(4)}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-full rounded-full transition-all"
                      style={{ width: `${Math.min(discriminatorLoss * 100, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-700">
                  <div className="flex items-start gap-2 text-xs text-gray-400">
                    <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <p>
                      Healthy training: D-loss around 0.5-0.7, G-loss around 0.7-1.5
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 shadow-xl">
              <div className="flex items-center gap-2 mb-4">
                <RefreshCw className="w-5 h-5 text-purple-400" />
                <h2 className="text-xl font-bold">Generate Images</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Model Checkpoint</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="epoch_10">Epoch 10</option>
                    <option value="epoch_25">Epoch 25</option>
                    <option value="epoch_50">Epoch 50 (Best)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">Number of Images</label>
                  <select
                    value={numImagesToGenerate}
                    onChange={(e) => setNumImagesToGenerate(parseInt(e.target.value))}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="16">16 (4×4)</option>
                    <option value="64">64 (8×8)</option>
                    <option value="100">100 (10×10)</option>
                  </select>
                </div>

                <button
                  onClick={generateImages}
                  className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 px-6 py-3 rounded-lg font-semibold transition-all shadow-lg hover:shadow-blue-500/50"
                >
                  <RefreshCw className="w-5 h-5" />
                  Generate New Digits
                </button>

                <button
                  onClick={downloadResults}
                  className="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 px-6 py-3 rounded-lg font-semibold transition-colors"
                >
                  <Download className="w-5 h-5" />
                  Download Results
                </button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 shadow-xl">
              <h2 className="text-2xl font-bold mb-6">Generated Results</h2>
              
              {generatedImages.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                  <Brain className="w-16 h-16 mb-4 opacity-50" />
                  <p className="text-lg">No results yet</p>
                  <p className="text-sm">Start training or generate images to see results</p>
                </div>
              ) : (
                <div className="space-y-8">
                  {generatedImages.map((result, idx) => (
                    <div key={idx}>
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold text-purple-300">
                          {result.epoch === 'generated' ? 'Generated Samples' : `Epoch ${result.epoch}`}
                        </h3>
                        <span className="text-sm text-gray-400">
                          {result.images.length} images
                        </span>
                      </div>
                      <div className="grid grid-cols-8 gap-2 bg-slate-900/50 p-4 rounded-lg">
                        {result.images.map((img, i) => (
                          <div
                            key={i}
                            className="aspect-square bg-slate-700 rounded overflow-hidden border border-slate-600 hover:border-purple-500 transition-colors"
                          >
                            <img
                              src={img}
                              alt={`Generated digit ${i}`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-gray-300">
              <p className="font-semibold mb-1">How to use:</p>
              <ol className="list-decimal list-inside space-y-1 text-gray-400">
                <li>Configure training parameters and click "Start Training"</li>
                <li>Watch the loss metrics and generated samples during training</li>
                <li>After training, select a checkpoint and generate new digit images</li>
                <li>Download results for analysis or further use</li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MNISTGANInterface;