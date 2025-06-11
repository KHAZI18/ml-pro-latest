import { useState, useRef } from 'react';

const emotions = ["anger", "happy", "sad", "neutral", "fear"];
const emotionColors = {
  anger: "bg-red-500",
  happy: "bg-yellow-400",
  sad: "bg-blue-500",
  neutral: "bg-gray-400",
  fear: "bg-purple-500"
};

const emotionEmojis = {
  anger: "😠",
  happy: "😄",
  sad: "😢",
  neutral: "😐",
  fear: "😨"
};

export default function EmotionRecognitionApp() {
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);
  const [mode, setMode] = useState("both");
  const [prediction, setPrediction] = useState(null);
  const [confidences, setConfidences] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("main"); // 'main', 'history', 'about'
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // Mock prediction history
  const [history, setHistory] = useState([
    {
      id: 1,
      date: "2025-05-18",
      text: "I'm feeling really anxious about the presentation",
      audioName: "anxiety_sample.wav",
      mode: "both",
      prediction: "fear",
      confidence: 0.81
    },
    {
      id: 2,
      date: "2025-05-17",
      text: "Today was absolutely wonderful!",
      audioName: null,
      mode: "text",
      prediction: "happy",
      confidence: 0.92
    }
  ]);
  
  

  const predictEmotion= () => {
    setLoading(true);
  
    // Simulate API call delay
    setTimeout(() => {
      // Demo predefined results based on text content
      let result;
      let mockConfidences = {};
  
      // Handle a broader range of emotions with more keywords
      const lowerText = text.toLowerCase();
  
      // Happy emotions
      if (lowerText.includes("happy") || lowerText.includes("joy") || lowerText.includes("excited") || lowerText.includes("great") || lowerText.includes("elated") || lowerText.includes("cheerful") || lowerText.includes("thrilled") || lowerText.includes("wonderful")) {
        result = "happy";
        mockConfidences = {
          "happy": 0.9, 
          "neutral": 0.05, 
          "anger": 0.02,
          "sad": 0.02,
          "fear": 0.01
        };
      }
      // Sad emotions
      else if (lowerText.includes("sad") || lowerText.includes("down") || lowerText.includes("depressed") || lowerText.includes("heartbroken") || lowerText.includes("unhappy") || lowerText.includes("mournful") || lowerText.includes("grief")) {
        result = "sad";
        mockConfidences = {
          "sad": 0.85, 
          "neutral": 0.1, 
          "fear": 0.02,
          "anger": 0.02,
          "happy": 0.01
        };
      }
      // Anger-related emotions
      else if (lowerText.includes("angry") || lowerText.includes("furious") || lowerText.includes("mad") || lowerText.includes("rage") || lowerText.includes("irritated") || lowerText.includes("enraged") || lowerText.includes("annoyed")) {
        result = "anger";
        mockConfidences = {
          "anger": 0.8, 
          "fear": 0.1, 
          "sad": 0.05,
          "neutral": 0.03,
          "happy": 0.02
        };
      }
      // Fear-related emotions
      else if (lowerText.includes("fear") || lowerText.includes("scared") || lowerText.includes("afraid") || lowerText.includes("terrified") || lowerText.includes("nervous") || lowerText.includes("anxious") || lowerText.includes("worried")) {
        result = "fear";
        mockConfidences = {
          "fear": 0.85, 
          "neutral": 0.1, 
          "sad": 0.03,
          "anger": 0.02,
          "happy": 0.01
        };
      }
      // Neutral or Confused emotions
      else if (lowerText.includes("confused") || lowerText.includes("uncertain") || lowerText.includes("meh") || lowerText.includes("whatever") || lowerText.includes("indifferent") || lowerText.includes("ambivalent")) {
        result = "neutral";
        mockConfidences = {
          "neutral": 0.7, 
          "sad": 0.15, 
          "happy": 0.1,
          "fear": 0.05,
          "anger": 0.05
        };
      }
      // Surprise emotions (expressions of unexpectedness or astonishment)
      else if (lowerText.includes("surprised") || lowerText.includes("shocked") || lowerText.includes("amazed") || lowerText.includes("astonished") || lowerText.includes("incredible") || lowerText.includes("wow")) {
        result = "happy";  // Can be mapped as a happy surprise
        mockConfidences = {
          "happy": 0.75, 
          "neutral": 0.15, 
          "fear": 0.05,
          "anger": 0.03,
          "sad": 0.02
        };
      }
      // Affectionate or loving emotions
      else if (lowerText.includes("love") || lowerText.includes("care") || lowerText.includes("appreciate") || lowerText.includes("affection") || lowerText.includes("fond") || lowerText.includes("adorable")) {
        result = "happy";  // Love and affection are typically mapped to happy emotions
        mockConfidences = {
          "happy": 0.8, 
          "neutral": 0.15, 
          "sad": 0.02,
          "anger": 0.01,
          "fear": 0.02
        };
      }
      // More neutral emotions (if no strong emotion is identified)
      else {
        result = "neutral";
        mockConfidences = {
          "neutral": 0.65, 
          "sad": 0.15, 
          "happy": 0.10,
          "fear": 0.05,
          "anger": 0.05
        };
      }
  
      // Setting the prediction and confidence results
      setPrediction(result);
      setConfidences(mockConfidences);
  
      // Add to history
      if (result) {
        const newHistoryItem = {
          id: Date.now(),
          date: new Date().toISOString().slice(0, 10),
          text: mode !== "audio" ? text : null,
          audioName: audioFile ? audioFile.name : null,
          mode: mode,
          prediction: result,
          confidence: mockConfidences[result]
        };
  
        setHistory(prev => [newHistoryItem, ...prev]);
      }
  
      setLoading(false);
    }, 1500);
  };
  
  
  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioPreview(URL.createObjectURL(file));
    }
  };
  
  // const handleSubmit = () => {
  //   if ((mode === "text" && text) || 
  //       (mode === "audio" && audioFile) || 
  //       (mode === "both" && text && audioFile)) {
  //     predictEmotion();
  //   } else {
  //     alert(`Please provide ${mode === "both" ? "both text and audio" : mode} input.`);
  //   }
  // };

  const handleSubmit = async () => {
    if (!canSubmit()) return;
  
    const formData = new FormData();
    formData.append("mode", mode);
    if (text) formData.append("text", text);
    if (audioFile) formData.append("audio", audioFile);
  
    setLoading(true);
  
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      const result = data.emotion;
  
      setPrediction(result);
      setConfidences({ [result]: 1 }); // optional - update if using actual confidences
    } catch (err) {
      console.error("Prediction failed:", err);
      alert("Prediction error");
    } finally {
      setLoading(false);
    }
  };
  
  
  const canSubmit = () => {
    return (mode === "text" && text.trim()) || 
           (mode === "audio" && audioFile) || 
           (mode === "both" && text.trim() && audioFile);
  };
  
  const handleReset = () => {
    setText("");
    setAudioFile(null);
    setAudioPreview(null);
    setPrediction(null);
    setConfidences(null);
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Generate a mock voice visualization
  const generateWaveform = () => {
    const segments = 20;
    let waveform = [];
    
    for (let i = 0; i < segments; i++) {
      // Generate a random height between 10% and 100%
      const height = 10 + Math.random() * 90;
      waveform.push(height);
    }
    
    return waveform;
  };

  // Main Analysis UI
  const renderMainTab = () => (
    <div className="flex flex-col md:flex-row gap-6">
      {/* Input Panel */}
      <div className="flex-1">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Input</h2>
        
        {/* Mode Selector */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Analysis Mode</label>
          <div className="grid grid-cols-3 gap-2">
            <button 
              onClick={() => setMode("text")} 
              className={`py-2 px-4 rounded-md border ${mode === "text" ? "bg-blue-500 text-white border-blue-600" : "bg-white text-gray-800 border-gray-300"}`}
            >
              Text Only
            </button>
            <button 
              onClick={() => setMode("audio")} 
              className={`py-2 px-4 rounded-md border ${mode === "audio" ? "bg-blue-500 text-white border-blue-600" : "bg-white text-gray-800 border-gray-300"}`}
            >
              Audio Only
            </button>
            <button 
              onClick={() => setMode("both")} 
              className={`py-2 px-4 rounded-md border ${mode === "both" ? "bg-blue-500 text-white border-blue-600" : "bg-white text-gray-800 border-gray-300"}`}
            >
              Multimodal
            </button>
          </div>
        </div>
        
        <div>
          {/* Text Input */}
          {(mode === "text" || mode === "both") && (
            <div className="mb-4">
              <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
                Text Input
              </label>
              <textarea
                id="text-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md shadow-sm"
                rows="4"
                placeholder="Enter text to analyze emotion (e.g., 'I'm feeling really happy today!')"
              />
            </div>
          )}
          
          {/* Audio Input */}
          {(mode === "audio" || mode === "both") && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Audio Input
              </label>
              <div className="border border-gray-300 rounded-md p-4 bg-gray-50">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="audio/*"
                  onChange={handleAudioChange}
                  className="hidden"
                />
                <div 
                  onClick={triggerFileInput}
                  className="cursor-pointer flex items-center justify-center p-4 border-2 border-dashed border-gray-400 rounded-md hover:border-blue-500 transition-colors"
                >
                  <div className="text-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    <p className="mt-2 text-sm text-gray-600">Click to upload audio file</p>
                    <p className="text-xs text-gray-500">WAV, MP3, or OGG</p>
                  </div>
                </div>
                
                {audioPreview && (
                  <div className="mt-4">
                    <audio 
                      ref={audioRef} 
                      controls 
                      className="w-full" 
                      src={audioPreview}
                    >
                      Your browser does not support the audio element.
                    </audio>
                    <button
                      type="button"
                      onClick={() => {
                        setAudioFile(null);
                        setAudioPreview(null);
                      }}
                      className="mt-2 text-sm text-red-600 hover:text-red-800"
                    >
                      Remove Audio
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Submit Button */}
          <div className="flex space-x-3">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit() || loading}
              className={`flex-1 py-2 px-4 rounded-md text-white font-medium ${canSubmit() && !loading ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-400 cursor-not-allowed'}`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </span>
              ) : (
                'Analyze Emotion'
              )}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="py-2 px-4 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-100"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
      
      {/* Results Panel */}
      <div className="flex-1 mt-6 md:mt-0">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
        {prediction ? (
          <div className="border rounded-lg overflow-hidden">
            <div className={`p-6 ${emotionColors[prediction]} text-white`}>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-bold">
                    {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
                  </h3>
                  <p>Detected Emotion</p>
                </div>
                <div className="text-5xl">
                  {emotionEmojis[prediction]}
                </div>
              </div>
            </div>
            
            <div className="bg-white p-6">
              <h4 className="font-medium text-gray-700 mb-2">Confidence Scores</h4>
              {confidences && emotions.map((emotion) => (
                <div key={emotion} className="mb-2">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                    </span>
                    <span className="text-sm font-medium text-gray-700">
                      {Math.round((confidences[emotion] || 0) * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${emotionColors[emotion]}`} 
                      style={{ width: `${(confidences[emotion] || 0) * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
              
              <div className="mt-4 pt-4 border-t border-gray-200">
                <h4 className="font-medium text-gray-700 mb-2">Analysis Details</h4>
                <p className="text-sm text-gray-600 mb-1">
                  <span className="font-medium">Mode:</span> {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </p>
                {text && mode !== "audio" && (
                  <p className="text-sm text-gray-600 mb-1">
                    <span className="font-medium">Text:</span> "{text}"
                  </p>
                )}
                {audioFile && mode !== "text" && (
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Audio:</span> {audioFile.name}
                  </p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 rounded-lg p-8 text-center border border-gray-200">
            <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-800 mb-2">No Analysis Yet</h3>
            <p className="text-gray-600">
              Input text and/or audio and click "Analyze Emotion" to see results.
            </p>
          </div>
        )}
        
        {/* Info Card */}
        <div className="mt-4 p-4 bg-blue-50 rounded-md border border-blue-100">
          <h4 className="text-sm font-medium text-blue-800 mb-1">About This Model</h4>
          <p className="text-xs text-blue-700">
            This multimodal emotion recognition system combines BERT for text analysis and CNN for audio processing to detect 5 emotions: anger, happiness, sadness, fear, and neutral states with high accuracy.
          </p>
        </div>
      </div>
    </div>
  );

  // History Tab
  const renderHistoryTab = () => (
    <div className="w-full">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Analysis History</h2>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Input</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mode</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {history.map((item) => (
              <tr key={item.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.date}</td>
                <td className="px-6 py-4 text-sm text-gray-900">
                  {item.text && <div className="truncate max-w-xs">{item.text}</div>}
                  {item.audioName && (
                    <div className="flex items-center text-gray-500 mt-1">
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                      </svg>
                      <span className="text-xs">{item.audioName}</span>
                    </div>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {item.mode.charAt(0).toUpperCase() + item.mode.slice(1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${item.prediction === 'neutral' ? 'bg-gray-100 text-gray-800' : item.prediction === 'happy' ? 'bg-yellow-100 text-yellow-800' : item.prediction === 'sad' ? 'bg-blue-100 text-blue-800' : item.prediction === 'anger' ? 'bg-red-100 text-red-800' : 'bg-purple-100 text-purple-800'}`}>
                    {item.prediction.charAt(0).toUpperCase() + item.prediction.slice(1)}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {Math.round(item.confidence * 100)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {history.length === 0 && (
        <div className="text-center py-8 bg-gray-50 rounded-md">
          <p className="text-gray-500">No analysis history yet</p>
        </div>
      )}
    </div>
  );

  // About Tab
  const renderAboutTab = () => (
    <div className="w-full">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">About This Application</h2>
      
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Multimodal Emotion Recognition System</h3>
          
          <div className="prose text-gray-700">
            <p>This application uses an advanced multimodal emotion recognition system that can analyze emotions from text, audio, or both inputs simultaneously. The system is based on state-of-the-art deep learning models.</p>
            
            <h4 className="text-md font-medium text-gray-800 mt-4 mb-2">Model Architecture</h4>
            <p>The system utilizes:</p>
            <ul className="list-disc pl-5 mb-4">
              <li><span className="font-medium">BERT</span> (Bidirectional Encoder Representations from Transformers) for processing text inputs</li>
              <li><span className="font-medium">CNN</span> (Convolutional Neural Network) for audio feature extraction from MFCC features</li>
              <li>A <span className="font-medium">fusion layer</span> that combines both modalities for more accurate predictions</li>
            </ul>
            
            <h4 className="text-md font-medium text-gray-800 mb-2">Detected Emotions</h4>
            <p>The system can recognize five basic emotions:</p>
            <div className="grid grid-cols-5 gap-2 my-4">
              {emotions.map(emotion => (
                <div key={emotion} className={`p-2 rounded-md text-center text-white ${emotionColors[emotion]}`}>
                  <div className="text-2xl">{emotionEmojis[emotion]}</div>
                  <div className="text-sm">{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</div>
                </div>
              ))}
            </div>
            
            <h4 className="text-md font-medium text-gray-800 mb-2">Dataset</h4>
            <p>The model was trained on the Indian Emotional Speech Corpus (IESC), which contains labeled emotional speech samples from multiple speakers in various emotional states.</p>
            
            <h4 className="text-md font-medium text-gray-800 mt-4 mb-2">Technical Details</h4>
            <div className="bg-gray-50 p-4 rounded-md">
              <ul className="list-disc pl-5">
                <li>Backend built with PyTorch</li>
                <li>Frontend using React and Tailwind CSS</li>
                <li>Audio processing with librosa</li>
                <li>Text processing with the transformers library</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-100 to-gray-200 p-4">
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-6">
          <h1 className="text-3xl font-bold text-white">Multimodal Emotion Recognition</h1>
          <p className="text-blue-100 mt-2">Analyze emotions from text and speech using advanced deep learning</p>
          
          {/* Main Navigation Tabs */}
          <div className="mt-6 flex space-x-1 border-b border-blue-500">
            <button
              onClick={() => setActiveTab("main")}
              className={`py-2 px-4 text-sm font-medium rounded-t-md transition-colors ${activeTab === "main" ? "bg-white text-blue-600" : "bg-blue-700 text-white hover:bg-blue-800"}`}
            >
              Analysis
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`py-2 px-4 text-sm font-medium rounded-t-md transition-colors ${activeTab === "history" ? "bg-white text-blue-600" : "bg-blue-700 text-white hover:bg-blue-800"}`}
            >
              History
            </button>
            <button
              onClick={() => setActiveTab("about")}
              className={`py-2 px-4 text-sm font-medium rounded-t-md transition-colors ${activeTab === "about" ? "bg-white text-blue-600" : "bg-blue-700 text-white hover:bg-blue-800"}`}
            >
              About
            </button>
          </div>
        </div>
        
        <div className="p-6">
          {activeTab === "main" && renderMainTab()}
          {activeTab === "history" && renderHistoryTab()}
          {activeTab === "about" && renderAboutTab()}
        </div>
      </div>
      
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>Multimodal Emotion Recognition System • Based on the IESC Dataset</p>
      </div>
    </div>
  );
}



// import { useState, useRef } from 'react';

// const emotions = ["anger", "happy", "sad", "neutral", "fear"];
// const emotionColors = {
//   anger: "bg-red-500",
//   happy: "bg-yellow-400",
//   sad: "bg-blue-500",
//   neutral: "bg-gray-400",
//   fear: "bg-purple-500"
// };

// const emotionEmojis = {
//   anger: "😠",
//   happy: "😄",
//   sad: "😢",
//   neutral: "😐",
//   fear: "😨"
// };

// export default function EmotionRecognitionApp() {
//   const [text, setText] = useState("");
//   const [audioFile, setAudioFile] = useState(null);
//   const [audioPreview, setAudioPreview] = useState(null);
//   const [mode, setMode] = useState("both");
//   const [prediction, setPrediction] = useState(null);
//   const [confidences, setConfidences] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [activeTab, setActiveTab] = useState("main"); // 'main', 'history', 'about'
//   const audioRef = useRef(null);
//   const fileInputRef = useRef(null);
//   const [error, setError] = useState(null);
  
//   // Mock prediction history
//   const [history, setHistory] = useState([
//     {
//       id: 1,
//       date: "2025-05-18",
//       text: "I'm feeling really anxious about the presentation",
//       audioName: "anxiety_sample.wav",
//       mode: "both",
//       prediction: "fear",
//       confidence: 0.81
//     },
//     {
//       id: 2,
//       date: "2025-05-17",
//       text: "Today was absolutely wonderful!",
//       audioName: null,
//       mode: "text",
//       prediction: "happy",
//       confidence: 0.92
//     }
//   ]);
  
//   const handleSubmit = async () => {
//     if (!canSubmit()) return;
    
//     setError(null);
//     setLoading(true);
    
//     // Check if we should use the backend API or fallback to demo mode
//     const useBackendApi = false; // Set to true when backend is available
    
//     try {
//       let result;
//       let confidence;
      
//       if (useBackendApi) {
//         // Real API implementation
//         const formData = new FormData();
//         formData.append("mode", mode);
//         if (text) formData.append("text", text);
//         if (audioFile) formData.append("audio", audioFile);
        
//         const response = await fetch("http://localhost:8000/predict", {
//           method: "POST",
//           body: formData,
//         });
        
//         if (!response.ok) {
//           throw new Error(`API error: ${response.status}`);
//         }
        
//         const data = await response.json();
//         result = data.emotion;
//         confidence = 0.9; // Assume 90% confidence for API results
//       } else {
//         // Demo implementation using client-side prediction
//         result = predictEmotionDemo(text, audioFile?.name);
//         confidence = generateConfidence(result);
//       }
      
//       // Set prediction results
//       setPrediction(result);
      
//       // Generate confidence scores
//       const mockConfidences = {};
//       emotions.forEach(emotion => {
//         if (emotion === result) {
//           mockConfidences[emotion] = confidence;
//         } else {
//           // Distribute remaining probability among other emotions
//           mockConfidences[emotion] = ((1 - confidence) / (emotions.length - 1)).toFixed(2);
//         }
//       });
      
//       setConfidences(mockConfidences);
      
//       // Add to history
//       const newHistoryItem = {
//         id: Date.now(),
//         date: new Date().toISOString().slice(0, 10),
//         text: mode !== "audio" ? text : null,
//         audioName: audioFile ? audioFile.name : null,
//         mode: mode,
//         prediction: result,
//         confidence: confidence
//       };
      
//       setHistory(prev => [newHistoryItem, ...prev]);
      
//     } catch (err) {
//       console.error("Prediction failed:", err);
//       setError("Failed to analyze emotion. Please try again.");
//     } finally {
//       setLoading(false);
//     }
//   };
  
//   // Demo function to predict emotion based on text content
//   const predictEmotionDemo = (textInput, audioFileName) => {
//     if (!textInput && !audioFileName) return "neutral";
    
//     // Text-based prediction
//     if (textInput) {
//       const lowerText = textInput.toLowerCase();
      
//       // Happy emotions
//       if (lowerText.includes("happy") || lowerText.includes("joy") || 
//           lowerText.includes("excited") || lowerText.includes("great") || 
//           lowerText.includes("elated") || lowerText.includes("cheerful") || 
//           lowerText.includes("thrilled") || lowerText.includes("wonderful")) {
//         return "happy";
//       }
//       // Sad emotions
//       else if (lowerText.includes("sad") || lowerText.includes("down") || 
//               lowerText.includes("depressed") || lowerText.includes("heartbroken") || 
//               lowerText.includes("unhappy") || lowerText.includes("mournful") || 
//               lowerText.includes("grief")) {
//         return "sad";
//       }
//       // Anger-related emotions
//       else if (lowerText.includes("angry") || lowerText.includes("furious") || 
//               lowerText.includes("mad") || lowerText.includes("rage") || 
//               lowerText.includes("irritated") || lowerText.includes("enraged") || 
//               lowerText.includes("annoyed")) {
//         return "anger";
//       }
//       // Fear-related emotions
//       else if (lowerText.includes("fear") || lowerText.includes("scared") || 
//               lowerText.includes("afraid") || lowerText.includes("terrified") || 
//               lowerText.includes("nervous") || lowerText.includes("anxious") || 
//               lowerText.includes("worried")) {
//         return "fear";
//       }
//     }
    
//     // Audio-based prediction (simplistic demo)
//     if (audioFileName && !textInput) {
//       const lowerFileName = audioFileName.toLowerCase();
//       for (const emotion of emotions) {
//         if (lowerFileName.includes(emotion)) {
//           return emotion;
//         }
//       }
//       // Random emotion if no match in filename
//       return emotions[Math.floor(Math.random() * emotions.length)];
//     }
    
//     // Default to neutral
//     return "neutral";
//   };
  
//   // Generate a random confidence score with a bias toward high confidence
//   const generateConfidence = (emotion) => {
//     // Base confidence between 0.7 and 0.95
//     const baseConfidence = 0.7 + (Math.random() * 0.25);
//     return parseFloat(baseConfidence.toFixed(2));
//   };
  
//   const handleAudioChange = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       setAudioFile(file);
//       setAudioPreview(URL.createObjectURL(file));
//     }
//   };
  
//   const canSubmit = () => {
//     return (mode === "text" && text.trim()) || 
//            (mode === "audio" && audioFile) || 
//            (mode === "both" && text.trim() && audioFile);
//   };
  
//   const handleReset = () => {
//     setText("");
//     setAudioFile(null);
//     setAudioPreview(null);
//     setPrediction(null);
//     setConfidences(null);
//     setError(null);
//   };

//   const triggerFileInput = () => {
//     if (fileInputRef.current) {
//       fileInputRef.current.click();
//     }
//   };

//   // Main Analysis UI
//   const renderMainTab = () => (
//     <div className="flex flex-col md:flex-row gap-6">
//       {/* Input Panel */}
//       <div className="flex-1">
//         <h2 className="text-xl font-semibold text-gray-800 mb-4">Input</h2>
        
//         {/* Mode Selector */}
//         <div className="mb-4">
//           <label className="block text-sm font-medium text-gray-700 mb-2">Analysis Mode</label>
//           <div className="grid grid-cols-3 gap-2">
//             <button 
//               onClick={() => setMode("text")} 
//               className={`py-2 px-4 rounded-md border ${mode === "text" ? "bg-blue-500 text-white border-blue-600" : "bg-white text-gray-800 border-gray-300"}`}
//             >
//               Text Only
//             </button>
//             <button 
//               onClick={() => setMode("audio")} 
//               className={`py-2 px-4 rounded-md border ${mode === "audio" ? "bg-blue-500 text-white border-blue-600" : "bg-white text-gray-800 border-gray-300"}`}
//             >
//               Audio Only
//             </button>
//             <button 
//               onClick={() => setMode("both")} 
//               className={`py-2 px-4 rounded-md border ${mode === "both" ? "bg-blue-500 text-white border-blue-600" : "bg-white text-gray-800 border-gray-300"}`}
//             >
//               Multimodal
//             </button>
//           </div>
//         </div>
        
//         <div>
//           {/* Text Input */}
//           {(mode === "text" || mode === "both") && (
//             <div className="mb-4">
//               <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
//                 Text Input
//               </label>
//               <textarea
//                 id="text-input"
//                 value={text}
//                 onChange={(e) => setText(e.target.value)}
//                 className="w-full p-3 border border-gray-300 rounded-md shadow-sm"
//                 rows="4"
//                 placeholder="Enter text to analyze emotion (e.g., 'I'm feeling really happy today!')"
//               />
//             </div>
//           )}
          
//           {/* Audio Input */}
//           {(mode === "audio" || mode === "both") && (
//             <div className="mb-4">
//               <label className="block text-sm font-medium text-gray-700 mb-2">
//                 Audio Input
//               </label>
//               <div className="border border-gray-300 rounded-md p-4 bg-gray-50">
//                 <input
//                   type="file"
//                   ref={fileInputRef}
//                   accept="audio/*"
//                   onChange={handleAudioChange}
//                   className="hidden"
//                 />
//                 <div 
//                   onClick={triggerFileInput}
//                   className="cursor-pointer flex items-center justify-center p-4 border-2 border-dashed border-gray-400 rounded-md hover:border-blue-500 transition-colors"
//                 >
//                   <div className="text-center">
//                     <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
//                     </svg>
//                     <p className="mt-2 text-sm text-gray-600">Click to upload audio file</p>
//                     <p className="text-xs text-gray-500">WAV, MP3, or OGG</p>
//                   </div>
//                 </div>
                
//                 {audioPreview && (
//                   <div className="mt-4">
//                     <audio 
//                       ref={audioRef} 
//                       controls 
//                       className="w-full" 
//                       src={audioPreview}
//                     >
//                       Your browser does not support the audio element.
//                     </audio>
//                     <button
//                       type="button"
//                       onClick={() => {
//                         setAudioFile(null);
//                         setAudioPreview(null);
//                       }}
//                       className="mt-2 text-sm text-red-600 hover:text-red-800"
//                     >
//                       Remove Audio
//                     </button>
//                   </div>
//                 )}
//               </div>
//             </div>
//           )}
          
//           {/* Error Message */}
//           {error && (
//             <div className="mb-4 p-3 bg-red-100 border border-red-200 rounded-md text-red-700 text-sm">
//               {error}
//             </div>
//           )}
          
//           {/* Submit Button */}
//           <div className="flex space-x-3">
//             <button
//               type="button"
//               onClick={handleSubmit}
//               disabled={!canSubmit() || loading}
//               className={`flex-1 py-2 px-4 rounded-md text-white font-medium ${canSubmit() && !loading ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-400 cursor-not-allowed'}`}
//             >
//               {loading ? (
//                 <span className="flex items-center justify-center">
//                   <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
//                     <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
//                     <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
//                   </svg>
//                   Analyzing...
//                 </span>
//               ) : (
//                 'Analyze Emotion'
//               )}
//             </button>
//             <button
//               type="button"
//               onClick={handleReset}
//               className="py-2 px-4 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-100"
//             >
//               Reset
//             </button>
//           </div>
//         </div>
//       </div>
      
//       {/* Results Panel */}
//       <div className="flex-1 mt-6 md:mt-0">
//         <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
//         {prediction ? (
//           <div className="border rounded-lg overflow-hidden">
//             <div className={`p-6 ${emotionColors[prediction]} text-white`}>
//               <div className="flex items-center justify-between">
//                 <div>
//                   <h3 className="text-2xl font-bold">
//                     {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
//                   </h3>
//                   <p>Detected Emotion</p>
//                 </div>
//                 <div className="text-5xl">
//                   {emotionEmojis[prediction]}
//                 </div>
//               </div>
//             </div>
            
//             <div className="bg-white p-6">
//               <h4 className="font-medium text-gray-700 mb-2">Confidence Scores</h4>
//               {confidences && emotions.map((emotion) => (
//                 <div key={emotion} className="mb-2">
//                   <div className="flex justify-between mb-1">
//                     <span className="text-sm font-medium text-gray-700">
//                       {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
//                     </span>
//                     <span className="text-sm font-medium text-gray-700">
//                       {Math.round((confidences[emotion] || 0) * 100)}%
//                     </span>
//                   </div>
//                   <div className="w-full bg-gray-200 rounded-full h-2">
//                     <div 
//                       className={`h-2 rounded-full ${emotionColors[emotion]}`} 
//                       style={{ width: `${(confidences[emotion] || 0) * 100}%` }}
//                     ></div>
//                   </div>
//                 </div>
//               ))}
              
//               <div className="mt-4 pt-4 border-t border-gray-200">
//                 <h4 className="font-medium text-gray-700 mb-2">Analysis Details</h4>
//                 <p className="text-sm text-gray-600 mb-1">
//                   <span className="font-medium">Mode:</span> {mode.charAt(0).toUpperCase() + mode.slice(1)}
//                 </p>
//                 {text && mode !== "audio" && (
//                   <p className="text-sm text-gray-600 mb-1">
//                     <span className="font-medium">Text:</span> "{text}"
//                   </p>
//                 )}
//                 {audioFile && mode !== "text" && (
//                   <p className="text-sm text-gray-600">
//                     <span className="font-medium">Audio:</span> {audioFile.name}
//                   </p>
//                 )}
//               </div>
//             </div>
//           </div>
//         ) : (
//           <div className="bg-gray-50 rounded-lg p-8 text-center border border-gray-200">
//             <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
//               <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
//               </svg>
//             </div>
//             <h3 className="text-lg font-medium text-gray-800 mb-2">No Analysis Yet</h3>
//             <p className="text-gray-600">
//               Input text and/or audio and click "Analyze Emotion" to see results.
//             </p>
//           </div>
//         )}
        
//         {/* Info Card */}
//         <div className="mt-4 p-4 bg-blue-50 rounded-md border border-blue-100">
//           <h4 className="text-sm font-medium text-blue-800 mb-1">About This Model</h4>
//           <p className="text-xs text-blue-700">
//             This multimodal emotion recognition system combines BERT for text analysis and CNN for audio processing to detect 5 emotions: anger, happiness, sadness, fear, and neutral states with high accuracy.
//           </p>
//         </div>
//       </div>
//     </div>
//   );
//   // History Tab
//   const renderHistoryTab = () => (
//     <div className="w-full">
//       <h2 className="text-xl font-semibold text-gray-800 mb-4">Analysis History</h2>
      
//       <div className="overflow-x-auto">
//         <table className="min-w-full divide-y divide-gray-200">
//           <thead className="bg-gray-50">
//             <tr>
//               <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
//               <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Input</th>
//               <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mode</th>
//               <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
//               <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
//             </tr>
//           </thead>
//           <tbody className="bg-white divide-y divide-gray-200">
//             {history.map((item) => (
//               <tr key={item.id} className="hover:bg-gray-50">
//                 <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.date}</td>
//                 <td className="px-6 py-4 text-sm text-gray-900">
//                   {item.text && <div className="truncate max-w-xs">{item.text}</div>}
//                   {item.audioName && (
//                     <div className="flex items-center text-gray-500 mt-1">
//                       <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
//                       </svg>
//                       <span className="text-xs">{item.audioName}</span>
//                     </div>
//                   )}
//                 </td>
//                 <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
//                   {item.mode.charAt(0).toUpperCase() + item.mode.slice(1)}
//                 </td>
//                 <td className="px-6 py-4 whitespace-nowrap">
//                   <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${item.prediction === 'neutral' ? 'bg-gray-100 text-gray-800' : item.prediction === 'happy' ? 'bg-yellow-100 text-yellow-800' : item.prediction === 'sad' ? 'bg-blue-100 text-blue-800' : item.prediction === 'anger' ? 'bg-red-100 text-red-800' : 'bg-purple-100 text-purple-800'}`}>
//                     {item.prediction.charAt(0).toUpperCase() + item.prediction.slice(1)}
//                   </span>
//                 </td>
//                 <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
//                   {Math.round(item.confidence * 100)}%
//                 </td>
//               </tr>
//             ))}
//           </tbody>
//         </table>
//       </div>
      
//       {history.length === 0 && (
//         <div className="text-center py-8 bg-gray-50 rounded-md">
//           <p className="text-gray-500">No analysis history yet</p>
//         </div>
//       )}
//     </div>
//   );

//   // About Tab
//   const renderAboutTab = () => (
//     <div className="w-full">
//       <h2 className="text-xl font-semibold text-gray-800 mb-4">About This Application</h2>
      
//       <div className="bg-white rounded-lg shadow overflow-hidden">
//         <div className="p-6">
//           <h3 className="text-lg font-medium text-gray-900 mb-4">Multimodal Emotion Recognition System</h3>
          
//           <div className="prose text-gray-700">
//             <p>This application uses an advanced multimodal emotion recognition system that can analyze emotions from text, audio, or both inputs simultaneously. The system is based on state-of-the-art deep learning models.</p>
            
//             <h4 className="text-md font-medium text-gray-800 mt-4 mb-2">Model Architecture</h4>
//             <p>The system utilizes:</p>
//             <ul className="list-disc pl-5 mb-4">
//               <li><span className="font-medium">BERT</span> (Bidirectional Encoder Representations from Transformers) for processing text inputs</li>
//               <li><span className="font-medium">CNN</span> (Convolutional Neural Network) for audio feature extraction from MFCC features</li>
//               <li>A <span className="font-medium">fusion layer</span> that combines both modalities for more accurate predictions</li>
//             </ul>
            
//             <h4 className="text-md font-medium text-gray-800 mb-2">Detected Emotions</h4>
//             <p>The system can recognize five basic emotions:</p>
//             <div className="grid grid-cols-5 gap-2 my-4">
//               {emotions.map(emotion => (
//                 <div key={emotion} className={`p-2 rounded-md text-center text-white ${emotionColors[emotion]}`}>
//                   <div className="text-2xl">{emotionEmojis[emotion]}</div>
//                   <div className="text-sm">{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</div>
//                 </div>
//               ))}
//             </div>
            
//             <h4 className="text-md font-medium text-gray-800 mb-2">Dataset</h4>
//             <p>The model was trained on the Indian Emotional Speech Corpus (IESC), which contains labeled emotional speech samples from multiple speakers in various emotional states.</p>
            
//             <h4 className="text-md font-medium text-gray-800 mt-4 mb-2">Technical Details</h4>
//             <div className="bg-gray-50 p-4 rounded-md">
//               <ul className="list-disc pl-5">
//                 <li>Backend built with PyTorch</li>
//                 <li>Frontend using React and Tailwind CSS</li>
//                 <li>Audio processing with librosa</li>
//                 <li>Text processing with the transformers library</li>
//               </ul>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );

//   return (
//     <div className="min-h-screen bg-gradient-to-b from-gray-100 to-gray-200 p-4">
//       <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
//         <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-6">
//           <h1 className="text-3xl font-bold text-white">Multimodal Emotion Recognition</h1>
//           <p className="text-blue-100 mt-2">Analyze emotions from text and speech using advanced deep learning</p>
          
//           {/* Main Navigation Tabs */}
//           <div className="mt-6 flex space-x-1 border-b border-blue-500">
//             <button
//               onClick={() => setActiveTab("main")}
//               className={`py-2 px-4 text-sm font-medium rounded-t-md transition-colors ${activeTab === "main" ? "bg-white text-blue-600" : "bg-blue-700 text-white hover:bg-blue-800"}`}
//             >
//               Analysis
//             </button>
//             <button
//               onClick={() => setActiveTab("history")}
//               className={`py-2 px-4 text-sm font-medium rounded-t-md transition-colors ${activeTab === "history" ? "bg-white text-blue-600" : "bg-blue-700 text-white hover:bg-blue-800"}`}
//             >
//               History
//             </button>
//             <button
//               onClick={() => setActiveTab("about")}
//               className={`py-2 px-4 text-sm font-medium rounded-t-md transition-colors ${activeTab === "about" ? "bg-white text-blue-600" : "bg-blue-700 text-white hover:bg-blue-800"}`}
//             >
//               About
//             </button>
//           </div>
//         </div>
        
//         <div className="p-6">
//           {activeTab === "main" && renderMainTab()}
//           {activeTab === "history" && renderHistoryTab()}
//           {activeTab === "about" && renderAboutTab()}
//         </div>
//       </div>
      
//       <div className="mt-8 text-center text-sm text-gray-500">
//         <p>Multimodal Emotion Recognition System • Based on the IESC Dataset</p>
//       </div>
//     </div>
//   );
// }