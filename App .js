import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      // ✅ CORRECT ENDPOINT: /predict (not /predict-image)
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      <h1 className="text-3xl font-bold mb-6">DeepFake & Emotion Detection</h1>
      <div className="bg-white rounded-xl shadow-md p-6 w-full max-w-2xl">
        <input
          type="file"
          accept="image/*,video/*"
          onChange={handleFileChange}
          className="mb-4 w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        {preview && (
          <div className="mb-4">
            <p className="font-medium">Preview:</p>
            {file?.type.startsWith('image/') ? (
              <img src={preview} alt="Preview" className="max-h-64 rounded border mt-2" />
            ) : (
              <video src={preview} controls className="max-h-64 rounded border mt-2" />
            )}
          </div>
        )}
        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:bg-gray-400 transition"
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
        {error && <p className="text-red-500 mt-4">{error}</p>}
        {result && (
          <div className="mt-6 p-4 bg-gray-50 rounded border">
            <h2 className="text-xl font-semibold mb-2">Results</h2>
            <p>
              <span className="font-medium">Deepfake Detection:</span>{' '}
              <span className={result.label === 'FAKE' ? 'text-red-600' : 'text-green-600'}>
                {result.label}
              </span>
              {' '}
              (confidence: {(result.confidence * 100).toFixed(2)}%)
            </p>
            {result.emotion && (
              <p className="mt-2">
                <span className="font-medium">Detected Emotion:</span> {result.emotion}
              </p>
            )}
          </div>
        )}
      </div>
      <footer className="mt-8 text-gray-500 text-sm">
        Upload an image or video – we'll detect if it's a deepfake and recognize emotion (on images).
      </footer>
    </div>
  );
}

export default App;