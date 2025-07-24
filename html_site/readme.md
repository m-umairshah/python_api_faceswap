# Face Analysis API Documentation

## Overview

The Face Analysis API provides comprehensive facial analysis and attractiveness rating capabilities with professional-grade mesh highlighting. This API can analyze faces in images from URLs and return detailed analysis results with highlighted facial features.

## Base URL
\`\`\`
http://localhost:5001
\`\`\`

## Features

- **Comprehensive Face Analysis**: Detailed analysis of facial features, shape, and characteristics
- **Face Rating**: Attractiveness scoring with detailed breakdown
- **Mesh Highlighting**: Professional facial mesh overlay with structural mapping
- **Multi-face Support**: Analyze multiple faces in a single image
- **RESTful API**: JSON request/response format
- **CORS Enabled**: Cross-origin requests supported

## Authentication

No authentication is required for this API.

## Rate Limits

Currently, no rate limits are implemented.

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

## Maximum File Size

16MB per image

---

# API Endpoints

## 1. Health Check

Check if the API is running and get basic information.

**Endpoint:** `GET /api/health`

**Response:**
\`\`\`json
{
  "success": true,
  "message": "Face Analysis API is running",
  "timestamp": "2025-01-23T18:40:02.123Z",
  "data": {
    "version": "1.0.0",
    "available_tools": ["face_analysis", "face_rating"],
    "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "webp"],
    "max_file_size_mb": 16
  }
}
\`\`\`

## 2. Analyze Image

Main endpoint for face analysis and rating.

**Endpoint:** `POST /api/analyze`

**Request Body:**
\`\`\`json
{
  "image_url": "https://example.com/image.jpg",
  "tool_type": "face_analysis"
}
\`\`\`

**Parameters:**
- `image_url` (required): URL of the image to analyze
- `tool_type` (optional): Type of analysis to perform
  - `"face_analysis"` (default): Comprehensive facial feature analysis
  - `"face_rating"`: Attractiveness rating analysis

### Face Analysis Response

\`\`\`json
{
  "success": true,
  "message": "Face analysis with mesh highlighting completed successfully. Found 2 face(s).",
  "timestamp": "2025-01-23T18:40:02.123Z",
  "data": {
    "tool_type": "face_analysis",
    "face_count": 2,
    "mesh_highlighting": true,
    "original_image_url": "http://localhost:5001/api/image/api_20250123_184002_abc123.jpg",
    "filename_base": "api_20250123_184002_abc123",
    "analysis_results": [
      {
        "face_id": 1,
        "bbox": [100, 50, 300, 250],
        "face_shape": {
          "shape": "Oval",
          "description": "You have an oval face with balanced proportions.",
          "measurements": {
            "width": 150.5,
            "length": 200.3,
            "jaw_width": 120.1,
            "forehead_width": 140.2,
            "width_to_length_ratio": 0.75
          },
          "characteristics": {
            "chin": "Rounded",
            "cheekbone": "High",
            "temple": "Normal",
            "apple_cheeks": "Prominent"
          },
          "recommendations": [
            "Most styles work well",
            "Maintain natural balance",
            "Experiment with different looks"
          ]
        },
        "eyes": {
          "characteristics": {
            "size": "Medium",
            "shape": "Almond",
            "spacing": "Average"
          },
          "measurements": {
            "distance": 45.2,
            "avg_width": 30.1,
            "avg_height": 12.5,
            "left_width": 29.8,
            "right_width": 30.4
          },
          "description": "You have medium, almond-shaped eyes with average spacing."
        },
        "eyebrows": {
          "characteristics": {
            "thickness": "Medium",
            "arch": "High",
            "spacing": "Close spacing"
          },
          "measurements": {
            "length": 35.6,
            "height": 12.3,
            "spacing": 8.9
          },
          "description": "You have medium eyebrows with high arch and close spacing."
        },
        "nose": {
          "characteristics": {
            "width": "Medium",
            "length": "Medium",
            "shape": "Straight",
            "bridge": "Medium bridge"
          },
          "measurements": {
            "width": 25.4,
            "height": 45.7,
            "bridge_width": 18.2,
            "width_ratio": 0.56
          },
          "description": "You have a medium, medium nose with a straight shape and medium bridge."
        },
        "lips": {
          "characteristics": {
            "width": "Medium",
            "thickness": "Full",
            "shape": "Balanced",
            "cupid_bow": "Pronounced"
          },
          "measurements": {
            "width": 52.3,
            "height": 28.9,
            "upper_height": 12.1,
            "lower_height": 16.8,
            "width_ratio": 1.81
          },
          "description": "You have medium, full lips with a balanced shape."
        },
        "overall_score": 78
      }
    ],
    "image_urls": {
      "face_1": {
        "original": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_original.png",
        "features": {
          "face_outline": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_face_outline.png",
          "eyes": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_eyes.png",
          "eyebrows": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_eyebrows.png",
          "nose": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_nose.png",
          "lips": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_lips.png"
        }
      }
    }
  }
}
\`\`\`

### Face Rating Response

\`\`\`json
{
  "success": true,
  "message": "Face rating with mesh highlighting completed successfully. Rated 1 face(s).",
  "timestamp": "2025-01-23T18:40:02.123Z",
  "data": {
    "tool_type": "face_rating",
    "face_count": 1,
    "mesh_highlighting": true,
    "average_rating": 7.2,
    "average_percentage": 72,
    "highest_rated": {
      "face_id": 1,
      "overall_rating": 7.2,
      "percentage": 72,
      "rating_message": "You are good looking!"
    },
    "original_image_url": "http://localhost:5001/api/image/api_20250123_184002_abc123.jpg",
    "filename_base": "api_20250123_184002_abc123",
    "individual_ratings": [
      {
        "face_id": 1,
        "overall_rating": 7.2,
        "percentage": 72,
        "rating_message": "You are good looking!",
        "feature_analysis": [
          "Good face width",
          "Good forehead size",
          "Good eye spacing",
          "Good nose for face",
          "Good nose length",
          "Good mouth width",
          "Good chin size"
        ],
        "symmetry_score": 85.3,
        "highlighted_image": "api_20250123_184002_abc123_face_1_highlighted.png",
        "measurements": {
          "face_width": 150.5,
          "face_height": 200.3,
          "forehead_width": 140.2,
          "interocular_distance": 45.2,
          "eye_span": 120.8,
          "nose_length": 45.7,
          "nose_width": 25.4,
          "mouth_width": 52.3,
          "jaw_width": 120.1
        },
        "proportions": {
          "face_ratio": 0.75,
          "forehead_ratio": 0.93,
          "eye_spacing_ratio": 0.30,
          "nose_length_ratio": 0.23,
          "nose_width_ratio": 0.17,
          "mouth_width_ratio": 0.35,
          "jaw_width_ratio": 0.80
        }
      }
    ],
    "image_urls": {
      "face_1": {
        "highlighted": "http://localhost:5001/api/image/api_20250123_184002_abc123_face_1_highlighted.png"
      }
    }
  }
}
\`\`\`

## 3. Get Processed Image

Retrieve processed images with mesh highlighting.

**Endpoint:** `GET /api/image/<filename>`

**Example:** `GET /api/image/api_20250123_184002_abc123_face_1_highlighted.png`

**Response:** Binary image data

## 4. Download File

Download analysis results or processed images.

**Endpoint:** `GET /api/download/<filename>`

**Example:** `GET /api/download/api_20250123_184002_abc123_analysis.json`

**Response:** File download

## 5. API Documentation

Get comprehensive API documentation.

**Endpoint:** `GET /api/docs`

**Response:** Complete API documentation in JSON format

---

# Integration Examples

## JavaScript/Web Integration

### Basic Usage with Fetch API

\`\`\`javascript
// Basic face analysis
async function analyzeFace(imageUrl) {
  try {
    const response = await fetch('http://localhost:5001/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: imageUrl,
        tool_type: 'face_analysis'
      })
    });

    const result = await response.json();
    
    if (result.success) {
      console.log('Analysis successful:', result.data);
      return result.data;
    } else {
      console.error('Analysis failed:', result.message);
      throw new Error(result.message);
    }
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Usage
analyzeFace('https://example.com/photo.jpg')
  .then(data => {
    console.log(`Found ${data.face_count} faces`);
    data.analysis_results.forEach(face => {
      console.log(`Face ${face.face_id}: ${face.face_shape.shape} shape, score: ${face.overall_score}`);
    });
  })
  .catch(error => {
    console.error('Error:', error);
  });
\`\`\`

### Face Rating with Results Display

\`\`\`javascript
async function rateFace(imageUrl) {
  try {
    const response = await fetch('http://localhost:5001/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: imageUrl,
        tool_type: 'face_rating'
      })
    });

    const result = await response.json();
    
    if (result.success) {
      displayRatingResults(result.data);
      return result.data;
    } else {
      throw new Error(result.message);
    }
  } catch (error) {
    console.error('Rating failed:', error);
    throw error;
  }
}

function displayRatingResults(data) {
  const resultsDiv = document.getElementById('results');
  
  let html = `
    <h3>Rating Results</h3>
    <p>Average Rating: ${data.average_rating}/10 (${data.average_percentage}%)</p>
    <div class="faces">
  `;
  
  data.individual_ratings.forEach(face => {
    html += `
      <div class="face-result">
        <h4>Face ${face.face_id}</h4>
        <p>Rating: ${face.overall_rating}/10</p>
        <p>${face.rating_message}</p>
        <p>Symmetry: ${face.symmetry_score.toFixed(1)}%</p>
        <img src="${data.image_urls[`face_${face.face_id}`].highlighted}" alt="Highlighted Face ${face.face_id}">
        <div class="features">
          <h5>Feature Analysis:</h5>
          <ul>
            ${face.feature_analysis.map(feature => `<li>${feature}</li>`).join('')}
          </ul>
        </div>
      </div>
    `;
  });
  
  html += '</div>';
  resultsDiv.innerHTML = html;
}
\`\`\`

### Complete Integration Example

\`\`\`html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Analysis Integration</title>
    <style>
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; }
        .form-group input, .form-group select { width: 100%; padding: 8px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .results { margin-top: 20px; }
        .face-result { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
        .face-result img { max-width: 300px; height: auto; }
        .loading { display: none; text-align: center; padding: 20px; }
        .error { color: red; padding: 10px; background: #ffe6e6; border: 1px solid red; }
        .success { color: green; padding: 10px; background: #e6ffe6; border: 1px solid green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Analysis API Integration</h1>
        
        <form id="analysisForm">
            <div class="form-group">
                <label for="imageUrl">Image URL:</label>
                <input type="url" id="imageUrl" required 
                       placeholder="https://example.com/image.jpg"
                       value="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500&h=500&fit=crop&crop=face">
            </div>
            
            <div class="form-group">
                <label for="toolType">Analysis Type:</label>
                <select id="toolType">
                    <option value="face_analysis">Face Analysis (Comprehensive)</option>
                    <option value="face_rating">Face Rating (Attractiveness)</option>
                </select>
            </div>
            
            <button type="submit" class="btn">Analyze Image</button>
        </form>
        
        <div id="loading" class="loading">
            <p>Processing image... Please wait.</p>
        </div>
        
        <div id="message"></div>
        <div id="results" class="results"></div>
    </div>

    <script>
        const API_BASE_URL = 'http://localhost:5001';
        
        document.getElementById('analysisForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const imageUrl = document.getElementById('imageUrl').value;
            const toolType = document.getElementById('toolType').value;
            
            showLoading(true);
            clearResults();
            
            try {
                const data = await analyzeImage(imageUrl, toolType);
                showMessage('Analysis completed successfully!', 'success');
                displayResults(data, toolType);
            } catch (error) {
                showMessage(`Error: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        });
        
        async function analyzeImage(imageUrl, toolType) {
            const response = await fetch(`${API_BASE_URL}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_url: imageUrl,
                    tool_type: toolType
                })
            });
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.message);
            }
            
            return result.data;
        }
        
        function displayResults(data, toolType) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <h3>Analysis Results</h3>
                <p><strong>Tool:</strong> ${data.tool_type}</p>
                <p><strong>Faces Found:</strong> ${data.face_count}</p>
                <p><strong>Mesh Highlighting:</strong> ${data.mesh_highlighting ? 'Enabled' : 'Disabled'}</p>
                <div class="original-image">
                    <h4>Original Image</h4>
                    <img src="${data.original_image_url}" alt="Original Image" style="max-width: 400px;">
                </div>
            `;
            
            if (toolType === 'face_analysis') {
                html += displayFaceAnalysisResults(data);
            } else if (toolType === 'face_rating') {
                html += displayFaceRatingResults(data);
            }
            
            resultsDiv.innerHTML = html;
        }
        
        function displayFaceAnalysisResults(data) {
            let html = '<div class="analysis-results">';
            
            data.analysis_results.forEach((face, index) => {
                const faceImages = data.image_urls[`face_${face.face_id}`];
                
                html += `
                    <div class="face-result">
                        <h4>Face ${face.face_id} Analysis</h4>
                        <div class="face-info">
                            <p><strong>Face Shape:</strong> ${face.face_shape.shape}</p>
                            <p><strong>Description:</strong> ${face.face_shape.description}</p>
                            <p><strong>Overall Score:</strong> ${face.overall_score}/100</p>
                        </div>
                        
                        <div class="face-images">
                            <h5>Original Face</h5>
                            <img src="${faceImages.original}" alt="Face ${face.face_id}" style="max-width: 200px;">
                            
                            <h5>Feature Highlights (with Mesh)</h5>
                            <div class="feature-images" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                                <div>
                                    <p>Face Outline</p>
                                    <img src="${faceImages.features.face_outline}" alt="Face Outline" style="width: 100%;">
                                </div>
                                <div>
                                    <p>Eyes</p>
                                    <img src="${faceImages.features.eyes}" alt="Eyes" style="width: 100%;">
                                </div>
                                <div>
                                    <p>Eyebrows</p>
                                    <img src="${faceImages.features.eyebrows}" alt="Eyebrows" style="width: 100%;">
                                </div>
                                <div>
                                    <p>Nose</p>
                                    <img src="${faceImages.features.nose}" alt="Nose" style="width: 100%;">
                                </div>
                                <div>
                                    <p>Lips</p>
                                    <img src="${faceImages.features.lips}" alt="Lips" style="width: 100%;">
                                </div>
                            </div>
                        </div>
                        
                        <div class="detailed-analysis">
                            <h5>Detailed Analysis</h5>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                                <div>
                                    <h6>Eyes</h6>
                                    <p>Size: ${face.eyes.characteristics.size}</p>
                                    <p>Shape: ${face.eyes.characteristics.shape}</p>
                                    <p>Spacing: ${face.eyes.characteristics.spacing}</p>
                                </div>
                                <div>
                                    <h6>Nose</h6>
                                    <p>Width: ${face.nose.characteristics.width}</p>
                                    <p>Length: ${face.nose.characteristics.length}</p>
                                    <p>Shape: ${face.nose.characteristics.shape}</p>
                                </div>
                                <div>
                                    <h6>Lips</h6>
                                    <p>Width: ${face.lips.characteristics.width}</p>
                                    <p>Thickness: ${face.lips.characteristics.thickness}</p>
                                    <p>Shape: ${face.lips.characteristics.shape}</p>
                                </div>
                                <div>
                                    <h6>Eyebrows</h6>
                                    <p>Thickness: ${face.eyebrows.characteristics.thickness}</p>
                                    <p>Arch: ${face.eyebrows.characteristics.arch}</p>
                                    <p>Spacing: ${face.eyebrows.characteristics.spacing}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            return html;
        }
        
        function displayFaceRatingResults(data) {
            let html = `
                <div class="rating-results">
                    <div class="summary">
                        <h4>Rating Summary</h4>
                        <p><strong>Average Rating:</strong> ${data.average_rating}/10 (${data.average_percentage}%)</p>
                        <p><strong>Highest Rated:</strong> Face ${data.highest_rated.face_id} - ${data.highest_rated.overall_rating}/10</p>
                    </div>
            `;
            
            data.individual_ratings.forEach(face => {
                const faceImages = data.image_urls[`face_${face.face_id}`];
                
                html += `
                    <div class="face-result">
                        <h4>Face ${face.face_id} Rating</h4>
                        <div class="rating-info">
                            <p><strong>Rating:</strong> ${face.overall_rating}/10 (${face.percentage}%)</p>
                            <p><strong>Message:</strong> ${face.rating_message}</p>
                            <p><strong>Symmetry Score:</strong> ${face.symmetry_score.toFixed(1)}%</p>
                        </div>
                        
                        <div class="highlighted-image">
                            <h5>Highlighted Face (with Mesh)</h5>
                            <img src="${faceImages.highlighted}" alt="Highlighted Face ${face.face_id}" style="max-width: 300px;">
                        </div>
                        
                        <div class="feature-analysis">
                            <h5>Feature Analysis</h5>
                            <ul>
                                ${face.feature_analysis.map(feature => `<li>${feature}</li>`).join('')}
                            </ul>
                        </div>
                        
                        <div class="measurements">
                            <h5>Measurements</h5>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                                <div>Face Width: ${face.measurements.face_width.toFixed(1)}</div>
                                <div>Face Height: ${face.measurements.face_height.toFixed(1)}</div>
                                <div>Nose Width: ${face.measurements.nose_width.toFixed(1)}</div>
                                <div>Mouth Width: ${face.measurements.mouth_width.toFixed(1)}</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            return html;
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function showMessage(message, type) {
            const messageDiv = document.getElementById('message');
            messageDiv.innerHTML = `<div class="${type}">${message}</div>`;
        }
        
        function clearResults() {
            document.getElementById('results').innerHTML = '';
            document.getElementById('message').innerHTML = '';
        }
    </script>
</body>
</html>
\`\`\`

## React Integration

\`\`\`jsx
import React, { useState } from 'react';

const FaceAnalysisComponent = () => {
  const [imageUrl, setImageUrl] = useState('');
  const [toolType, setToolType] = useState('face_analysis');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const API_BASE_URL = 'http://localhost:5001';

  const analyzeImage = async () => {
    if (!imageUrl) {
      setError('Please enter an image URL');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: imageUrl,
          tool_type: toolType
        })
      });

      const result = await response.json();

      if (result.success) {
        setResults(result.data);
      } else {
        setError(result.message);
      }
    } catch (err) {
      setError(`API request failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="face-analysis-component">
      <h2>Face Analysis</h2>
      
      <div className="form-group">
        <label>Image URL:</label>
        <input
          type="url"
          value={imageUrl}
          onChange={(e) => setImageUrl(e.target.value)}
          placeholder="https://example.com/image.jpg"
        />
      </div>

      <div className="form-group">
        <label>Analysis Type:</label>
        <select value={toolType} onChange={(e) => setToolType(e.target.value)}>
          <option value="face_analysis">Face Analysis</option>
          <option value="face_rating">Face Rating</option>
        </select>
      </div>

      <button onClick={analyzeImage} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze Image'}
      </button>

      {error && <div className="error">{error}</div>}

      {results && (
        <div className="results">
          <h3>Results</h3>
          <p>Found {results.face_count} face(s)</p>
          <p>Mesh highlighting: {results.mesh_highlighting ? 'Enabled' : 'Disabled'}</p>
          
          <img src={results.original_image_url || "/placeholder.svg"} alt="Original" style={{maxWidth: '400px'}} />

          {toolType === 'face_analysis' && (
            <div className="analysis-results">
              {results.analysis_results.map(face => (
                <div key={face.face_id} className="face-result">
                  <h4>Face {face.face_id}</h4>
                  <p>Shape: {face.face_shape.shape}</p>
                  <p>Score: {face.overall_score}/100</p>
                  
                  <div className="feature-images">
                    {Object.entries(results.image_urls[`face_${face.face_id}`].features).map(([feature, url]) => (
                      <div key={feature}>
                        <p>{feature.replace('_', ' ')}</p>
                        <img src={url || "/placeholder.svg"} alt={feature} style={{width: '150px'}} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {toolType === 'face_rating' && (
            <div className="rating-results">
              <p>Average Rating: {results.average_rating}/10</p>
              {results.individual_ratings.map(face => (
                <div key={face.face_id} className="face-result">
                  <h4>Face {face.face_id}</h4>
                  <p>Rating: {face.overall_rating}/10</p>
                  <p>{face.rating_message}</p>
                  <img src={results.image_urls[`face_${face.face_id || "/placeholder.svg"}`].highlighted} alt="Highlighted" style={{maxWidth: '300px'}} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FaceAnalysisComponent;
\`\`\`

## Python Integration

```python
import requests
import json

class FaceAnalysisAPI:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def analyze_image(self, image_url, tool_type="face_analysis"):
        """Analyze image using the API"""
        try:
            payload = {
                "image_url": image_url,
                "tool_type": tool_type
            }
            
            response = requests.post(
                f"{self.base_url}/api/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def download_image(self, filename, save_path):
        """Download processed image"""
        try:
            response = requests.get(f"{self.base_url}/api/image/{filename}")
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                return False
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return False

# Usage example
if __name__ == "__main__":
    api = FaceAnalysisAPI()
    
    # Health check
    health = api.health_check()
    print("API Health:", health)
    
    # Analyze image
    image_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500&h=500&fit=crop&crop=face"
    
    # Face analysis
    analysis_result = api.analyze_image(image_url, "face_analysis")
    if analysis_result["success"]:
        print(f"Found {analysis_result['data']['face_count']} faces")
        for face in analysis_result["data"]["analysis_results"]:
            print(f"Face {face['face_id']}: {face['face_shape']['shape']} shape, score: {face['overall_score']}")
    else:
        print("Analysis failed:", analysis_result["message"])
    
    # Face rating
    rating_result = api.analyze_image(image_url, "face_rating")
    if rating_result["success"]:
        print(f"Average rating: {rating_result['data']['average_rating']}/10")
        for face in rating_result["data"]["individual_ratings"]:
            print(f"Face {face['face_id']}: {face['overall_rating']}/10 - {face['rating_message']}")
    else:
        print("Rating failed:", rating_result["message"])
