# Damage Detection Model API

This project implements a deep learning model for detecting damage in images, served through a Flask API.

## Project Structure
```
.
├── app/
│   ├── model/
│   │   └── best_model/     # Will be generated when training
│   └── main.py            # Flask application
├── coe_project3_part1n2.ipynb  # Model training notebook
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Complete Setup and Testing Instructions

### 1. Training the Model (Optional)
If you want to train the model yourself:
1. Make sure you have the dataset in `coe379L-sp25/datasets/unit03/Project3/`
2. Open and run `coe_project3_part1n2.ipynb` in Jupyter Notebook
3. The trained model will be saved to `app/model/best_model/`

### 2. Running the API Server

#### Option 1: Using Pre-built Docker Image (Quickest)
```bash
# Pull the image
docker pull parpat/damage-detection:latest

# Start the server
docker-compose up
```

#### Option 2: Building Locally (From Source)
```bash
# Build the image
docker build -t damage-detection .

# Run the container
docker run -p 5000:5000 damage-detection
```

### 3. Testing the Model

#### A. Using Python
```python
import requests
from PIL import Image
import os

def test_model(image_path):
    # Test the endpoints
    base_url = 'http://localhost:5000'
    
    # Test health check
    response = requests.get(f"{base_url}/")
    print("Health Check:", response.text)
    
    # Test model summary
    response = requests.get(f"{base_url}/summary")
    print("\nModel Summary:", response.json())
    
    # Test inference
    with open(image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post(f"{base_url}/inference", files=files)
        print(f"\nPrediction for {os.path.basename(image_path)}:", response.json())

# Example usage
test_model('path_to_your_test_image.jpg')
```

#### B. Using cURL
```bash
# Health check
curl http://localhost:5000/

# Get model summary
curl http://localhost:5000/summary

# Test inference (replace path_to_image.jpg with your image)
curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/inference
```

#### C. Using the Test Dataset
If you have access to the original dataset:
```bash
# Test a damage image
curl -X POST -F "image=@coe379L-sp25/datasets/unit03/Project3/damage/-93.66109_30.212114.jpeg" http://localhost:5000/inference

# Test a no_damage image
curl -X POST -F "image=@coe379L-sp25/datasets/unit03/Project3/no_damage/-95.06212_29.829257000000002.jpeg" http://localhost:5000/inference
```

### 4. Expected Responses

- GET `/`
  ```
  "Inference server running."
  ```

- GET `/summary`
  ```json
  {
    "model_summary": "Model architecture details..."
  }
  ```

- POST `/inference`
  ```json
  {
    "prediction": "damage"  // or "no_damage"
  }
  ```

## Requirements
- Python 3.8+
- Docker and Docker Compose
- For local training:
  - TensorFlow 2.x
  - Jupyter Notebook
  - NumPy
  - Pillow
  - scikit-learn

## Troubleshooting
1. If the server doesn't start, check if port 5000 is available:
   ```bash
   # On Windows
   netstat -ano | findstr :5000
   # On Linux/Mac
   lsof -i :5000
   ```

2. If you get connection refused errors, wait a few seconds after starting the container for the model to load

3. For image errors, ensure:
   - Image is in JPEG/PNG format
   - Image can be opened by Pillow
   - Image has 3 color channels (RGB) 