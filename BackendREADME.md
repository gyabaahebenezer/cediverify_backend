# CediVerify Backend

A deep learning-powered API for authenticating Ghanaian currency (Ghana Cedis) using a two-stage verification system with denomination classification and anomaly detection.

## Overview

**CediVerify Backend** is a Python-based REST API that uses neural networks to verify the authenticity of Ghana Cedis notes and coins. It employs a sophisticated two-stage verification approach:

- **Stage 1:** Denomination Classifier - Identifies which denomination of currency is in the image
- **Stage 2:** Per-Denomination Autoencoders - Detects anomalies (counterfeits) by measuring reconstruction error on genuine currency patterns

### Key Features

✅ **Two-Stage Verification** - Combine denomination recognition with anomaly detection  
✅ **13 Currency Denominations** - Supports coins and notes in various denominations  
✅ **Pre-trained Models** - Ready-to-use models trained on genuine Cedis images  
✅ **Anomaly Detection** - Autoencoders trained ONLY on genuine currency  
✅ **Fast Inference** - GPU support with optional CPU fallback  
✅ **REST API** - FastAPI-based HTTP endpoints  
✅ **CORS Enabled** - Cross-origin requests supported  
✅ **API Key Security** - Optional authentication for production  
✅ **Model Preloading** - Load models at startup for sub-second responses  

## Tech Stack

**Deep Learning:**
- PyTorch v2.0+ - Deep learning framework
- torchvision - Computer vision utilities and pre-trained models
- Pillow - Image processing

**Web Framework:**
- FastAPI - Modern, fast web framework for building APIs
- Uvicorn - ASGI web server
- python-multipart - File upload handling

**Hardware:**
- CUDA Support - GPU acceleration (NVIDIA)
- MPS Support - Metal Performance Shaders (Apple Silicon)
- CPU Fallback - Automatic CPU processing if GPU unavailable

**Other:**
- Python 3.11.9 - Programming language

## Project Structure

```
cediverify_backend/
├── api.py                           # FastAPI application & endpoints
├── model.py                         # Neural network architectures
├── inference.py                     # Verification logic & two-stage pipeline
├── train.py                         # Training script for autoencoders
├── train_denomination.py            # Training script for classifier
├── train_per_denomination.py        # Specialized denomination training
├── inference.py                     # Full verification pipeline
├── denomination_classifier.pth      # Pre-trained denomination classifier
├── class_names.txt                  # List of supported denominations
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version specification
├── models/
│   ├── 1_Cedi_coin_autoencoder.pth
│   ├── 1_Cedi_note_autoencoder.pth
│   ├── 2_Cedi_coin_autoencoder.pth
│   ├── 2_Cedi_note_autoencoder.pth
│   ├── 5_Cedi_note_autoencoder.pth
│   ├── 10_Cedi_note_autoencoder.pth
│   ├── 10_pesewas_coin_autoencoder.pth
│   ├── 20_Cedi_note_autoencoder.pth
│   ├── 20_pesewas_coin_autoencoder.pth
│   ├── 50_Cedi_note_autoencoder.pth
│   ├── 50_pesewas_coin_autoencoder.pth
│   ├── 100_Cedi_note_autoencoder.pth
│   ├── 200_Cedi_note_autoencoder.pth
│   └── thresholds.json              # Anomaly thresholds per denomination
├── data/
│   └── genuine/
│       ├── 1_Cedi_coin/             # Training images
│       ├── 1_Cedi_note/
│       ├── 2_Cedi_coin/
│       ├── ... (other denominations)
│       └── 200_Cedi_note/
├── testImages/                      # Images for manual testing
└── README.md                        # This file
```

## Installation & Setup

### Prerequisites

- Python 3.11.9
- pip or conda package manager
- (Optional) CUDA toolkit for GPU acceleration
- (Optional) For Apple Silicon: Metal Performance Shaders support

### Clone & Install

```bash
# Clone the repository
git clone <repository-url>
cd cediverify/cediverify_backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Getting Started

### Starting the API Server

```bash
# Development mode (auto-reload)
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

**Interactive API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Environment Variables

```bash
# Optional: Set API key for production (security)
export API_KEY="your-secret-key"

# Optional: Set port (default: 8000)
export PORT=8000
```

## API Endpoints

### Health Check

**GET** `/health`

Check if the API server is running and healthy.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

### Currency Verification

**POST** `/verify`

Verify the authenticity of a currency image.

**Headers:**
```
Content-Type: multipart/form-data
X-Api-Key: your-api-key (optional, if API_KEY is set)
```

**Parameters:**
- `file` (required): Image file (JPEG, PNG, or WebP, max 10MB)

**Request (cURL):**
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "X-Api-Key: your-api-key" \
  -F "file=@currency_image.jpg"
```

**Request (Python):**
```python
import requests

files = {'file': open('currency_image.jpg', 'rb')}
headers = {'X-Api-Key': 'your-api-key'}

response = requests.post(
    'http://localhost:8000/verify',
    files=files,
    headers=headers
)

print(response.json())
```

**Success Response (200):**
```json
{
  "denomination": "100_Cedi_note",
  "is_genuine": true,
  "confidence": 0.92,
  "reconstruction_error": 0.0145,
  "threshold": 0.0234,
  "stage_1": {
    "top_class": "100_Cedi_note",
    "top_confidence": 0.96
  },
  "stage_2": {
    "passed": true,
    "error": 0.0145
  }
}
```

**Error Responses:**
- `400` - Invalid file type or file too large
- `401` - Invalid or missing API key
- `500` - Server error

## How It Works

### Two-Stage Verification Pipeline

```
Input Image
    ↓
Stage 1: Denomination Classifier
├─ MobileNetV2 neural network
├─ Identifies currency denomination
└─ Filters out low-confidence predictions
    ↓
Stage 2: Per-Denomination Anomaly Detection
├─ Load denomination-specific autoencoder
├─ Compare original image with reconstruction
├─ Calculate reconstruction error
└─ Compare against learned anomaly threshold
    ↓
Result: GENUINE / COUNTERFEIT + Confidence Score
```

### Architecture Details

#### Stage 1: Denomination Classifier

**Model:** Fine-tuned MobileNetV2  
**Training:** Transfer learning on genuine Cedis images  
**Input:** 224×224 RGB image  
**Output:** Probability distribution over 13 denominations  
**Normalization:** ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  

**Class Names:** See [class_names.txt](class_names.txt)

```python
# Stage 1 in code
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
])
```

#### Stage 2: Per-Denomination Autoencoders

**Model:** Convolutional Autoencoder (CNN-based)  
**Architecture:**
- **Encoder:** 3 convolutional blocks (3→32→64→128 channels)
- **Decoder:** 3 transposed convolution blocks (128→64→32→3 channels)
- **Activation:** ReLU in encoder, Sigmoid in decoder output

**Training:** Only on genuine currency images (85% train, 15% validation)  
**Detection Method:** Reconstruction error - Counterfeits don't match learned patterns  
**Input:** 224×224 RGB image  
**Output:** Reconstructed image (same size)  
**Normalization:** No normalization (decoder outputs [0,1])  

```python
# Encoder: 224 → 112 → 56 → 28
Conv2d(3, 32, stride=2) → Conv2d(32, 64, stride=2) → Conv2d(64, 128, stride=2)

# Decoder: 28 → 56 → 112 → 224
ConvTranspose2d(128, 64, stride=2) → ConvTranspose2d(64, 32, stride=2) → ConvTranspose2d(32, 3, stride=2)
```

### Supported Denominations

**Coins (5):**
- 10 Pesewas
- 20 Pesewas
- 50 Pesewas
- 1 Cedi
- 2 Cedi

**Notes (8):**
- 1 Cedi
- 2 Cedi
- 5 Cedi
- 10 Cedi
- 20 Cedi
- 50 Cedi
- 100 Cedi
- 200 Cedi

**Total: 13 denominations**

### Verification Thresholds

| Metric | Value | Purpose |
|--------|-------|---------|
| DENOMINATION_CONFIDENCE_FLOOR | 0.85 | Min confidence to trust Stage 1 result |
| Autoencoder Threshold | Per-denomination | 3-sigma cutoff on reconstruction error |

**Verification Logic:**
1. Stage 1 classifies denomination (confidence must be ≥ 0.85)
2. If confidence < 0.85: Return "Not Recognized"
3. Load Stage 2 autoencoder for identified denomination
4. Calculate reconstruction error
5. If error < threshold: **GENUINE**
6. If error ≥ threshold: **COUNTERFEIT**

## Training

### Training New Models

#### 1. Prepare Dataset

Organize genuine currency images in folder structure:

```
data/genuine/
├── 1_Cedi_coin/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 1_Cedi_note/
│   └── ...
└── ... (other denominations)
```

#### 2. Train Denomination Classifier

```bash
python train_denomination.py
```

**Output:**
- `denomination_classifier.pth` - Trained classifier weights
- Console output - Training progress and accuracy

**Configuration (edit in script):**
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-4
- Train/val split: 80/20

#### 3. Train Autoencoders

```bash
python train_per_denomination.py
```

**Output:**
- `models/<denomination>_autoencoder.pth` - Per-denomination models
- `models/thresholds.json` - Anomaly thresholds

**Configuration (edit in script):**
- Epochs: 30
- Batch size: 16
- Learning rate: 1e-3
- Train/val split: 85/15
- Anomaly threshold: mean + 3σ

**Alternative (train single generic model):**
```bash
python train.py
```

## Model Details

### Denomination Classifier

- **Type:** MobileNetV2 (transfer learning)
- **Size:** ~3-4 MB
- **Inference:** ~50-100ms per image
- **File:** [denomination_classifier.pth](denomination_classifier.pth)

### Autoencoders (per denomination)

- **Type:** Convolutional Autoencoder
- **Size:** ~1-2 MB each (13 total ≈ 20-26 MB)
- **Inference:** ~100-150ms per image
- **Files:** `models/*_autoencoder.pth`
- **Total models:** 13 (one per denomination)

### Configuration

- **Thresholds:** `models/thresholds.json`
- **Class names:** `class_names.txt`

## Performance

### Inference Speed

| Operation | GPU | CPU | Notes |
|-----------|-----|-----|-------|
| Preload all models | 2-5s | 10-30s | First startup |
| Stage 1 (classify) | 50ms | 200-500ms | Per image |
| Stage 2 (autoencoder) | 100ms | 300-800ms | Per image |
| Full verification | ~200ms | ~1s | Both stages |

### Accuracy Metrics

Depends on:
- Quality and size of training dataset
- Diversity of genuine images
- Authenticity threshold tuning
- Image quality at inference

## Deployment

### Local Deployment

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker

```dockerfile
FROM python:3.11.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV API_KEY=your-secret-key
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t cediverify-backend .
docker run -p 8000:8000 -e API_KEY=your-secret-key cediverify-backend
```

### Cloud Deployment

**Environment Variables to Set:**
- `API_KEY` - Secret key for API authentication
- `PORT` - Port to listen on (default: 8000)

**Heroku:**
```bash
heroku create cediverify-api
git push heroku main
heroku config:set API_KEY=your-secret-key
```

**Google Cloud Run / AWS Lambda:**
- Deploy as container
- Set environment variables in cloud console
- Configure CORS as needed

## Security Considerations

### Production Checklist

- [ ] Set unique `API_KEY` environment variable
- [ ] Restrict CORS to your frontend domain
- [ ] Use HTTPS in production
- [ ] Validate file uploads (size, type)
- [ ] Implement rate limiting
- [ ] Monitor model inference performance
- [ ] Add logging for all requests
- [ ] Use secrets management (not hardcoded keys)
- [ ] Regular security updates for dependencies

### API Key Setup

```python
# In api.py - CORS restrictions for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Api-Key"],
)
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size in training scripts
- Use CPU instead: Set `device = 'cpu'`
- Restart server to clear cache

### Issue: Slow first inference

**Expected behavior** - First request triggers model loading:
- Models load on first use (5-10 seconds)
- Use `preload_all()` at startup to avoid per-request latency
- Subsequent requests use cached models (<200ms)

### Issue: Low accuracy

**Solutions:**
1. Check image quality (clear, well-lit photos)
2. Verify training data covers all angles
3. Tune `DENOMINATION_CONFIDENCE_FLOOR` threshold
4. Increase training epochs in training scripts
5. Add more diverse training images

### Issue: "Model file not found"

**Solution:**
- Verify folder structure matches exactly
- Check file paths in `thresholds.json` are correct
- Files must be in `models/` subdirectory

### Issue: GPU not detected

**Verify:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Fix:**
- Reinstall PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Or use MPS (Apple Silicon): Works automatically on compatible systems

## Dependencies

See [requirements.txt](requirements.txt) for complete list:
- PyTorch - Deep learning
- torchvision - Computer vision utilities
- FastAPI - Web framework
- Uvicorn - ASGI server
- Pillow - Image processing
- python-multipart - File handling

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contributing Guidelines

- Keep code clean and well-documented
- Add tests for new features
- Update this README if adding features
- Use consistent code style
- Test locally before submitting PR

## API Response Schema

### Successful Verification Response

```json
{
  "denomination": "string (e.g., '100_Cedi_note')",
  "is_genuine": "boolean",
  "confidence": "float (0.0 to 1.0)",
  "reconstruction_error": "float",
  "threshold": "float",
  "stage_1": {
    "top_class": "string",
    "top_confidence": "float",
    "all_probabilities": {
      "denomination_name": "float"
    }
  },
  "stage_2": {
    "passed": "boolean",
    "error": "float"
  }
}
```

## License

[Add your license information here]

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check troubleshooting section above

## Acknowledgments

- PyTorch team for deep learning framework
- FastAPI for modern Python web framework
- Torchvision for pre-trained models
- Contributors and testers

---

**Version:** 1.0.0  
**Python Version:** 3.11.9  
**Status:** Production Ready  
**Last Updated:** August 2026
