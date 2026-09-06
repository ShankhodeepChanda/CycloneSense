# 🌪️ CycloneSense

**AI-Powered Tropical Cyclone Detection, Classification & Forecasting Platform**

<p align="center">
  <em>Deep Learning • Computer Vision • Remote Sensing • Disaster Management</em>
</p>

---

## 🎯 Overview

CycloneSense is an **AI-powered scientific platform** for automated tropical cyclone analysis using satellite imagery. Built with PyTorch deep learning models and real-time satellite data processing, it provides:

- **🔍 Automated Detection** — Classical morphological algorithms with quality filters locate cyclone structures in satellite imagery
- **🧠 Pattern Classification** — ResNet-based CNN identifies 7 Dvorak morphological patterns (clear, developing, curved band, CDO, eye, sheared, dissipating)
- **📈 Track & Intensity Forecasting** — Multi-task LSTM predicts next position, wind speed, and IMD intensity category from historical observations
- **🎨 Mission-Control Interface** — Glassmorphic HUD with real-time telemetry, canvas-based detection overlay, and live metrics

The platform combines **Computer Vision, Time-Series Forecasting, and Remote Sensing** into a production-grade web application with FastAPI backend and React frontend.

---

## 🚀 Key Features

### Detection Pipeline
- Morphological cyclone detection using `scipy.ndimage`
- 5-stage quality filter (variance, saturation, area bounds, aspect ratio, contrast)
- Bounding box localization with centroid calculation
- Dual-channel support (IR thermal + VIS visible light)

### Pattern Classification
- **7 Dvorak Morphology Classes**: Based on operational tropical cyclone analysis
  - `clear` — No cyclonic structure
  - `developing` — Early-stage disturbance
  - `curved_band` — Organized curved cloud bands
  - `central_dense_overcast` — Dense central convection (CDO)
  - `eye` — Mature eye structure
  - `sheared` — Wind-sheared/asymmetric
  - `dissipating` — Weakening system

### Forecasting System
- **Multi-task LSTM** (2 layers, 64 hidden units)
- Sequence length: 4 timesteps
- Outputs: Next lat/lon, predicted wind speed, intensity class, confidence
- **4 IMD Intensity Classes**: depression, tropical_storm, severe_cyclonic_storm, very_severe_cyclonic_storm

### Mission-Control Interface
- Deep space cockpit aesthetic with Fira Code monospace typography
- Glassmorphism design: `backdrop-filter: blur(16px)` with translucent cards
- Large central viewport with HTML5 canvas detection overlay
- Real-time telemetry sidebar with pattern severity, coordinates, metrics
- Historical storm samples: AMPHAN (2020), FANI (2019), BIPARJOY (2023)
- Responsive grid layout with glowing cyan/emerald accents

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** — Modern async Python web framework
- **PyTorch** — Deep learning models (CNN + LSTM)
- **Pydantic** — Request/response validation
- **Pillow** — Image preprocessing
- **NumPy + SciPy** — Numerical computation and morphological processing
- **Uvicorn** — ASGI server

### Frontend
- **React 19** — Latest React with concurrent features
- **Vite 6** — Next-generation build tool
- **Lucide React** — Icon system
- **Vanilla CSS** — Custom glassmorphic design system
- **HTML5 Canvas** — Real-time detection overlay rendering

### Data Sources
- **NASA MODIS Terra** — IR Band 31 (thermal) + TrueColor VIS (visible)
- **NOAA IBTrACS** — Best-track historical cyclone data
- Real satellite imagery from North Indian Ocean cyclones (2019-2024)

### ML Architecture
- **CyclonePatternCNN** — ResNet-style with Grad-CAM hooks (128×128 input)
- **CycloneTrackLSTM** — Multi-task sequence forecasting (8 features)
- Trained on real MODIS satellite data with data augmentation
- Model checkpoints: `models/cyclone_model.pt`, `models/forecast_model.pt`

### DevOps
- **Docker** + **Docker Compose** — Containerization
- **GitHub Actions** — CI/CD pipeline (lint, test, build, deploy)
- **GitHub Container Registry** — Docker image hosting
- **Pytest** — Automated testing
- **Black** — Code formatting

---

## 📁 Project Structure

```
CycloneSense/
├── app/                          # FastAPI backend
│   ├── main.py                   # REST API endpoints
│   ├── model.py                  # Model inference pipeline
│   └── schemas.py                # Pydantic request/response models
│
├── frontend/                     # React mission-control UI
│   ├── src/
│   │   ├── main.jsx              # Main application component
│   │   └── styles.css            # Glassmorphic design system
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── ml/                           # Machine learning
│   ├── models.py                 # PyTorch CNN & LSTM architectures
│   ├── detection.py              # Morphological detection algorithm
│   ├── train.py                  # Training pipeline
│   └── requirements-ml.txt
│
├── data/
│   ├── real_satellite/           # NASA MODIS imagery (18 cyclones)
│   └── processed/                # Augmented training data
│
├── models/                       # Trained model checkpoints
│   ├── cyclone_model.pt          # CyclonePatternCNN weights
│   └── forecast_model.pt         # CycloneTrackLSTM weights
│
├── tests/                        # Automated test suite
│   └── test_api.py
│
├── docs/                         # Technical documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── MODEL.md
│   ├── DEPLOYMENT.md
│   └── DEMO.md
│
├── .github/workflows/
│   └── ci-cd.yml                 # CI/CD pipeline
│
├── Dockerfile                    # Production container
├── docker-compose.yml            # Development stack
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/CycloneSense.git
cd CycloneSense
```

### 2. Backend Setup

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Start Backend:**
```bash
uvicorn app.main:app --reload
```

API runs at: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

### 4. Test the Application

**Option A: Use Historical Samples**
- Click "AMPHAN Peak", "FANI Eye", or "BIPARJOY" in the interface
- The system loads real NASA MODIS imagery from the data directory

**Option B: Upload Custom Image**
- Click "UPLOAD SATELLITE IMAGE"
- Select any satellite image (PNG, JPG, WebP)
- Choose channel: IR (thermal) or VIS (visible)

---

## 🐳 Docker Deployment

### Build and Run

```bash
docker build -t cyclonesense .
docker run -p 8000:8000 cyclonesense
```

### Docker Compose (Full Stack)

```bash
docker compose up --build
```

Services:
- **Backend**: `http://localhost:8000`
- **Frontend**: `http://localhost:3000`
- **API Docs**: `http://localhost:8000/docs`

---

## 🔌 API Reference

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "cyclonesense-api"
}
```

### Get Dvorak Patterns
```http
GET /patterns
```

**Response:**
```json
{
  "patterns": [
    "clear", "developing", "curved_band", 
    "central_dense_overcast", "eye", "sheared", "dissipating"
  ]
}
```

### Detect and Classify
```http
POST /predict/detect_and_classify
```

**Parameters:**
- `file` (form-data): Satellite image file
- `channel` (query): `"ir"` or `"vis"` (default: `"ir"`)

**Response:**
```json
{
  "detection": {
    "bbox": [120, 85, 180, 145],
    "centroid": [150, 115],
    "area": 3600,
    "confidence": 0.87,
    "channel": "ir",
    "method": "morphological-detection"
  },
  "classification": {
    "pattern": "eye",
    "confidence": 0.9234,
    "model": "cyclone-pattern-cnn-pytorch"
  },
  "is_cyclone": true,
  "message": "Tropical cyclone detected: eye",
  "disclaimer": "Prototype result; not an official warning."
}
```

### Pattern Classification Only
```http
POST /predict/pattern
```

**Parameters:**
- `file` (form-data): Satellite image file

**Response:**
```json
{
  "pattern": "central_dense_overcast",
  "confidence": 0.8756,
  "model": "cyclone-pattern-cnn-pytorch",
  "disclaimer": "Prototype result; not an official warning."
}
```

### Track & Intensity Forecast
```http
POST /predict/forecast
```

**Request Body:**
```json
{
  "observations": [
    {"lat": 14.2, "lon": 72.1, "wind_kts": 45, "pressure_hpa": 995},
    {"lat": 14.7, "lon": 72.8, "wind_kts": 52, "pressure_hpa": 989},
    {"lat": 15.1, "lon": 73.5, "wind_kts": 58, "pressure_hpa": 984}
  ]
}
```

**Response:**
```json
{
  "next_lat": 15.5,
  "next_lon": 74.2,
  "predicted_wind_kts": 64.0,
  "intensity_class": "severe_cyclonic_storm",
  "confidence": 0.64
}
```

---

## 🧪 Testing

Run the automated test suite:

```bash
pytest -q
```

**Test Coverage:**
- Health check endpoint
- Pattern list endpoint
- Classification endpoint (with real cyclone image)
- Forecast endpoint (with observation sequence)
- Detection pipeline integration
- Classification pipeline integration

All tests run automatically in CI/CD before deployment.

---

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow:

```yaml
Trigger: Push to main
    ↓
Install Python & Dependencies
    ↓
Run Black Formatter Check
    ↓
Run Pytest Suite
    ↓
Build Docker Image
    ↓
Push to GitHub Container Registry
    ↓
Deploy to Production (Render/Railway)
```

**Pipeline Features:**
- Automated linting and formatting checks
- Full test suite execution
- Docker multi-stage build optimization
- Automated deployment on successful tests
- Environment variable injection
- Health check validation

---

## 📊 Model Details

### CyclonePatternCNN
- **Architecture**: ResNet-style convolutional neural network
- **Input**: 128×128×3 RGB satellite image
- **Output**: 7-class softmax (Dvorak patterns)
- **Features**:
  - Residual connections for gradient flow
  - Batch normalization
  - Grad-CAM hooks for interpretability (currently model-only; API exposure planned)
  - Data augmentation: rotation, flip, brightness, contrast

### CycloneTrackLSTM
- **Architecture**: 2-layer bidirectional LSTM
- **Input**: Sequence of 4 observations (8 features each: lat, lon, wind, pressure, + deltas)
- **Output**: Multi-task prediction
  - Position regression (lat, lon)
  - Wind speed regression (kts)
  - Intensity classification (4 classes)
- **Features**:
  - Sequence length: 4 timesteps
  - Hidden units: 64
  - Normalization: Z-score standardization
  - Fallback: Kinematic baseline for untrained scenarios

### Training Data
- **18 Real North Indian Ocean Cyclones** (2019-2024):
  - AMPHAN, FANI, BIPARJOY, TAUKTAE, MOCHA, REMAL, DANA, YAAS, NISARGA, GULAB, JAWAD, ASANI, SITRANG, MANDOUS, MICHAUNG, HAMOON, MIDHILI
  - + Clear/dissipating control samples
- **Source**: NASA MODIS Terra (Level 1B and corrected reflectance)
- **Channels**: IR Band 31 (11 μm thermal) + TrueColor VIS
- **Augmentation**: 5× per image (rotation, flip, brightness, contrast, noise)
- **Split**: By storm ID (no data leakage across train/val/test)

---

## 🎨 Interface Design

The mission-control HUD features:

### Visual Design
- **Color Palette**: Deep space background (#080C15) with cyan/emerald/amber/orange accents
- **Typography**: Fira Code monospace for metrics, Inter sans-serif for labels
- **Glassmorphism**: Translucent cards with `backdrop-filter: blur(16px)`
- **Glow Effects**: Cyan box-shadow on hover and active detection overlays

### Layout Components
1. **Command Bar** — Logo, system status, module navigation (DETECTION/FORECAST)
2. **Central Viewport** — Large satellite image display with canvas-based detection overlay
3. **Telemetry Sidebar** — Pattern severity, coordinates, confidence, area, method
4. **Sample Loader** — Quick access to historical cyclone imagery
5. **File Upload** — Drag-and-drop or click to upload custom satellite images
6. **Forecast Module** — Observation sequence input with JSON editor

### Interaction States
- Loading spinner during API calls
- Error messages with retry options
- Success confirmations with visual feedback
- Empty states with instructional text
- Responsive breakpoints at 1200px and 768px

---

## 🌍 Real-World Applications

CycloneSense can support:

### Disaster Management
- **Early Warning Systems** — Automated scanning of satellite feeds
- **Pattern Recognition** — Consistent morphological classification
- **Track Prediction** — Assist in forecasting cyclone paths
- **Decision Support** — Visual telemetry for meteorological analysts

### Research & Education
- **Climate Studies** — Historical cyclone pattern analysis
- **ML Research** — Benchmark dataset for satellite image classification
- **Student Projects** — End-to-end ML deployment example
- **Remote Sensing** — Practical application of Earth observation data

### Operational Integration
- Compatible with meteorological workflows
- RESTful API for system integration
- Docker deployment for scalability
- Explainable AI for analyst trust (Grad-CAM in model)

---

## ⚠️ Important Disclaimers

1. **Research Prototype** — CycloneSense is a demonstration platform for AI-assisted cyclone analysis
2. **Not an Official Warning System** — Do not use for emergency decision-making
3. **Complement, Not Replace** — Predictions should be validated against official forecasts from:
   - India Meteorological Department (IMD)
   - Joint Typhoon Warning Center (JTWC)
   - Regional Specialized Meteorological Centres (RSMCs)
4. **Model Limitations**:
   - Trained on North Indian Ocean cyclones only
   - Limited to single-channel analysis (IR or VIS, not fused)
   - No ensemble forecasting
   - Grad-CAM visualization not yet exposed via API

---

## 📈 Future Enhancements

Planned improvements for production deployment:

### Data Pipeline
- [ ] Real-time satellite feed ingestion (INSAT-3D, Himawari-9)
- [ ] Multi-channel fusion (IR + VIS + WV)
- [ ] Automated cyclone detection cron jobs
- [ ] Historical cyclone database with PostGIS

### Model Improvements
- [ ] Vision Transformer (ViT) for higher accuracy
- [ ] Temporal Transformer for longer forecast horizons
- [ ] Ensemble forecasting with uncertainty quantification
- [ ] Multi-basin training (Atlantic, Pacific, Indian Ocean)

### Interface Features
- [ ] Interactive map with storm tracks
- [ ] Probability cone visualization
- [ ] Time-series animation of cyclone evolution
- [ ] Comparison with official forecasts
- [ ] Mobile-responsive PWA

### Infrastructure
- [ ] Kubernetes deployment
- [ ] Model monitoring with MLflow
- [ ] A/B testing framework
- [ ] Edge inference for low-latency
- [ ] GPU acceleration for batch processing

### Explainability
- [ ] Expose Grad-CAM via `/explain` endpoint
- [ ] Feature importance analysis
- [ ] Uncertainty estimation
- [ ] Attention visualization for LSTM

---


✅ **Polished UI** — Mission-control glassmorphic design, not a basic form  

**Technical Depth:**
- Classical detection algorithm with 5-stage quality filter
- Multi-task LSTM with position + intensity + classification
- Canvas-based real-time detection overlay rendering
- Comprehensive test suite with CI/CD automation
- Proper data split by storm ID to prevent leakage

**Documentation:**
- README with full API reference
- Architecture, dataset, model, deployment docs
- Code comments and type hints
- Swagger/OpenAPI interactive docs

---

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m "feat: add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

**Development Guidelines:**
- Follow Black code formatting
- Write tests for new features
- Update documentation
- Maintain type hints

---

## 👥 Team

**CycloneSense Development Team**

*Building the future of AI-assisted disaster management*

---

## 🌟 Acknowledgments

- **NASA Earth Observations** — MODIS satellite imagery
- **NOAA IBTrACS** — Historical cyclone best-track data
- **India Meteorological Department** — Dvorak classification standards
- **PyTorch Community** — Deep learning framework
- **FastAPI** — Modern Python web framework

---


---

<p align="center">
  <strong>🌪️ CycloneSense — AI-Powered Cyclone Intelligence</strong><br>
  <em>Computer Vision × Remote Sensing × Deep Learning × Disaster Management</em>
</p>

<p align="center">
  Made with ❤️ for safer communities
</p>
