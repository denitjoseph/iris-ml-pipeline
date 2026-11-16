# 🌸 Iris ML Pipeline — End-to-End MLOps with MLflow, FastAPI & Evidently AI

A production-grade ML pipeline that trains a Random Forest classifier on the Iris dataset, tracks experiments and registers versions using **MLflow**, serves predictions via a **FastAPI** REST API (containerized with **Docker**), and implements **Phase 2: Continuous Training & Data Drift Monitoring** using **Evidently AI**.

---

## 📁 Project Structure

```
├── mlruns/                  # MLflow tracking data directory
├── src/
│   ├── monitor_drift.py     # Evidently AI data drift detection script
│   └── retrain_pipeline.py  # Automated retraining trigger on drift
├── app.py                   # FastAPI prediction server
├── train.py                 # Initial model training & MLflow logging
├── Dockerfile               # Container build configuration
├── requirements.txt         # Project dependencies manifest
├── drift_report.html        # Generated visual drift dashboard
└── README.md
```

---

## ⚙️ Requirements & Installation

Activate your virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Initial Model Training & Registry

```bash
python train.py
```

This will:
- Train a `RandomForestClassifier` on the Iris dataset.
- Log parameters and evaluation metrics to MLflow.
- Register the model under **`IrisRandomForest`** in the MLflow Model Registry.

---

### Step 2 — Phase 2: Data Drift Monitoring & Automated Retraining

Run the automated drift-detection and retraining pipeline:

```bash
python src/retrain_pipeline.py
```

What this does:
1. **Drift Detection (`src/monitor_drift.py`):** Uses Evidently AI to compare reference training data against current production samples.
2. **Visual Dashboard:** Generates `drift_report.html` in the root folder.
3. **Automated Retraining:** If data drift is detected, it automatically retrains the model with updated parameters, logs the new experiment run, and registers the new version in the MLflow Model Registry.

To view the generated visual drift report in your browser:

```powershell
# Windows
start drift_report.html

# macOS / Linux
open drift_report.html
```

---

### Step 3 — Launch the MLflow UI *(Optional)*

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view runs, parameters, metrics, and registered model versions.

---

### Step 4 — Run the Inference Server

**Option A — Run locally with Uvicorn:**
```bash
uvicorn app:app --reload --port 8000
```

**Option B — Run with Docker:**
```bash
# Build Docker image
docker build -t iris-api .

# Run Docker container
docker run -d --name iris-api -p 8000:8000 iris-api

# Check container logs
docker logs iris-api
```

The API will be live at [http://localhost:8000](http://localhost:8000).

---

### Step 5 — Send Prediction Requests

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Expected Response:**
```json
{"class_id": 0}
```

Interactive Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🛠️ Troubleshooting

| Issue | Resolution |
|---|---|
| `ImportError: cannot import name 'infer_dtype_from_object'` | Version clash between Pandas 2.2+/3.x and older Evidently. Fixed via compatibility layer in `src/monitor_drift.py`. |
| Port 8000 already bound (`WinError 10013`) | Run `netstat -ano \| findstr :8000` to find the PID, then `taskkill /PID <PID> /F`. |
| Docker registry not found | Ensure `mlflow.db` is present and copied during build, or load directly from `mlruns/` artifact folder. |
