# 🌸 Iris ML Pipeline — End-to-End MLOps with MLflow, FastAPI & Evidently AI

A complete production-grade MLOps pipeline implementing continuous training, automated data drift monitoring with **Evidently AI**, model tracking and registry with **MLflow**, serving via a **FastAPI** REST API, automated unit testing with **pytest**, containerization with **Docker** (pushed to **Docker Hub**), and CI/CD automation with **GitHub Actions**.

---

## 📁 Project Structure

```
├── .github/
│   └── workflows/
│       └── deploy.yml        # CI/CD GitHub Actions workflow
├── mlruns/                   # MLflow tracking directory
├── src/
│   ├── monitor_drift.py      # Evidently AI data drift detection
│   └── retrain_pipeline.py   # Continuous retraining & registry trigger
├── tests/
│   ├── test_app.py           # FastAPI endpoint tests (health, predict)
│   └── test_retrain.py       # Drift check & retraining execution tests
├── app.py                    # FastAPI serving layer (dynamic registry load)
├── train.py                  # Baseline training & MLflow logging
├── Dockerfile                # Docker container definition
├── requirements.txt          # Python dependencies
├── drift_report.html         # Generated Evidently visual dashboard
└── README.md
```

---

## ⚙️ Requirements & Installation

Activate your virtual environment and install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 🧪 Running Unit & Integration Tests

Run the full test suite with `pytest`:

```powershell
pytest tests/
```

Test coverage includes:
- **`test_app.py`**: Validates `/health` check status and model availability, `/predict` valid inputs, and Pydantic schema rejection (`422 Unprocessable Entity`).
- **`test_retrain.py`**: Validates drift detection boolean interface, automated retraining logging to MLflow, and accuracy quality gate ($\ge 0.90$).

---

## 🚀 Step-by-Step Workflow

### Step 1 — Baseline Model Training & Registry

```powershell
python train.py
```
Trains a `RandomForestClassifier` baseline on the Iris dataset and registers the model in the MLflow Model Registry.

---

### Step 2 — Continuous Retraining on Data Drift (Phase 3)

```powershell
python src/retrain_pipeline.py
```

* **Workflow:**
  1. Checks production data against reference baseline for drift using Evidently AI.
  2. Generates visual dashboard: `drift_report.html`.
  3. If drift is detected, triggers automated retraining with candidate hyperparameters.
  4. Applies quality gate ($\ge 90\%$ accuracy) before registering the new candidate model under **`IrisRandomForestModel`**.

To view the generated visual drift report in your browser:

```powershell
# Windows
start drift_report.html

# macOS / Linux
open drift_report.html
```

---

### Step 3 — Inspect MLflow Dashboard

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to view experiment runs, metrics, and registered model versions.

---

### Step 4 — Run Local Serving API

Start the FastAPI application:

```powershell
uvicorn app:app --reload --port 8000
```

#### Test Endpoints:

* **Health Check (`GET /health`):**
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
  ```
  *Response:* `{"status": "healthy", "model_loaded": true}`

* **Inference (`POST /predict`):**
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
  ```
  *Response:* `{"prediction": 0}`

Interactive Swagger documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🐳 Docker Deployment & Docker Hub

The containerized image is hosted on Docker Hub: **`denitjoseph/iris-mlops-api`**

### Pull and Run from Docker Hub:
```powershell
# Pull latest image
docker pull denitjoseph/iris-mlops-api:latest

# Run container
docker run -d -p 8000:8000 --name iris-api denitjoseph/iris-mlops-api:latest

# Send test prediction
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Stop and remove container
docker stop iris-api; docker rm iris-api
```

---

## 🔄 CI/CD Automation (GitHub Actions)

Defined in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml):
- **Triggers:** Push to `master` and manual dispatch (`workflow_dispatch`).
- **Pipeline:**
  1. Checks out repository code.
  2. Sets up Python 3.10 and installs dependencies.
  3. Executes test suite (`pytest tests/`).
  4. Builds Docker image.
  5. *(Optional)* SSH deployment step to remote Linux VM.
