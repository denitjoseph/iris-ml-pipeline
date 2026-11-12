# 🌸 Iris ML Pipeline — End-to-End MLOps with MLflow & FastAPI

A production-style ML pipeline that trains a Random Forest classifier on the Iris dataset, tracks experiments with **MLflow**, registers the model in the **MLflow Model Registry**, and serves predictions via a **FastAPI** REST API.

---

## 📁 Project Structure

```
├── train.py       # Train, log, and register the model with MLflow
├── app.py         # FastAPI inference server
├── mlruns/        # MLflow tracking data (auto-generated)
└── README.md
```

---

## ⚙️ Requirements

Install dependencies via pip:

```bash
pip install fastapi uvicorn mlflow scikit-learn pandas pydantic
```

---

## 🚀 Quick Start

### Step 1 — Train & Register the Model

```bash
python train.py
```

This will:
- Train a `RandomForestClassifier` on the Iris dataset
- Log parameters (`n_estimators`, `max_depth`) and metrics (`accuracy`) to MLflow
- Register the model as **`IrisRandomForest`** in the MLflow Model Registry

**Expected output:**
```
Model logged with Run ID: <run_id>
Test Accuracy: 1.0000
```

---

### Step 2 — Launch the MLflow UI *(optional)*

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore runs, metrics, and registered models.

---

### Step 3 — Start the Inference Server

```bash
uvicorn app:app --reload --port 8000
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

### Step 4 — Send a Prediction Request

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' |
  Select-Object -ExpandProperty Content
```

**Bash / Linux / macOS:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Expected Response:**
```json
{"class_id": 0}
```

---

## 📡 API Reference

### `POST /predict`

Accepts Iris flower measurements and returns the predicted class.

**Request Body:**
| Field          | Type  | Description          |
|----------------|-------|----------------------|
| `sepal_length` | float | Sepal length in cm   |
| `sepal_width`  | float | Sepal width in cm    |
| `petal_length` | float | Petal length in cm   |
| `petal_width`  | float | Petal width in cm    |

**Response:**
| Field      | Type | Description                              |
|------------|------|------------------------------------------|
| `class_id` | int  | Predicted class (0 = Setosa, 1 = Versicolor, 2 = Virginica) |

**Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) *(Swagger UI)*

---

## 🧠 Model Details

| Parameter      | Value               |
|----------------|---------------------|
| Algorithm      | Random Forest       |
| `n_estimators` | 50                  |
| `max_depth`    | 3                   |
| Dataset        | Iris (sklearn)      |
| Train/Test     | 80% / 20% split     |
| Registry Name  | `IrisRandomForest`  |

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `WinError 10013` — port already in use | Run `netstat -ano \| findstr :8000` to find the PID, then `taskkill /PID <pid> /F` |
| `UntrustedTypesFoundException` | Add `skops_trusted_types=["sklearn.tree._tree.Tree"]` to `log_model()` |
| `models:/IrisRandomForest/latest` not found | Use a version alias like `/1` or `@champion` in newer MLflow versions |
| PowerShell `curl` syntax error | Use `Invoke-WebRequest` instead — PowerShell's `curl` is not the same as bash `curl` |

