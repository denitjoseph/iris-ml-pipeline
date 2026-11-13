
import mlflow
from mlflow.tracking import MlflowClient

# Connect to MLflow
client = MlflowClient()

# Get details of the latest registered model version
model_version = client.get_model_version_by_alias(
    "IrisRandomForest",
    "latest"
)

print(f"Model Name: {model_version.name}")
print(f"Model Version: {model_version.version}")
print(f"Source Run ID: {model_version.run_id}")
print(f"Artifact URI: {model_version.source}")
print(f"Model Status: {model_version.status}")
