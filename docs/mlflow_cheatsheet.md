# MLflow Cheatsheet

Quick reference for the workshop. Keep this open while working through the notebooks!

---

## Setup & Connection

```python
import mlflow
import mlflow.sklearn

# Connect to tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create/select experiment
mlflow.set_experiment("my-experiment")
```

---

## Basic Run Structure

```python
with mlflow.start_run(run_name="my-run"):
    # Your training code here
    mlflow.log_param("key", value)
    mlflow.log_metric("accuracy", 0.95)
```

---

## Logging Parameters

```python
# Single parameter
mlflow.log_param("learning_rate", 0.01)

# Multiple parameters (recommended)
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.01
})
```

---

## Logging Metrics

```python
# Single metric
mlflow.log_metric("accuracy", 0.95)

# Multiple metrics (recommended)
mlflow.log_metrics({
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.89,
    "f1": 0.90
})

# Metric with step (for training curves)
mlflow.log_metric("loss", 0.5, step=1)
mlflow.log_metric("loss", 0.3, step=2)
```

---

## Logging Models

```python
# Sklearn model
mlflow.sklearn.log_model(model, name="model")

# Load model later
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Or from registry
loaded_model = mlflow.sklearn.load_model("models:/model-name/latest")
```

---

## Logging Artifacts

### Figures (matplotlib)
```python
fig, ax = plt.subplots()
ax.plot(data)
mlflow.log_figure(fig, "plot.png")
plt.close()
```

### Files (pickles, CSVs, etc.)
```python
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "data.csv")
    df.to_csv(path)
    mlflow.log_artifact(path)  # Root of artifacts
    # OR
    mlflow.log_artifact(path, artifact_path="data")  # In subfolder
```

---

## Tags

```python
# Single tag
mlflow.set_tag("author", "your-name")

# Multiple tags
mlflow.set_tags({
    "author": "your-name",
    "model_type": "RandomForest",
    "stage": "experimentation"
})
```

---

## Searching Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("my-experiment")

# Find best runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.f1 DESC"],
    max_results=10
)

for run in runs:
    print(f"{run.info.run_name}: {run.data.metrics}")
```

---

## Model Registry

```python
# Register a model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="my-model"
)

# Load from registry
model = mlflow.sklearn.load_model("models:/my-model/latest")
# OR specific version
model = mlflow.sklearn.load_model("models:/my-model/1")
```

---

## Autologging (Alternative to Manual)

```python
# Enable autolog for sklearn
mlflow.sklearn.autolog()

# Now just train - MLflow logs everything automatically!
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Disable when done
mlflow.sklearn.autolog(disable=True)
```

**Note**: Autolog metrics are on TRAINING data. Add test metrics manually for fair comparison.

---

## Common Patterns

### Full Training Run
```python
with mlflow.start_run(run_name="full-example"):
    # Tags
    mlflow.set_tag("author", "workshop")

    # Parameters
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

    # Model
    mlflow.sklearn.log_model(model, name="model")

    # Plot
    fig, ax = plt.subplots()
    # ... create plot ...
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()
```

### Save Preprocessing (Scaler)
```python
import joblib
import tempfile
import os

with mlflow.start_run():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scaler.pkl")
        joblib.dump(scaler, path)
        mlflow.log_artifact(path, artifact_path="preprocessing")

    mlflow.set_tag("requires_scaling", "true")
```

### Load Preprocessing Later
```python
run_id = "your-run-id"
artifact_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
scaler = joblib.load(artifact_path)
```

---

## Quick Debugging

```python
# Current tracking URI
print(mlflow.get_tracking_uri())

# Current experiment
print(mlflow.get_experiment_by_name("name"))

# List artifacts for a run
artifacts = client.list_artifacts(run_id)
for a in artifacts:
    print(a.path)
```

---

## UI Access

- **MLflow UI**: http://localhost:5000
- **Experiments**: Left sidebar
- **Runs**: Click experiment name
- **Artifacts**: Click run, then "Artifacts" tab
- **Compare**: Select runs with checkboxes, click "Compare"

---

## Key Concepts Summary

| Concept | What it is | Example |
|---------|-----------|---------|
| **Experiment** | Group of related runs | "churn-prediction" |
| **Run** | Single execution | "rf-100-trees" |
| **Parameter** | Input to model | n_estimators=100 |
| **Metric** | Output/result | accuracy=0.95 |
| **Artifact** | File (model, plot) | model.pkl |
| **Tag** | Metadata | author="me" |
| **Registry** | Model versioning | "prod-model" v1, v2 |
| **Projects** | Reproducible code | MLproject file |

---

## LLM/GenAI Support (Bonus)

MLflow now supports LLM applications:

```python
# Trace LLM calls
@mlflow.trace
def chat_with_llm(prompt):
    response = openai_client.chat(prompt)
    return response

# Log LLM model
mlflow.openai.log_model(...)

# Evaluate LLM outputs
mlflow.evaluate(model, data, model_type="text")
```

**Use cases**: RAG pipelines, chatbot debugging, agent monitoring
