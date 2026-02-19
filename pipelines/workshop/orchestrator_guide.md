# Orchestrating ML Workflows: A Practical Guide

## Table of Contents

1. [Why Orchestrators?](#1-why-orchestrators)
2. [The Three Orchestrators](#2-the-three-orchestrators)
3. [Orchestration Patterns for ML](#3-orchestration-patterns-for-ml)
4. [Where MLflow Fits](#4-where-mlflow-fits)
5. [Workshop Structure](#5-workshop-structure)
6. [Running the Workshops](#6-running-the-workshops)
7. [Quick Reference](#7-quick-reference)

---

## 1. Why Orchestrators?

### The Problem with Notebooks

You've built a great ML model in a notebook. Now what?

```
Your current workflow:
────────────────────────────────────────────────────────────────
Monday:     Run notebook manually, model trained
Tuesday:    Forgot to run it
Wednesday:  "Why is the model stale?"
Thursday:   Run it, API fails at cell 5, re-run everything
Friday:     Manager: "Why isn't this automated?"
```

### What Orchestrators Solve

| ML Problem | Orchestration Solution |
|------------|------------------------|
| API fails randomly | **Retries** with backoff |
| Feature engineering is slow | **Caching** - skip if unchanged |
| Comparing 5 models takes 5x time | **Parallel execution** |
| Different hyperparameters = code changes | **Parameters** - configure at runtime |
| Training and inference share code | **Subflows** - compose pipelines |
| No visibility into pipeline state | **Monitoring** - logs, UI, alerts |

### The Orchestrator's Job

```
                    ORCHESTRATOR
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐  │
    │   │ Load │ →  │ Feat │ →  │Train │ →  │ Reg  │  │
    │   │ Data │    │ Eng  │    │Model │    │ Model│  │
    │   └──────┘    └──────┘    └──────┘    └──────┘  │
    │      │           │           │           │      │
    │   Retry?      Cache?      Track?      Version?  │
    │   3 times     1 hour      MLflow      Registry  │
    │                                                  │
    └──────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         Prefect:                MLflow:
         - Retries               - Experiments
         - Caching               - Metrics
         - Scheduling            - Model versions
         - Monitoring            - Artifacts
```

**Key insight**: Orchestrators handle HOW your pipeline runs. MLflow handles WHAT gets tracked.

---

## 2. The Three Orchestrators

### Overview

| Tool | Philosophy | Best For |
|------|------------|----------|
| **Airflow** | "Configure everything explicitly" | Enterprise, existing infra |
| **Prefect** | "Just write Python" | Python teams, quick start |
| **Dagster** | "Think about data, not tasks" | Data platforms, lineage |

### Airflow: Industry Standard

- Created by Airbnb (2014)
- Battle-tested at massive scale
- Rich ecosystem (Spark, databases, cloud)

**Pain points for ML:**
- XCom can't handle DataFrames (48KB limit)
- Must save/load files between tasks
- Manual cleanup of temp files
- Heavy configuration before any logic

### Prefect: Pythonic Approach

- Created 2018 as Python-native alternative
- Decorators on regular functions
- Return values = data flow
- Minimal infrastructure

**Key advantage**: Feels like writing normal Python.

### Dagster: Asset-Centric

- Created 2018, focus on "data assets"
- Dependencies inferred from function parameters
- Built-in data lineage graph
- Re-run only what changed

**Key advantage**: Think "what data exists" not "what tasks run".

---

## 3. Orchestration Patterns for ML

These patterns solve real ML problems. Each is covered in the Prefect workshop.

### Pattern 1: Resilience (Retries)

**ML Problem**: Data APIs fail randomly - rate limits, timeouts, network issues.

```python
# Your overnight job fails at 3 AM. You find out at 9 AM.

@task(retries=3, retry_delay_seconds=60)
def load_from_api() -> pd.DataFrame:
    """Automatically retry on failure."""
    return requests.get(API_URL).json()

# Exponential backoff for rate limits
@task(retries=3, retry_delay_seconds=[10, 30, 60])
def load_with_backoff() -> pd.DataFrame:
    """Wait longer between each retry: 10s, 30s, 60s."""
    ...
```

### Pattern 2: Efficiency (Caching)

**ML Problem**: Feature engineering takes 30 minutes. Pipeline fails at training. Now you re-run feature engineering again.

```python
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Skip if same input seen in last hour."""
    # Expensive computation here...
    return df
```

### Pattern 3: Efficiency (Parallelism)

**ML Problem**: Comparing 5 models but running them sequentially = 5x slower.

```python
@flow
def compare_models():
    # Data prep (sequential)
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # Training (parallel - no dependencies between these!)
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)

    # Selection (waits for all above)
    best = select_best([rf_result, gb_result, xgb_result])
    return best
```

### Pattern 4: Flexibility (Parameters)

**ML Problem**: Different hyperparameters = changing code every time.

```python
@flow
def training_flow(
    n_estimators: int = 100,
    max_depth: int = 10,
    experiment_name: str = "default"
):
    """Configure without changing code."""
    ...

# Usage:
training_flow()  # Use defaults
training_flow(n_estimators=200)  # Override one
training_flow(n_estimators=50, max_depth=5)  # Override multiple
```

### Pattern 5: Flexibility (Subflows)

**ML Problem**: Training and inference are separate but share data prep logic.

```python
@flow
def data_preparation() -> pd.DataFrame:
    """Reusable data prep."""
    df = load_data()
    df = engineer_features(df)
    return df

@flow
def training_pipeline():
    df = data_preparation()  # Reuse!
    model = train(df)
    register(model)

@flow
def inference_pipeline():
    df = data_preparation()  # Reuse!
    model = load_from_registry()
    predictions = predict(model, df)
    save(predictions)
```

---

## 4. Where MLflow Fits

### Division of Responsibilities

| Concern | Orchestrator (Prefect) | Tracker (MLflow) |
|---------|------------------------|------------------|
| Retry on failure | ✅ | |
| Cache computations | ✅ | |
| Run tasks in parallel | ✅ | |
| Schedule pipelines | ✅ | |
| Log parameters | | ✅ |
| Log metrics | | ✅ |
| Store model artifacts | | ✅ |
| Version models | | ✅ |
| Serve models | | ✅ |

### Integration Point: The Training Task

```python
@task(retries=2, cache_expiration=timedelta(hours=1))
def train_model(df: pd.DataFrame, n_estimators: int) -> dict:
    """
    Prefect handles:
    - Retrying if training fails
    - Caching if same data/params

    MLflow handles:
    - Tracking the experiment
    - Storing the model
    - Versioning
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        return {"model": model, "run_id": mlflow.active_run().info.run_id}
```

### Best Practices

| Practice | Why |
|----------|-----|
| Separate train/inference flows | Different schedules, different failure modes |
| Load models from registry | `models:/name/latest` not `runs:/id/model` |
| Log `orchestrator` as param | Know which tool ran this experiment |
| Register after validation | Only register models that meet quality bar |

---

## 5. Workshop Structure

### Prefect Workshop (Main Focus)

Located in `prefect/Prefect_Workshop.py`

| Part | Orchestration Pattern | ML Problem Solved |
|------|----------------------|-------------------|
| Part 1 | Tasks & Flows | Notebook chaos → structured pipeline |
| Part 2 | Retries, Backoff | API failures, rate limits |
| Part 3 | Caching, Parallelism | Expensive features, slow model comparison |
| Part 4 | Parameters, Subflows | Hyperparameter tuning, reusable components |
| Part 5 | Full Pipeline | All patterns + MLflow integration |

### Airflow Overview (Guided Reading)

Located in `airflow/airflow_overview.md`

- Why Airflow exists (industry standard)
- Pain points for ML (XCom limits, file I/O)
- Reference: `pipelines/examples/Airflow_ML_Pipeline.py`

### Dagster Workshop (Bonus)

Located in `dagster/Dagster_Workshop.py`

- Transform your Prefect pipeline to Dagster assets
- Learn asset-centric thinking
- Use the Dagster UI for visualization

---

## 6. Running the Workshops

### Prerequisites

```bash
# 1. Start MLflow server
docker-compose up -d

# 2. Install dependencies
pip install prefect dagster dagster-webserver mlflow scikit-learn pandas

# 3. Generate sample data
python generate_sample_data.py
```

### Prefect Workshop

```bash
# Part 1: Tasks & Flows
python pipelines/workshop/prefect/Prefect_Workshop.py part1

# Part 2: Resilience (Retries)
python pipelines/workshop/prefect/Prefect_Workshop.py part2

# Part 3: Efficiency (Caching, Parallel)
python pipelines/workshop/prefect/Prefect_Workshop.py part3

# Part 4: Flexibility (Parameters, Subflows)
python pipelines/workshop/prefect/Prefect_Workshop.py part4

# Part 5: Full Pipeline with MLflow
python pipelines/workshop/prefect/Prefect_Workshop.py full
```

### Dagster Workshop (Bonus)

```bash
# Command line
python pipelines/workshop/dagster/Dagster_Workshop.py full

# With UI (recommended)
dagster dev -f pipelines/workshop/dagster/Dagster_Workshop.py
# Open http://localhost:3000
```

### Reference Implementations

```bash
# Complete Prefect pipeline
python pipelines/examples/Prefect_ML_Pipeline.py

# Complete Dagster pipeline
dagster dev -f pipelines/examples/Dagster_ML_Pipeline.py

# Airflow reference (read the code)
# pipelines/examples/Airflow_ML_Pipeline.py
```

### Viewing Results

- **MLflow UI**: http://localhost:5000
- **Dagster UI**: http://localhost:3000
- **Prefect UI**: `prefect server start` → http://localhost:4200

---

## 7. Quick Reference

### Prefect Patterns

```python
# Basic task
@task
def my_task(df: pd.DataFrame) -> pd.DataFrame:
    return process(df)

# With retries
@task(retries=3, retry_delay_seconds=60)
def resilient_task():
    ...

# With caching
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def cached_task(df: pd.DataFrame):
    ...

# Flow with parameters
@flow(log_prints=True)
def my_flow(n_estimators: int = 100):
    ...

# Subflow composition
@flow
def parent_flow():
    data = data_prep_subflow()
    result = training_subflow(data)
```

### Dagster Patterns

```python
# Asset definition
@asset(group_name="features", description="Engineered features")
def customer_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Dependency inferred from parameter name!
    return process(raw_data)

# Materialize assets
materialize([raw_data, customer_features, model])
```

### MLflow in Orchestrators

```python
# Inside a Prefect task
@task
def train_with_mlflow(df, params):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("my-experiment")

    with mlflow.start_run():
        mlflow.log_params(params)
        model = train(df)
        mlflow.log_metrics(evaluate(model))
        mlflow.sklearn.log_model(model, "model")

    return model
```

### Comparison Table

| Aspect | Prefect | Airflow | Dagster |
|--------|---------|---------|---------|
| Define unit | `@task` | `PythonOperator` | `@asset` |
| Define pipeline | `@flow` | `DAG(...)` | `Definitions` |
| Data passing | Return values | XCom + files | Function params |
| Retries | `@task(retries=3)` | `default_args` | Policies |
| Caching | `cache_key_fn=...` | Manual | Built-in |
| Dependencies | Implicit (calls) | Explicit (`>>`) | Inferred |
| Philosophy | Task-centric | Task-centric | Asset-centric |

---

## Next Steps

1. **Start with Prefect**: `python pipelines/workshop/prefect/Prefect_Workshop.py part1`
2. **Progress through all parts**: part1 → part2 → part3 → part4 → full
3. **Try Dagster bonus**: Transform your knowledge to asset-centric thinking
4. **Review Airflow**: Understand why alternatives exist
5. **Explore reference pipelines**: See production-ready patterns
