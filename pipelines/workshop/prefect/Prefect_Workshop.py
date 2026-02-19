# =============================================================================
# Prefect Workshop - Orchestrating ML Workflows
# =============================================================================
#
# This workshop teaches ORCHESTRATION through ML use cases.
#
# SECTIONS:
#   Part 1: Tasks & Flows - The basics
#   Part 2: Resilience - Retries, error handling
#   Part 3: Efficiency - Caching, parallel execution
#   Part 4: Flexibility - Parameters, subflows
#   Part 5: Full Pipeline - Everything with MLflow
#   Part 6: AUTOMATION - Deploy, schedule, watch it run!
#
# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
#
# 1. Start the orchestration stack:
#      docker-compose up -d
#
# 2. Access the UIs:
#      - Prefect: http://localhost:4200 (flows, deployments, runs)
#      - MLflow:  http://localhost:5000 (experiments, models)
#
# 3. Run workshop parts:
#      python pipelines/workshop/prefect/Prefect_Workshop.py part1
#      ...
#      python pipelines/workshop/prefect/Prefect_Workshop.py deploy
#
# =============================================================================

from prefect import flow, task, serve
from prefect.tasks import task_input_hash
from prefect.client import get_client
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import random
import os
import asyncio

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# When running in Docker, these env vars are set by docker-compose
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")

FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age'
]


# =============================================================================
# PART 1: TASKS & FLOWS - The Basics
# =============================================================================
#
# ML PROBLEM:
#   Your notebook runs cells sequentially. If cell 5 fails, you re-run
#   everything. There's no structure, no visibility, no reusability.
#
# ORCHESTRATION SOLUTION:
#   Break your code into TASKS (single responsibilities) and FLOWS (pipelines).
#   Each task is independent, logged, and can be retried.
#
# =============================================================================

@task
def load_data() -> pd.DataFrame:
    """
    A TASK is a single unit of work.

    Key points:
    - @task decorator turns a function into an orchestrated task
    - Return values flow to the next task (no file I/O needed!)
    - Prefect automatically logs: start time, duration, success/failure
    """
    print("Loading customer data...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Generate synthetic data
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'customer_id': range(1, n + 1),
            'recency_days': np.random.exponential(30, n).astype(int),
            'frequency': np.random.poisson(5, n),
            'monetary_value': np.random.exponential(500, n),
            'avg_order_value': np.random.exponential(100, n),
            'days_since_signup': np.random.randint(30, 1000, n),
            'total_orders': np.random.poisson(8, n),
            'support_tickets': np.random.poisson(2, n),
            'age': np.random.randint(18, 70, n),
            'churned': np.random.binomial(1, 0.3, n)
        })

    print(f"Loaded {len(df)} rows")
    return df


@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering task."""
    print("Engineering features...")
    df = df.copy()

    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)

    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Created {len(df.columns)} features")
    return df


@flow(name="basic-pipeline", log_prints=True)
def basic_pipeline():
    """
    A FLOW orchestrates tasks.

    Call tasks like normal functions - data flows through return values.
    """
    raw_data = load_data()
    features = engineer_features(raw_data)
    print(f"Pipeline complete! Shape: {features.shape}")
    return features


# =============================================================================
# PART 2: RESILIENCE - Handling Failures
# =============================================================================

@task(retries=3, retry_delay_seconds=5)
def load_from_unreliable_api() -> pd.DataFrame:
    """
    RETRIES handle transient failures automatically.

    This task will:
    1. Try to run
    2. If it fails, wait 5 seconds
    3. Try again (up to 3 times)
    """
    if random.random() < 0.5:
        print("API call failed! (simulated)")
        raise ConnectionError("API temporarily unavailable")

    print("API call succeeded!")
    return load_data()


@task(retries=3, retry_delay_seconds=[10, 30, 60])
def load_with_backoff() -> pd.DataFrame:
    """
    EXPONENTIAL BACKOFF: Wait longer between each retry.
    First failure: 10s, Second: 30s, Third: 60s
    """
    if random.random() < 0.7:
        raise ConnectionError("Rate limited!")
    return load_data()


# =============================================================================
# PART 3: EFFICIENCY - Caching & Parallelism
# =============================================================================

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def expensive_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    CACHING: Skip recomputation if inputs haven't changed.
    """
    print("Running expensive feature engineering...")
    time.sleep(2)  # Simulate expensive computation
    df = df.copy()
    df['expensive_feature'] = df['monetary_value'] * df['frequency']
    print("Feature engineering complete!")
    return df


@task
def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Train a Random Forest model."""
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Random Forest accuracy: {accuracy:.4f}")
    return {"name": "RandomForest", "model": model, "accuracy": accuracy}


@task
def train_gradient_boosting(X_train, y_train, X_test, y_test) -> dict:
    """Train a Gradient Boosting model."""
    print("Training Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Gradient Boosting accuracy: {accuracy:.4f}")
    return {"name": "GradientBoosting", "model": model, "accuracy": accuracy}


@task
def select_best_model(results: list) -> dict:
    """Select the model with highest accuracy."""
    best = max(results, key=lambda x: x["accuracy"])
    print(f"Best model: {best['name']} with accuracy {best['accuracy']:.4f}")
    return best


@flow(name="parallel-training", log_prints=True)
def parallel_training_flow():
    """
    PARALLEL EXECUTION: Train multiple models simultaneously.
    """
    df = load_data()
    df = engineer_features(df)

    X = df[FEATURE_COLS + ['rfm_score']]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # These run in parallel - no dependency between them!
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)

    best = select_best_model([rf_result, gb_result])
    return best


# =============================================================================
# PART 4: FLEXIBILITY - Parameters & Subflows
# =============================================================================

@task
def train_with_params(df: pd.DataFrame, n_estimators: int, max_depth: int) -> dict:
    """Task that accepts hyperparameters."""
    X = df[FEATURE_COLS + ['rfm_score']]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    print(f"Model: n_estimators={n_estimators}, max_depth={max_depth}, accuracy={accuracy:.4f}")
    return {"model": model, "params": {"n_estimators": n_estimators, "max_depth": max_depth}, "metrics": {"accuracy": accuracy, "f1": f1}}


@flow(name="parameterized-training", log_prints=True)
def parameterized_training_flow(n_estimators: int = 100, max_depth: int = 10):
    """
    PARAMETERIZED FLOW: Configure without changing code.
    """
    df = load_data()
    df = engineer_features(df)
    result = train_with_params(df, n_estimators, max_depth)
    return result


@flow(name="data-preparation", log_prints=True)
def data_preparation_subflow() -> pd.DataFrame:
    """SUBFLOW: Reusable data preparation."""
    df = load_data()
    df = engineer_features(df)
    return df


@flow(name="training-subflow", log_prints=True)
def training_subflow(df: pd.DataFrame, n_estimators: int = 100) -> dict:
    """Training as a separate subflow."""
    return train_with_params(df, n_estimators=n_estimators, max_depth=10)


# =============================================================================
# PART 5: FULL PIPELINE WITH MLFLOW
# =============================================================================

@task(retries=3, retry_delay_seconds=5)
def load_data_with_retry() -> pd.DataFrame:
    """Data loading with retry logic."""
    return load_data()


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def engineer_features_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering with caching."""
    return engineer_features(df)


@task
def train_with_mlflow(df: pd.DataFrame, n_estimators: int, max_depth: int, experiment_name: str) -> dict:
    """
    Training task with MLflow integration.

    Prefect handles: retries, caching, scheduling
    MLflow handles: experiment tracking, model versioning
    """
    feature_cols = FEATURE_COLS + ['rfm_score']
    X = df[feature_cols]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"prefect-{datetime.now().strftime('%H%M%S')}") as run:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "orchestrator": "prefect",
            "n_features": len(feature_cols)
        })

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {"accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred)}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"MLflow run: {run.info.run_id}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    return {"model": model, "run_id": run.info.run_id, "metrics": metrics}


@task
def register_model(run_id: str, model_name: str) -> str:
    """Register model to MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Registered {model_name} version {result.version}")
    return result.version


@task
def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for all customers."""
    feature_cols = FEATURE_COLS + ['rfm_score']
    X = df[feature_cols]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_predicted': predictions,
        'churn_probability': probabilities,
        'predicted_at': datetime.now()
    })

    high_risk = (probabilities > 0.7).sum()
    print(f"Generated {len(result)} predictions ({high_risk} high-risk)")
    return result


@task
def save_predictions(predictions: pd.DataFrame, output_path: str) -> str:
    """Save predictions to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path


@flow(name="churn-prediction-pipeline", log_prints=True)
def production_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    experiment_name: str = "workshop-prefect",
    model_name: str = "churn-predictor"
):
    """
    PRODUCTION ML PIPELINE

    Combines all patterns:
    - RETRIES on data loading
    - CACHING on feature engineering
    - PARAMETERS for hyperparameters
    - MLFLOW for tracking

    This flow can be:
    - Run manually: production_pipeline()
    - Deployed with schedule: See Part 6
    """
    print("=" * 60)
    print("CHURN PREDICTION PIPELINE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Data
    df = load_data_with_retry()
    df = engineer_features_cached(df)

    # Training
    result = train_with_mlflow(df, n_estimators, max_depth, experiment_name)

    # Registration
    version = register_model(result["run_id"], model_name)

    # Inference
    predictions = generate_predictions(result["model"], df)
    output_path = os.path.join(PROJECT_ROOT, "data", "predictions_workshop.csv")
    save_predictions(predictions, output_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model: {model_name} v{version}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Predictions: {len(predictions)}")

    return {"model_version": version, "metrics": result["metrics"], "predictions_count": len(predictions)}


# =============================================================================
# PART 6: AUTOMATION - Deploy & Schedule
# =============================================================================
#
# This is where ORCHESTRATION becomes real automation!
#
# Until now, we ran flows manually. In production:
# - Flows run on SCHEDULES (daily retraining, hourly predictions)
# - A WORKER executes the flows (in Docker)
# - You MONITOR in the UI (see runs, failures, logs)
#
# =============================================================================

def deploy_with_schedule():
    """
    DEPLOY the pipeline with a schedule.

    This creates a DEPLOYMENT:
    - Registers the flow with Prefect server
    - Sets up a schedule (every 2 minutes for demo)
    - Worker picks it up and executes it

    After running this:
    1. Open Prefect UI: http://localhost:4200
    2. Go to Deployments
    3. See "churn-prediction-pipeline/scheduled-training"
    4. Watch runs appear every 2 minutes!
    5. Check MLflow UI: http://localhost:5000 for new experiments
    """
    print("=" * 60)
    print("DEPLOYING PIPELINE WITH SCHEDULE")
    print("=" * 60)
    print("\nThis will:")
    print("1. Register the flow with Prefect server")
    print("2. Set up a schedule (every 2 minutes)")
    print("3. Worker will automatically execute it")
    print("\nAfter deployment:")
    print(f"  - Prefect UI: http://localhost:4200")
    print(f"  - MLflow UI:  {MLFLOW_TRACKING_URI}")
    print("=" * 60)

    # Deploy using serve() - this runs the scheduler locally
    # In production, you'd use flow.deploy() with a work pool
    production_pipeline.serve(
        name="scheduled-training",
        cron="*/2 * * * *",  # Every 2 minutes (for demo)
        tags=["workshop", "ml", "churn"],
        description="Automated churn model retraining - runs every 2 minutes",
        parameters={
            "n_estimators": 100,
            "max_depth": 10,
            "experiment_name": "workshop-automated",
            "model_name": "churn-predictor-auto"
        }
    )


def run_single():
    """Run the pipeline once (for testing before deployment)."""
    print("Running pipeline once...")
    result = production_pipeline(
        n_estimators=100,
        max_depth=10,
        experiment_name="workshop-prefect",
        model_name="churn-predictor"
    )
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("PREFECT WORKSHOP - Orchestrating ML Workflows")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "part1":
        print("\nPART 1: Basic Tasks & Flows")
        print("-" * 40)
        result = basic_pipeline()
        print(f"Result shape: {result.shape}")

    elif mode == "part2":
        print("\nPART 2: Resilience (Retries)")
        print("-" * 40)

        @flow
        def retry_demo():
            return load_from_unreliable_api()

        try:
            retry_demo()
        except Exception as e:
            print(f"Failed after retries: {e}")

    elif mode == "part3":
        print("\nPART 3: Efficiency (Caching & Parallelism)")
        print("-" * 40)
        result = parallel_training_flow()

    elif mode == "part4":
        print("\nPART 4: Flexibility (Parameters & Subflows)")
        print("-" * 40)
        result = parameterized_training_flow(n_estimators=50)

    elif mode == "part5" or mode == "full":
        print("\nPART 5: Full Pipeline with MLflow")
        print("-" * 40)
        result = run_single()

    elif mode == "deploy":
        print("\nPART 6: Deploy with Schedule (AUTOMATION!)")
        print("-" * 40)
        print("\nThis will start a long-running process that:")
        print("1. Registers the flow")
        print("2. Runs it every 2 minutes")
        print("3. Logs to MLflow")
        print("\nPress Ctrl+C to stop.\n")
        deploy_with_schedule()

    else:
        print("""
PREFECT WORKSHOP - ML Orchestration
====================================

SETUP:
  docker-compose up -d

  UIs:
    Prefect: http://localhost:4200
    MLflow:  http://localhost:5000

MODES:
  part1    Tasks & Flows (basics)
  part2    Resilience (retries)
  part3    Efficiency (caching, parallelism)
  part4    Flexibility (parameters, subflows)
  part5    Full Pipeline with MLflow
  deploy   AUTOMATION - Deploy with schedule!

WORKSHOP FLOW:
  1. Run part1-part5 to learn patterns
  2. Run 'deploy' to see real automation
  3. Open Prefect UI to watch runs
  4. Open MLflow UI to see experiments

AUTOMATION (deploy mode):
  - Deploys flow to Prefect
  - Schedules to run every 2 minutes
  - Watch in Prefect UI: Deployments > Runs
  - Watch in MLflow UI: new experiments appear!

Example:
  python Prefect_Workshop.py part1
  python Prefect_Workshop.py deploy
""")
