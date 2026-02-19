# =============================================================================
# Dagster Workshop - BONUS: From Tasks to Assets
# =============================================================================
#
# You've completed the Prefect workshop. Now let's see how the SAME pipeline
# looks in Dagster - and why the difference matters.
#
# THE KEY SHIFT:
#   Prefect: "Run load_data, then engineer_features, then train_model..."
#   Dagster: "I want trained_model to exist. Figure out what's needed."
#
# This workshop is a GUIDED TRANSFORMATION:
#   1. We start with your completed Prefect pipeline
#   2. We convert each task to a Dagster asset
#   3. We explore the Dagster UI and asset graph
#   4. We learn materialization patterns
#   5. We add SCHEDULES for automation!
#
# -----------------------------------------------------------------------------
# SETUP (Docker - Recommended)
# -----------------------------------------------------------------------------
#
# 1. Start the stack:
#      docker-compose --profile dagster up -d
#
# 2. Access the UIs:
#      - Dagster: http://localhost:3000 (assets, schedules, runs)
#      - MLflow:  http://localhost:5000 (experiments, models)
#
# 3. In Dagster UI:
#      - View asset graph
#      - Materialize assets manually
#      - Enable schedules for automation
#
# =============================================================================

from dagster import (
    asset,
    Definitions,
    materialize,
    AssetSelection,
    ScheduleDefinition,
    define_asset_job,
    AssetKey,
)
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os

# -----------------------------------------------------------------------------
# Configuration (same as Prefect)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "workshop-dagster"

FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'rfm_score'
]


# =============================================================================
# PART 1: THE TRANSFORMATION - From @task to @asset
# =============================================================================
#
# Here's your Prefect code (from the workshop):
#
#   @task
#   def load_data() -> pd.DataFrame:
#       return pd.read_csv(DATA_PATH)
#
#   @task
#   def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#       ...
#       return df
#
#   @flow
#   def training_pipeline():
#       df = load_data()
#       features = engineer_features(df)
#       ...
#
# In Dagster, we don't define "tasks to run" - we define "data that exists".
# Let's transform each piece:
# =============================================================================

# -----------------------------------------------------------------------------
# ASSET 1: Raw Customer Data
# -----------------------------------------------------------------------------
# Prefect version:
#   @task
#   def load_data() -> pd.DataFrame:
#       return pd.read_csv(DATA_PATH)
#
# Dagster version:

@asset(
    group_name="ingestion",
    description="Raw customer data loaded from CSV or generated synthetically"
)
def raw_customer_data() -> pd.DataFrame:
    """
    SOURCE ASSET: No dependencies (no parameters).

    This is equivalent to your Prefect load_data() task, but:
    - It's named as DATA (raw_customer_data) not ACTION (load_data)
    - It has metadata (group_name, description)
    - It's a piece of data that EXISTS in your system
    """
    print("Loading raw customer data...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
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

    print(f"Asset contains {len(df)} customers")
    return df


# -----------------------------------------------------------------------------
# ASSET 2: Engineered Features
# -----------------------------------------------------------------------------
# Prefect version:
#   @task
#   def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#       ...
#
# Dagster version:
# NOTICE: The parameter name IS the dependency!

@asset(
    group_name="features",
    description="Customer features with RFM scores and ratios"
)
def customer_features(raw_customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    DERIVED ASSET: Depends on raw_customer_data.

    KEY INSIGHT: The parameter name 'raw_customer_data' matches the asset name
    above. Dagster AUTOMATICALLY knows this asset depends on raw_customer_data.

    No explicit wiring needed! Compare with Prefect:
        features = engineer_features(df)  # You wire it in the flow

    In Dagster, the dependency is DECLARED in the function signature.
    """
    print("Computing customer_features asset...")
    df = raw_customer_data.copy()

    # Feature engineering (same logic as Prefect)
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Asset shape: {df.shape}")
    return df


# -----------------------------------------------------------------------------
# ASSET 3: Trained Model
# -----------------------------------------------------------------------------
# In Prefect, train_model was a task that returned (model, X_test, y_test, run_id)
# In Dagster, we model this as a DICT asset containing everything

@asset(
    group_name="training",
    description="Trained RandomForest model with MLflow tracking"
)
def trained_model(customer_features: pd.DataFrame) -> dict:
    """
    MODEL ASSET: Contains model + metadata.

    This asset:
    - Depends on customer_features (inferred from parameter)
    - Integrates with MLflow for tracking
    - Returns a dict with model, metrics, and run_id

    When you ask Dagster to materialize trained_model, it automatically
    materializes raw_customer_data and customer_features first!
    """
    print("Training model (with MLflow tracking)...")

    # Prepare data
    X = customer_features[FEATURE_COLS]
    y = customer_features['churned']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"dagster-{datetime.now().strftime('%H%M%S')}") as run:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

        mlflow.log_params(params)
        mlflow.log_param("orchestrator", "dagster")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained! Accuracy: {metrics['accuracy']:.4f}")

    # Return everything as a dict (this IS the asset)
    return {
        "model": model,
        "run_id": run.info.run_id,
        "metrics": metrics,
        "feature_cols": FEATURE_COLS
    }


# -----------------------------------------------------------------------------
# ASSET 4: Predictions
# -----------------------------------------------------------------------------
# Notice: This asset has TWO dependencies!

@asset(
    group_name="predictions",
    description="Churn predictions for all customers"
)
def churn_predictions(
    trained_model: dict,
    customer_features: pd.DataFrame
) -> pd.DataFrame:
    """
    MULTI-DEPENDENCY ASSET: Depends on trained_model AND customer_features.

    Look at the asset graph in the UI - you'll see:

        raw_customer_data
              │
              v
        customer_features ────┐
              │               │
              v               │
        trained_model         │
              │               │
              └───────┬───────┘
                      v
            churn_predictions

    Both dependencies are inferred from parameter names!
    """
    print("Generating predictions...")

    model = trained_model["model"]
    X = customer_features[FEATURE_COLS]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        'customer_id': customer_features['customer_id'],
        'churn_predicted': predictions,
        'churn_probability': probabilities,
        'predicted_at': datetime.now()
    })

    high_risk = (probabilities > 0.7).sum()
    print(f"Generated {len(result)} predictions ({high_risk} high-risk)")

    return result


# -----------------------------------------------------------------------------
# ASSET 5: Saved Predictions (I/O as an asset!)
# -----------------------------------------------------------------------------

@asset(
    group_name="output",
    description="Predictions persisted to CSV file"
)
def saved_predictions(churn_predictions: pd.DataFrame) -> dict:
    """
    OUTPUT ASSET: Represents "predictions that have been saved".

    Even file I/O is modeled as data! This asset depends on churn_predictions
    and produces metadata about what was saved.

    This is powerful because:
    - You can see in the UI when predictions were last saved
    - You can re-materialize just this asset to re-save
    - The dependency chain is clear and visible
    """
    output_path = os.path.join(PROJECT_ROOT, "data", "predictions_dagster.csv")

    churn_predictions.to_csv(output_path, index=False)

    print(f"Saved {len(churn_predictions)} predictions to {output_path}")

    return {
        "path": output_path,
        "record_count": len(churn_predictions),
        "high_risk_count": int((churn_predictions['churn_probability'] > 0.7).sum()),
        "saved_at": datetime.now().isoformat()
    }


# =============================================================================
# PART 2: THE DEFINITIONS - Registering Assets
# =============================================================================
#
# In Prefect, you define a @flow that wires tasks together.
# In Dagster, you register assets with Definitions - no wiring needed!

all_assets = [
    raw_customer_data,
    customer_features,
    trained_model,
    churn_predictions,
    saved_predictions,
]


# =============================================================================
# PART 5: AUTOMATION - Jobs & Schedules
# =============================================================================
#
# This is where ORCHESTRATION becomes real automation!
#
# In Dagster:
# - A JOB defines WHICH assets to materialize
# - A SCHEDULE defines WHEN to run the job
# - The DAEMON (running in Docker) executes schedules
#
# =============================================================================

# -----------------------------------------------------------------------------
# JOB: Which assets to materialize
# -----------------------------------------------------------------------------

training_job = define_asset_job(
    name="churn_training_job",
    selection=AssetSelection.all(),  # Materialize all assets
    description="Full churn prediction pipeline - trains model and generates predictions"
)

# Alternative: selective jobs
data_prep_job = define_asset_job(
    name="data_preparation_job",
    selection=[AssetKey("raw_customer_data"), AssetKey("customer_features")],
    description="Prepare data without training"
)

# -----------------------------------------------------------------------------
# SCHEDULE: When to run the job
# -----------------------------------------------------------------------------

training_schedule = ScheduleDefinition(
    name="churn_training_schedule",
    job=training_job,
    cron_schedule="*/2 * * * *",  # Every 2 minutes (for demo)
    description="Automated retraining every 2 minutes"
)

# Alternative schedule examples (commented out):
# daily_training = ScheduleDefinition(
#     name="daily_training",
#     job=training_job,
#     cron_schedule="0 6 * * *",  # Every day at 6 AM
# )

# -----------------------------------------------------------------------------
# DEFINITIONS: Register everything with Dagster
# -----------------------------------------------------------------------------

defs = Definitions(
    assets=all_assets,
    jobs=[training_job, data_prep_job],
    schedules=[training_schedule],
)


# =============================================================================
# PART 3: MATERIALIZATION - Running the Pipeline
# =============================================================================
#
# "Materializing" an asset means computing it and storing the result.
#
# Key differences from Prefect:
#
# Prefect:
#   training_pipeline()  # Run the whole flow
#
# Dagster:
#   materialize([trained_model])  # "I want trained_model to exist"
#   # Dagster figures out it needs raw_customer_data and customer_features first!
#
# You can also materialize SUBSETS:
#   materialize([customer_features])  # Just data prep, no training
#   materialize([churn_predictions])  # Everything needed for predictions
# =============================================================================

def run_full_pipeline():
    """Materialize all assets (equivalent to running the full Prefect flow)."""
    print("=" * 60)
    print("MATERIALIZING ALL ASSETS")
    print("=" * 60)

    result = materialize(all_assets)

    # Get outputs
    saved = result.output_for_node("saved_predictions")
    model_data = result.output_for_node("trained_model")

    print("\n" + "=" * 60)
    print("MATERIALIZATION COMPLETE")
    print("=" * 60)
    print(f"Model accuracy:     {model_data['metrics']['accuracy']:.4f}")
    print(f"Predictions saved:  {saved['record_count']}")
    print(f"High-risk customers:{saved['high_risk_count']}")
    print(f"Output file:        {saved['path']}")
    print(f"MLflow run:         {model_data['run_id']}")

    return result


def run_data_prep_only():
    """Materialize only data preparation assets (no training)."""
    print("=" * 60)
    print("MATERIALIZING DATA PREP ONLY")
    print("=" * 60)

    # Only materialize up to customer_features
    result = materialize([raw_customer_data, customer_features])

    features = result.output_for_node("customer_features")
    print(f"\nData prepared: {features.shape}")

    return result


def run_from_existing_features():
    """
    Demonstrate partial materialization.

    In a real scenario, you might have customer_features already computed
    and just want to retrain the model.
    """
    print("=" * 60)
    print("SELECTIVE MATERIALIZATION")
    print("=" * 60)
    print("This shows how Dagster can re-run only what's needed.")
    print("In the UI, you can click individual assets to materialize them.")

    # Materialize everything
    result = materialize(all_assets)

    return result


# =============================================================================
# PART 4: THE UI - The Real Dagster Experience
# =============================================================================
#
# The command-line is useful, but Dagster's power is in the UI!
#
# Start with Docker:
#   docker-compose --profile dagster up -d
#
# Then open http://localhost:3000 and explore:
#
# 1. ASSET GRAPH (Assets > View global asset lineage)
#    - See the visual dependency graph
#    - Click an asset to see its metadata
#    - See which assets are materialized (have data) vs unmaterialized
#
# 2. MATERIALIZE (Manual)
#    - Click "Materialize all" to run everything
#    - Or click individual assets to materialize just that subset
#    - Watch the logs in real-time
#
# 3. SCHEDULES (Automation!)
#    - Go to Overview > Schedules
#    - Find "churn_training_schedule"
#    - Toggle it ON
#    - Watch it run every 2 minutes!
#    - Check MLflow UI for new experiments
#
# 4. JOBS
#    - Go to Overview > Jobs
#    - See defined jobs (churn_training_job, data_preparation_job)
#    - Launch jobs manually or via schedules
#
# 5. RUNS
#    - See history of all materializations
#    - Debug failed runs
#    - Re-run from failures
#
# =============================================================================


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DAGSTER WORKSHOP - From Tasks to Assets")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "full":
        run_full_pipeline()

    elif mode == "data":
        run_data_prep_only()

    elif mode == "selective":
        run_from_existing_features()

    else:
        print("""
DAGSTER WORKSHOP - BONUS
========================

This workshop transforms your Prefect pipeline into Dagster assets.

SETUP (Docker - Recommended):
  docker-compose --profile dagster up -d

  UIs:
    Dagster: http://localhost:3000
    MLflow:  http://localhost:5000

COMMAND LINE USAGE (for testing):
  python Dagster_Workshop.py full       # Materialize all assets
  python Dagster_Workshop.py data       # Only data prep (no training)
  python Dagster_Workshop.py selective  # Demo selective materialization

THE KEY CONCEPTS:

1. ASSETS vs TASKS
   Prefect: @task def load_data()     -> "run this function"
   Dagster: @asset def customer_data  -> "this data exists"

2. AUTOMATIC DEPENDENCIES
   Prefect: features = engineer(df)   -> you wire it in the flow
   Dagster: def features(raw_data):   -> dependency from parameter name!

3. MATERIALIZATION
   Prefect: flow()                    -> run the whole pipeline
   Dagster: materialize([asset])      -> "make this exist" (deps auto-resolved)

4. JOBS & SCHEDULES (Automation!)
   Jobs define WHICH assets to materialize
   Schedules define WHEN to run jobs
   The daemon (in Docker) executes them automatically

AUTOMATION:
  1. Open Dagster UI: http://localhost:3000
  2. Go to Overview > Schedules
  3. Enable "churn_training_schedule"
  4. Watch runs appear every 2 minutes!
  5. Check MLflow UI for new experiments

THE GRAPH:
  Open the UI to see your data lineage visually.
  Click assets to materialize them individually.
""")

        print(f"Dagster UI: http://localhost:3000")
        print(f"MLflow UI:  {MLFLOW_TRACKING_URI}")
