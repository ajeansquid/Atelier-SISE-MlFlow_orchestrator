# =============================================================================
# Prefect ML Pipeline - Customer Churn Prediction
# =============================================================================
#
# This pipeline demonstrates Prefect's Pythonic approach to orchestration:
# - @task and @flow decorators on regular Python functions
# - In-memory data passing (just return values!)
# - Native Python type hints
# - No XCom, no file I/O for intermediate data
# - Retries/caching declarative on decorators
#
# Two flows available:
# 1. Training pipeline: Train model, evaluate, register to MLflow
# 2. Inference pipeline: Load from registry, predict, save results
#
# Run locally: python Prefect_ML_Pipeline.py
# =============================================================================

from prefect import flow, task
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# MLflow Configuration
# Defaults to Docker-hosted MLflow server (start with: cd docker && docker-compose up -d)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer-churn-prefect"
MODEL_NAME = "churn-predictor-prefect"  # For model registry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Get project root (parent of pipelines/examples/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_prefect.csv")
RANDOM_SEED = 42

# Feature columns used for training
FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'order_frequency',
    'support_per_order', 'rfm_score'
]


# =============================================================================
# TASKS - Notice how simple they are compared to Airflow!
# =============================================================================

@task(retries=3, retry_delay_seconds=10)
def load_customer_data() -> pd.DataFrame:
    """
    Load customer data.

    Tries to load from CSV if available, otherwise generates synthetic data.
    This makes it easy to run the workshop without external dependencies.
    """
    print("Loading customer data...")

    # Try loading from CSV first
    if os.path.exists(DATA_PATH):
        print(f"Loading from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        # Generate synthetic data (same as notebooks)
        print("Generating synthetic customer data...")
        np.random.seed(RANDOM_SEED)
        n_customers = 5000

        df = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'recency_days': np.random.exponential(30, n_customers).astype(int),
            'frequency': np.random.poisson(5, n_customers),
            'monetary_value': np.random.exponential(500, n_customers),
            'avg_order_value': np.random.exponential(100, n_customers),
            'days_since_signup': np.random.randint(30, 1000, n_customers),
            'total_orders': np.random.poisson(8, n_customers),
            'support_tickets': np.random.poisson(2, n_customers),
            'age': np.random.randint(18, 70, n_customers),
        })

        # Create target variable (churn) based on realistic logic
        churn_prob = 1 / (1 + np.exp(-(
            0.02 * df['recency_days'] -
            0.1 * df['frequency'] -
            0.001 * df['monetary_value'] +
            0.2 * df['support_tickets'] -
            2
        )))
        df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

    print(f"Loaded {len(df)} customers")
    return df  # ✅ Just return it - no file I/O needed!


@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for the model.

    Notice: We receive a DataFrame directly as input and return one as output.
    No XCom, no file paths, no pickle - just Python!
    """
    print("Engineering features...")
    df = df.copy()

    # Ratio features
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    # RFM score (Recency, Frequency, Monetary)
    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Engineered features. Shape: {df.shape}")
    return df  # ✅ Just return it!


@task(retries=2)
def train_model(df: pd.DataFrame) -> tuple:
    """
    Train Random Forest classifier with MLflow tracking.

    Returns the model and test data for evaluation.
    """
    print("Training model...")

    # Prepare data
    X = df[FEATURE_COLS]
    y = df['churned']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"prefect-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        # Model hyperparameters
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 20,
            "random_state": RANDOM_SEED
        }

        # Train model
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("orchestrator", "prefect")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("training_samples", len(X_train))

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        mlflow_run_id = run.info.run_id

    print(f"Model trained. MLflow run: {mlflow_run_id}")

    # ✅ Return everything needed - no file I/O!
    return model, X_test, y_test, mlflow_run_id


@task
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, mlflow_run_id: str) -> dict:
    """
    Evaluate model performance and log metrics to MLflow.
    """
    print("Evaluating model...")

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    # Log metrics to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(metrics)

    return metrics  # ✅ Just return it!


@task
def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate churn predictions for all customers.
    """
    print("Generating predictions...")

    X = df[FEATURE_COLS]

    # Generate predictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Create results dataframe
    predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    print(f"Generated predictions for {len(predictions)} customers")
    return predictions  # ✅ Just return it!


@task
def save_predictions(predictions: pd.DataFrame, output_path: str = OUTPUT_PATH) -> str:
    """
    Save predictions to local CSV file.

    In production, this would write to a database or data warehouse.
    """
    print(f"Saving predictions to {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    predictions.to_csv(output_path, index=False)

    print(f"Saved {len(predictions)} predictions to {output_path}")
    return output_path


@task
def register_model(mlflow_run_id: str, metrics: dict) -> str:
    """
    Register the trained model in MLflow Model Registry.

    This allows the inference pipeline to load by name instead of run_id.
    """
    print(f"Registering model to registry: {MODEL_NAME}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{mlflow_run_id}/model"

    # Register model
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"Registered {MODEL_NAME} version {model_version.version}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")

    return model_version.version


@task(retries=2)
def load_model_from_registry(model_name: str = MODEL_NAME, version: str = "latest"):
    """
    Load model from MLflow Model Registry.

    This is the production pattern - load by name, not by run_id.
    """
    print(f"Loading model {model_name}/{version} from registry...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{version}"

    model = mlflow.sklearn.load_model(model_uri)

    print(f"Model loaded successfully")
    return model


@task
def run_batch_inference(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run batch inference on customer data.

    This simulates daily prediction jobs.
    """
    print(f"Running inference on {len(df)} customers...")

    X = df[FEATURE_COLS]

    # Generate predictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Create results dataframe
    predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    # Summary stats
    high_risk = (churn_probability > 0.7).sum()
    print(f"High risk customers (>70%): {high_risk}")

    return predictions


# =============================================================================
# FLOW - The main pipeline orchestration
# =============================================================================

@flow(name="churn-training-pipeline", log_prints=True)
def training_pipeline():
    """
    Training pipeline: Train, evaluate, and register model to MLflow.

    This would run on a schedule (e.g., weekly) or when new training data arrives.
    """

    # Load data
    raw_data = load_customer_data()

    # Engineer features
    features = engineer_features(raw_data)

    # Train model
    model, X_test, y_test, mlflow_run_id = train_model(features)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, mlflow_run_id)

    # Register model to registry
    model_version = register_model(mlflow_run_id, metrics)

    # Return summary
    return {
        "customers_trained_on": len(features),
        "model_accuracy": metrics['accuracy'],
        "model_f1": metrics['f1'],
        "mlflow_run_id": mlflow_run_id,
        "model_version": model_version
    }


@flow(name="churn-inference-pipeline", log_prints=True)
def inference_pipeline(model_version: str = "latest"):
    """
    Inference pipeline: Load model from registry and generate predictions.

    This would run daily on new customer data.
    Separating training from inference is a production best practice!
    """

    # Load data
    raw_data = load_customer_data()

    # Engineer features (same as training)
    features = engineer_features(raw_data)

    # Load model from registry (not from training!)
    model = load_model_from_registry(MODEL_NAME, model_version)

    # Run batch inference
    predictions = run_batch_inference(model, features)

    # Save predictions
    inference_output = os.path.join(PROJECT_ROOT, "data", "predictions_inference.csv")
    output_path = save_predictions(predictions, inference_output)

    # Return summary
    return {
        "customers_scored": len(predictions),
        "high_risk_customers": int((predictions['churn_probability'] > 0.7).sum()),
        "model_version": model_version,
        "output_file": output_path
    }


@flow(name="churn-full-pipeline", log_prints=True)
def full_pipeline():
    """
    Full pipeline: Train model AND run inference.

    Demonstrates the complete ML workflow in one flow.
    In production, these would typically be separate scheduled flows.
    """

    # Phase 1: Training
    print("=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60)

    raw_data = load_customer_data()
    features = engineer_features(raw_data)
    model, X_test, y_test, mlflow_run_id = train_model(features)
    metrics = evaluate_model(model, X_test, y_test, mlflow_run_id)
    model_version = register_model(mlflow_run_id, metrics)

    # Phase 2: Inference (using the just-registered model)
    print("\n" + "=" * 60)
    print("PHASE 2: INFERENCE")
    print("=" * 60)

    # Load from registry (proves the registration worked)
    loaded_model = load_model_from_registry(MODEL_NAME, "latest")
    predictions = run_batch_inference(loaded_model, features)
    output_path = save_predictions(predictions)

    return {
        "training": {
            "model_accuracy": metrics['accuracy'],
            "model_f1": metrics['f1'],
            "model_version": model_version,
            "mlflow_run_id": mlflow_run_id
        },
        "inference": {
            "customers_scored": len(predictions),
            "high_risk_customers": int((predictions['churn_probability'] > 0.7).sum()),
            "output_file": output_path
        }
    }


# =============================================================================
# MAIN - Run the pipeline
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Customer Churn Prediction Pipeline (Prefect)")
    print("=" * 60)

    # Parse command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "train":
        # Training only
        print("Mode: TRAINING ONLY")
        print("=" * 60)
        result = training_pipeline()

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Customers trained on:   {result['customers_trained_on']}")
        print(f"Model accuracy:         {result['model_accuracy']:.4f}")
        print(f"Model F1 score:         {result['model_f1']:.4f}")
        print(f"Model version:          {result['model_version']}")
        print(f"MLflow run ID:          {result['mlflow_run_id']}")

    elif mode == "inference":
        # Inference only (requires a trained model in registry)
        print("Mode: INFERENCE ONLY")
        print("=" * 60)
        version = sys.argv[2] if len(sys.argv) > 2 else "latest"
        result = inference_pipeline(model_version=version)

        print("\n" + "=" * 60)
        print("Inference completed!")
        print("=" * 60)
        print(f"Customers scored:       {result['customers_scored']}")
        print(f"High risk customers:    {result['high_risk_customers']}")
        print(f"Model version used:     {result['model_version']}")
        print(f"Predictions saved to:   {result['output_file']}")

    else:
        # Full pipeline (train + inference)
        print("Mode: FULL PIPELINE (train + inference)")
        print("=" * 60)
        result = full_pipeline()

        print("\n" + "=" * 60)
        print("Full pipeline completed!")
        print("=" * 60)
        print("\nTRAINING RESULTS:")
        print(f"  Model accuracy:       {result['training']['model_accuracy']:.4f}")
        print(f"  Model F1 score:       {result['training']['model_f1']:.4f}")
        print(f"  Model version:        {result['training']['model_version']}")
        print(f"  MLflow run ID:        {result['training']['mlflow_run_id']}")
        print("\nINFERENCE RESULTS:")
        print(f"  Customers scored:     {result['inference']['customers_scored']}")
        print(f"  High risk customers:  {result['inference']['high_risk_customers']}")
        print(f"  Predictions saved to: {result['inference']['output_file']}")

    print("\n" + "=" * 60)
    print("View experiments at: http://localhost:5000")
    print("=" * 60)
    print("\nUsage:")
    print("  python Prefect_ML_Pipeline.py           # Full pipeline")
    print("  python Prefect_ML_Pipeline.py train     # Training only")
    print("  python Prefect_ML_Pipeline.py inference # Inference only")
    print("  python Prefect_ML_Pipeline.py inference 1  # Inference with model v1")
