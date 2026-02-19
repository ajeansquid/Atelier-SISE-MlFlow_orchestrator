# =============================================================================
# Dagster ML Pipeline - Customer Churn Prediction
# =============================================================================
#
# This pipeline demonstrates Dagster's asset-centric approach:
# - Assets (data artifacts) as first-class citizens, not tasks
# - Declarative: "what data should exist" vs "what tasks to run"
# - Automatic dependency inference from function signatures
# - Rich metadata (group_name, description, compute_kind)
# - Think "what exists" not "what runs"
#
# Run locally:
#   dagster dev -f Dagster_ML_Pipeline.py
#
# Or materialize assets:
#   python Dagster_ML_Pipeline.py
# =============================================================================

from dagster import asset, Definitions, materialize
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
# Configuration
# Defaults to Docker-hosted MLflow server (start with: cd docker && docker-compose up -d)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer-churn-dagster"
MODEL_NAME = "churn-predictor-dagster"  # For model registry

# Get project root (parent of pipelines/examples/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_dagster.csv")
INFERENCE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_dagster_inference.csv")
RANDOM_SEED = 42

# Feature columns used for training
FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'order_frequency',
    'support_per_order', 'rfm_score'
]


# =============================================================================
# ASSETS - Think "what data should exist", not "what tasks to run"
# =============================================================================

@asset(
    group_name="data_ingestion",
    description="Raw customer data loaded from CSV or generated synthetically",
    compute_kind="pandas"
)
def customer_data() -> pd.DataFrame:
    """
    Load or generate customer data.

    DAGSTER KEY CONCEPT: This is an ASSET - a piece of data that exists.
    Dagster thinks about data artifacts, not tasks.
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

        # Create target variable (churn)
        churn_prob = 1 / (1 + np.exp(-(
            0.02 * df['recency_days'] -
            0.1 * df['frequency'] -
            0.001 * df['monetary_value'] +
            0.2 * df['support_tickets'] -
            2
        )))
        df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

    print(f"Loaded {len(df)} customers")
    return df  # ✅ Return the data asset


@asset(
    group_name="feature_engineering",
    description="Engineered features including RFM scores and ratios",
    compute_kind="pandas"
)
def customer_features(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw customer data.

    DAGSTER KEY CONCEPT: Dependencies are inferred from function signature!
    Dagster automatically knows this asset depends on `customer_data`.
    """
    print("Engineering features...")
    df = customer_data.copy()

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
    return df  # ✅ Return the features asset


@asset(
    group_name="model_training",
    description="Trained Random Forest model with MLflow tracking",
    compute_kind="sklearn"
)
def churn_model(customer_features: pd.DataFrame) -> dict:
    """
    Train Random Forest classifier on customer features.

    DAGSTER KEY CONCEPT: We return a dict containing the model and metadata.
    This becomes an asset that other assets can depend on.
    """
    print("Training model...")

    # Prepare data
    X = customer_features[FEATURE_COLS]
    y = customer_features['churned']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"dagster-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
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
        mlflow.log_param("orchestrator", "dagster")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("training_samples", len(X_train))

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        mlflow_run_id = run.info.run_id

    print(f"Model trained. MLflow run: {mlflow_run_id}")

    # ✅ Return model asset as dict (includes metadata)
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'mlflow_run_id': mlflow_run_id,
        'training_samples': len(X_train)
    }


@asset(
    group_name="model_evaluation",
    description="Model performance metrics (accuracy, precision, recall, F1)",
    compute_kind="sklearn"
)
def model_metrics(churn_model: dict) -> dict:
    """
    Evaluate model performance on test set.

    DAGSTER KEY CONCEPT: This asset depends on `churn_model` asset.
    Dagster infers this from the function signature.
    """
    print("Evaluating model...")

    model = churn_model['model']
    X_test = churn_model['X_test']
    y_test = churn_model['y_test']
    mlflow_run_id = churn_model['mlflow_run_id']

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'evaluated_at': datetime.now().isoformat()
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    # Log metrics to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1", metrics['f1'])

    return metrics  # ✅ Return metrics asset


@asset(
    group_name="predictions",
    description="Churn probability predictions for all customers",
    compute_kind="sklearn"
)
def customer_predictions(churn_model: dict, customer_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate churn predictions for all customers.

    DAGSTER KEY CONCEPT: This asset depends on BOTH `churn_model` AND `customer_features`.
    Dagster automatically builds the correct execution graph.
    """
    print("Generating predictions...")

    model = churn_model['model']
    X = customer_features[FEATURE_COLS]

    # Generate predictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Create results dataframe
    predictions = pd.DataFrame({
        'customer_id': customer_features['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    print(f"Generated predictions for {len(predictions)} customers")
    return predictions  # ✅ Return predictions asset


@asset(
    group_name="predictions",
    description="Predictions saved to local CSV file",
    compute_kind="csv"
)
def saved_predictions(customer_predictions: pd.DataFrame) -> dict:
    """
    Save predictions to local CSV file.

    DAGSTER KEY CONCEPT: Even I/O operations are assets!
    This asset represents "predictions that have been persisted".
    """
    print(f"Saving predictions to {OUTPUT_PATH}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save to CSV
    customer_predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(customer_predictions)} predictions to {OUTPUT_PATH}")

    # ✅ Return metadata about what was saved
    return {
        'rows_saved': len(customer_predictions),
        'output_path': OUTPUT_PATH,
        'saved_at': datetime.now().isoformat()
    }


@asset(
    group_name="model_registry",
    description="Model registered to MLflow Model Registry",
    compute_kind="mlflow"
)
def registered_model(churn_model: dict, model_metrics: dict) -> dict:
    """
    Register the trained model to MLflow Model Registry.

    DAGSTER KEY CONCEPT: This asset depends on both the model AND metrics.
    We only register models that have been evaluated.
    """
    print(f"Registering model to MLflow registry: {MODEL_NAME}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{churn_model['mlflow_run_id']}/model"

    # Register model
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"Registered {MODEL_NAME} version {model_version.version}")
    print(f"Model accuracy: {model_metrics['accuracy']:.4f}")

    return {
        'model_name': MODEL_NAME,
        'model_version': model_version.version,
        'mlflow_run_id': churn_model['mlflow_run_id'],
        'accuracy': model_metrics['accuracy'],
        'registered_at': datetime.now().isoformat()
    }


# =============================================================================
# INFERENCE ASSETS - Load from registry and predict
# =============================================================================

@asset(
    group_name="inference",
    description="Model loaded from MLflow Model Registry for inference",
    compute_kind="mlflow"
)
def inference_model() -> dict:
    """
    Load model from MLflow Model Registry.

    DAGSTER KEY CONCEPT: This is a SOURCE asset for inference - no dependencies.
    It loads an already-registered model, decoupled from training.

    In production, training and inference are separate pipelines.
    """
    print(f"Loading model {MODEL_NAME}/latest from registry...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        # Get model version info
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        version = latest_version.version
    except Exception as e:
        raise RuntimeError(
            f"Could not load model '{MODEL_NAME}' from registry. "
            f"Run training pipeline first to register a model. Error: {e}"
        )

    print(f"Loaded model version {version}")

    return {
        'model': model,
        'model_name': MODEL_NAME,
        'model_version': version,
        'loaded_at': datetime.now().isoformat()
    }


@asset(
    group_name="inference",
    description="Fresh customer data for inference (separate from training data)",
    compute_kind="pandas"
)
def inference_customer_data() -> pd.DataFrame:
    """
    Load customer data for inference.

    DAGSTER KEY CONCEPT: This is separate from training data.
    In production, this would load NEW customers to score.
    """
    print("Loading customer data for inference...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Generate synthetic data
        np.random.seed(RANDOM_SEED + 1)  # Different seed for variation
        n_customers = 1000

        df = pd.DataFrame({
            'customer_id': range(10001, 10001 + n_customers),
            'recency_days': np.random.exponential(30, n_customers).astype(int),
            'frequency': np.random.poisson(5, n_customers),
            'monetary_value': np.random.exponential(500, n_customers),
            'avg_order_value': np.random.exponential(100, n_customers),
            'days_since_signup': np.random.randint(30, 1000, n_customers),
            'total_orders': np.random.poisson(8, n_customers),
            'support_tickets': np.random.poisson(2, n_customers),
            'age': np.random.randint(18, 70, n_customers),
        })

    print(f"Loaded {len(df)} customers for inference")
    return df


@asset(
    group_name="inference",
    description="Engineered features for inference data",
    compute_kind="pandas"
)
def inference_features(inference_customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for inference data.

    DAGSTER KEY CONCEPT: Same feature engineering logic as training,
    but applied to inference data (separate asset lineage).
    """
    print("Engineering features for inference...")
    df = inference_customer_data.copy()

    # Same feature engineering as training
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Inference features shape: {df.shape}")
    return df


@asset(
    group_name="inference",
    description="Churn predictions from inference pipeline",
    compute_kind="sklearn"
)
def inference_predictions(inference_model: dict, inference_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions using model from registry.

    DAGSTER KEY CONCEPT: This depends on inference_model (from registry)
    and inference_features (freshly computed), not the training pipeline.
    """
    print("Running batch inference...")

    model = inference_model['model']
    X = inference_features[FEATURE_COLS]

    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    predictions = pd.DataFrame({
        'customer_id': inference_features['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'model_version': inference_model['model_version'],
        'predicted_at': datetime.now()
    })

    high_risk = (churn_probability > 0.7).sum()
    print(f"Generated {len(predictions)} predictions ({high_risk} high-risk)")

    return predictions


@asset(
    group_name="inference",
    description="Inference predictions saved to CSV",
    compute_kind="csv"
)
def saved_inference_predictions(inference_predictions: pd.DataFrame) -> dict:
    """
    Save inference predictions to file.

    DAGSTER KEY CONCEPT: Separate output asset from training predictions.
    """
    print(f"Saving inference predictions to {INFERENCE_OUTPUT_PATH}...")

    os.makedirs(os.path.dirname(INFERENCE_OUTPUT_PATH), exist_ok=True)
    inference_predictions.to_csv(INFERENCE_OUTPUT_PATH, index=False)

    high_risk = int((inference_predictions['churn_probability'] > 0.7).sum())
    print(f"Saved {len(inference_predictions)} predictions ({high_risk} high-risk)")

    return {
        'rows_saved': len(inference_predictions),
        'high_risk_count': high_risk,
        'output_path': INFERENCE_OUTPUT_PATH,
        'saved_at': datetime.now().isoformat()
    }


# =============================================================================
# DAGSTER DEFINITIONS
# This registers all assets with Dagster
# =============================================================================

# Training assets (data prep → train → evaluate → register)
training_assets = [
    customer_data,
    customer_features,
    churn_model,
    model_metrics,
    customer_predictions,
    saved_predictions,
    registered_model,
]

# Inference assets (load from registry → predict → save)
inference_assets = [
    inference_model,
    inference_customer_data,
    inference_features,
    inference_predictions,
    saved_inference_predictions,
]

# All assets
all_assets = training_assets + inference_assets

defs = Definitions(assets=all_assets)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def run_training():
    """Materialize training assets (train, evaluate, register)."""
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    result = materialize(training_assets)

    if result.success:
        metrics_result = result.output_for_node("model_metrics")
        registered = result.output_for_node("registered_model")

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Model Performance:")
        print(f"  Accuracy:  {metrics_result['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics_result['f1']:.4f}")
        print(f"\nModel registered: {registered['model_name']} v{registered['model_version']}")
    else:
        print("Training failed!")

    return result


def run_inference():
    """Materialize inference assets (load from registry, predict)."""
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)

    result = materialize(inference_assets)

    if result.success:
        saved = result.output_for_node("saved_inference_predictions")
        model_info = result.output_for_node("inference_model")

        print("\n" + "=" * 60)
        print("Inference completed!")
        print("=" * 60)
        print(f"Model used: {model_info['model_name']} v{model_info['model_version']}")
        print(f"Predictions saved: {saved['rows_saved']}")
        print(f"High-risk customers: {saved['high_risk_count']}")
        print(f"Output file: {saved['output_path']}")
    else:
        print("Inference failed!")

    return result


def run_full():
    """Materialize all assets (training + inference)."""
    print("=" * 60)
    print("FULL PIPELINE (Training + Inference)")
    print("=" * 60)

    result = materialize(all_assets)

    if result.success:
        metrics_result = result.output_for_node("model_metrics")
        registered = result.output_for_node("registered_model")
        saved = result.output_for_node("saved_inference_predictions")

        print("\n" + "=" * 60)
        print("Full pipeline completed!")
        print("=" * 60)
        print("\nTRAINING RESULTS:")
        print(f"  Accuracy:  {metrics_result['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics_result['f1']:.4f}")
        print(f"  Model:     {registered['model_name']} v{registered['model_version']}")
        print("\nINFERENCE RESULTS:")
        print(f"  Predictions: {saved['rows_saved']}")
        print(f"  High-risk:   {saved['high_risk_count']}")
        print(f"  Output:      {saved['output_path']}")
    else:
        print("Pipeline failed!")

    return result


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Customer Churn Prediction Pipeline (Dagster)")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "train":
        run_training()
    elif mode == "inference":
        run_inference()
    elif mode == "full":
        run_full()
    else:
        print(f"\nUnknown mode: {mode}")
        print("\nUsage:")
        print("  python Dagster_ML_Pipeline.py           # Full pipeline")
        print("  python Dagster_ML_Pipeline.py train     # Training only")
        print("  python Dagster_ML_Pipeline.py inference # Inference only")
        print("\nWith Dagster UI (recommended):")
        print("  dagster dev -f Dagster_ML_Pipeline.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"View experiments at: {MLFLOW_TRACKING_URI}")
    print("Launch Dagster UI: dagster dev -f Dagster_ML_Pipeline.py")
    print("=" * 60)
    print("\nUsage:")
    print("  python Dagster_ML_Pipeline.py           # Full pipeline")
    print("  python Dagster_ML_Pipeline.py train     # Training only")
    print("  python Dagster_ML_Pipeline.py inference # Inference only")
