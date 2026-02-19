# =============================================================================
# Airflow ML Pipeline - Customer Churn Prediction
# =============================================================================
#
# This pipeline demonstrates Airflow's task-centric approach:
# - DAG (Directed Acyclic Graph) as first-class citizen
# - PythonOperator for task definitions
# - XCom for inter-task communication (limited to small data)
# - File I/O required for DataFrames (too large for XCom)
# - Manual cleanup of temp files
# - **context for accessing Airflow metadata
#
# NOTE: This file is for demonstration purposes. In production, you would
# place this in your Airflow DAGs folder.
#
# Key pain points shown:
# - XCom push/pull everywhere
# - Must save DataFrames to disk between tasks
# - Manual temp file management
# - Verbose task definitions with PythonOperator
# =============================================================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import tempfile

# ---------------------------------------------------------------------------
# Configuration
# Defaults to Docker-hosted MLflow server (start with: cd docker && docker-compose up -d)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer-churn-airflow"
MODEL_NAME = "churn-predictor-airflow"  # For model registry

# Get project root (parent of pipelines/examples/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_airflow.csv")
INFERENCE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_airflow_inference.csv")
RANDOM_SEED = 42

# Feature columns used for training
FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'order_frequency',
    'support_per_order', 'rfm_score'
]


# =============================================================================
# TASK FUNCTIONS
# Notice: Every function needs **context and must use XCom for communication
# =============================================================================

def load_customer_data(**context):
    """
    Load customer data.

    AIRFLOW PAIN POINT: Can't return DataFrames directly!
    Must save to disk and pass file path via XCom.
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

    # ❌ AIRFLOW: Can't return DataFrame - must save to disk!
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"customer_data_{context['run_id']}.parquet")
    df.to_parquet(temp_path)

    # ❌ AIRFLOW: Must use XCom to pass data between tasks
    context['ti'].xcom_push(key='data_path', value=temp_path)
    context['ti'].xcom_push(key='row_count', value=len(df))


def engineer_features(**context):
    """
    Create additional features.

    AIRFLOW PAIN POINT: Must read from disk, then write back to disk!
    """
    print("Engineering features...")

    # ❌ AIRFLOW: Pull file path from XCom
    ti = context['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')

    # ❌ AIRFLOW: Load from disk
    df = pd.read_parquet(data_path)

    # Feature engineering (same logic as other pipelines)
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    # RFM score
    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Engineered features. Shape: {df.shape}")

    # ❌ AIRFLOW: Save to new temp file
    temp_dir = tempfile.gettempdir()
    features_path = os.path.join(temp_dir, f"features_{context['run_id']}.parquet")
    df.to_parquet(features_path)

    # ❌ AIRFLOW: Push path via XCom
    ti.xcom_push(key='features_path', value=features_path)


def train_model(**context):
    """
    Train Random Forest classifier with MLflow tracking.

    AIRFLOW PAIN POINT: Must pickle model to disk!
    """
    print("Training model...")

    ti = context['ti']
    features_path = ti.xcom_pull(key='features_path', task_ids='engineer_features')

    # ❌ AIRFLOW: Load features from disk
    df = pd.read_parquet(features_path)

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

    with mlflow.start_run(run_name=f"airflow-{context['ds']}") as run:
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
        mlflow.log_param("orchestrator", "airflow")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("training_samples", len(X_train))

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        mlflow_run_id = run.info.run_id

    # ❌ AIRFLOW: Save model to temp file (can't pass via XCom - too large)
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"model_{context['run_id']}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # ❌ AIRFLOW: Also save test data for evaluation
    test_data_path = os.path.join(temp_dir, f"test_data_{context['run_id']}.parquet")
    test_df = pd.DataFrame(X_test)
    test_df['churned'] = y_test.values
    test_df.to_parquet(test_data_path)

    # ❌ AIRFLOW: Push all paths via XCom
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='mlflow_run_id', value=mlflow_run_id)
    ti.xcom_push(key='test_data_path', value=test_data_path)

    print(f"Model trained. MLflow run: {mlflow_run_id}")


def evaluate_model(**context):
    """
    Evaluate model performance and log metrics to MLflow.
    """
    print("Evaluating model...")

    ti = context['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    test_data_path = ti.xcom_pull(key='test_data_path', task_ids='train_model')
    mlflow_run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')

    # ❌ AIRFLOW: Load model from disk
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ❌ AIRFLOW: Load test data from disk
    test_df = pd.read_parquet(test_data_path)
    y_test = test_df['churned']
    X_test = test_df.drop(columns=['churned'])

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

    # ❌ AIRFLOW: Pass metrics via XCom
    ti.xcom_push(key='accuracy', value=metrics['accuracy'])
    ti.xcom_push(key='precision', value=metrics['precision'])
    ti.xcom_push(key='recall', value=metrics['recall'])
    ti.xcom_push(key='f1', value=metrics['f1'])


def generate_predictions(**context):
    """
    Generate churn predictions for all customers.
    """
    print("Generating predictions...")

    ti = context['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    features_path = ti.xcom_pull(key='features_path', task_ids='engineer_features')

    # ❌ AIRFLOW: Load model from disk
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ❌ AIRFLOW: Load features from disk
    df = pd.read_parquet(features_path)

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

    # ❌ AIRFLOW: Save predictions to temp file
    temp_dir = tempfile.gettempdir()
    predictions_path = os.path.join(temp_dir, f"predictions_{context['run_id']}.parquet")
    predictions.to_parquet(predictions_path)

    ti.xcom_push(key='predictions_path', value=predictions_path)
    ti.xcom_push(key='prediction_count', value=len(predictions))

    print(f"Generated predictions for {len(predictions)} customers")


def save_predictions(**context):
    """
    Save predictions to local CSV file.
    """
    print(f"Saving predictions to {OUTPUT_PATH}...")

    ti = context['ti']
    predictions_path = ti.xcom_pull(key='predictions_path', task_ids='generate_predictions')

    # ❌ AIRFLOW: Load predictions from disk
    predictions = pd.read_parquet(predictions_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save to CSV
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(predictions)} predictions to {OUTPUT_PATH}")


def register_model(**context):
    """
    Register trained model to MLflow Model Registry.

    This allows the inference pipeline to load by name.
    """
    print(f"Registering model to registry: {MODEL_NAME}")

    ti = context['ti']
    mlflow_run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')
    accuracy = ti.xcom_pull(key='accuracy', task_ids='evaluate_model')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{mlflow_run_id}/model"

    # Register model
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"Registered {MODEL_NAME} version {model_version.version}")
    print(f"Accuracy: {accuracy:.4f}")

    ti.xcom_push(key='model_version', value=model_version.version)


def cleanup_temp_files(**context):
    """
    Clean up temporary files.

    AIRFLOW PAIN POINT: Must manually clean up all temp files!
    """
    print("Cleaning up temporary files...")

    ti = context['ti']

    # Get all temp file paths from training pipeline
    paths_to_delete = [
        ti.xcom_pull(key='data_path', task_ids='load_data'),
        ti.xcom_pull(key='features_path', task_ids='engineer_features'),
        ti.xcom_pull(key='model_path', task_ids='train_model'),
        ti.xcom_pull(key='test_data_path', task_ids='train_model'),
        ti.xcom_pull(key='predictions_path', task_ids='generate_predictions')
    ]

    # Delete files
    for path in paths_to_delete:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"Deleted {path}")

    print("Cleanup complete!")


# =============================================================================
# INFERENCE TASK FUNCTIONS
# =============================================================================

def load_inference_data(**context):
    """
    Load customer data for inference.

    In production, this would load NEW customers to score.
    """
    print("Loading customer data for inference...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Generate synthetic data
        np.random.seed(RANDOM_SEED + 1)
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

    # ❌ AIRFLOW: Save to disk
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"inference_data_{context['run_id']}.parquet")
    df.to_parquet(temp_path)

    context['ti'].xcom_push(key='inference_data_path', value=temp_path)
    context['ti'].xcom_push(key='inference_row_count', value=len(df))


def engineer_inference_features(**context):
    """
    Engineer features for inference data.
    """
    print("Engineering features for inference...")

    ti = context['ti']
    data_path = ti.xcom_pull(key='inference_data_path', task_ids='load_inference_data')

    df = pd.read_parquet(data_path)

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

    # ❌ AIRFLOW: Save to disk
    temp_dir = tempfile.gettempdir()
    features_path = os.path.join(temp_dir, f"inference_features_{context['run_id']}.parquet")
    df.to_parquet(features_path)

    ti.xcom_push(key='inference_features_path', value=features_path)


def load_model_from_registry(**context):
    """
    Load model from MLflow Model Registry.

    This is the production pattern - load by name, not by run_id.
    """
    print(f"Loading model {MODEL_NAME}/latest from registry...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        # Get version info
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        version = latest_version.version
    except Exception as e:
        raise RuntimeError(
            f"Could not load model '{MODEL_NAME}' from registry. "
            f"Run training pipeline first. Error: {e}"
        )

    print(f"Loaded model version {version}")

    # ❌ AIRFLOW: Save model to disk
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"inference_model_{context['run_id']}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    context['ti'].xcom_push(key='inference_model_path', value=model_path)
    context['ti'].xcom_push(key='loaded_model_version', value=version)


def run_batch_inference(**context):
    """
    Run batch inference using loaded model.
    """
    print("Running batch inference...")

    ti = context['ti']
    model_path = ti.xcom_pull(key='inference_model_path', task_ids='load_model_from_registry')
    features_path = ti.xcom_pull(key='inference_features_path', task_ids='engineer_inference_features')
    model_version = ti.xcom_pull(key='loaded_model_version', task_ids='load_model_from_registry')

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load features
    df = pd.read_parquet(features_path)
    X = df[FEATURE_COLS]

    # Generate predictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'model_version': model_version,
        'predicted_at': datetime.now()
    })

    high_risk = (churn_probability > 0.7).sum()
    print(f"Generated {len(predictions)} predictions ({high_risk} high-risk)")

    # ❌ AIRFLOW: Save to disk
    temp_dir = tempfile.gettempdir()
    predictions_path = os.path.join(temp_dir, f"inference_predictions_{context['run_id']}.parquet")
    predictions.to_parquet(predictions_path)

    ti.xcom_push(key='inference_predictions_path', value=predictions_path)
    ti.xcom_push(key='inference_prediction_count', value=len(predictions))
    ti.xcom_push(key='inference_high_risk_count', value=int(high_risk))


def save_inference_predictions(**context):
    """
    Save inference predictions to CSV.
    """
    print(f"Saving inference predictions to {INFERENCE_OUTPUT_PATH}...")

    ti = context['ti']
    predictions_path = ti.xcom_pull(key='inference_predictions_path', task_ids='run_batch_inference')

    predictions = pd.read_parquet(predictions_path)

    os.makedirs(os.path.dirname(INFERENCE_OUTPUT_PATH), exist_ok=True)
    predictions.to_csv(INFERENCE_OUTPUT_PATH, index=False)

    print(f"Saved {len(predictions)} predictions")

    ti.xcom_push(key='inference_output_path', value=INFERENCE_OUTPUT_PATH)


def cleanup_inference_temp_files(**context):
    """
    Clean up inference temporary files.
    """
    print("Cleaning up inference temporary files...")

    ti = context['ti']

    paths_to_delete = [
        ti.xcom_pull(key='inference_data_path', task_ids='load_inference_data'),
        ti.xcom_pull(key='inference_features_path', task_ids='engineer_inference_features'),
        ti.xcom_pull(key='inference_model_path', task_ids='load_model_from_registry'),
        ti.xcom_pull(key='inference_predictions_path', task_ids='run_batch_inference')
    ]

    for path in paths_to_delete:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"Deleted {path}")

    print("Inference cleanup complete!")


# =============================================================================
# DAG DEFINITION
# Notice all the boilerplate configuration required
# =============================================================================

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'customer_churn_prediction',
    default_args=default_args,
    description='Daily customer churn prediction pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
    tags=['ml', 'churn', 'prediction'],
)

# =============================================================================
# TASK DEFINITIONS
# Notice: Must wrap each function in PythonOperator
# =============================================================================

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_customer_data,
    dag=dag,
)

engineer_features_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

generate_predictions_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag,
)

save_predictions_task = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions,
    dag=dag,
)

register_model_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_temp_files,
    trigger_rule='all_done',  # Run even if previous tasks fail
    dag=dag,
)

# =============================================================================
# TASK DEPENDENCIES - TRAINING DAG
# =============================================================================

# Define the execution order
load_data_task >> engineer_features_task >> train_model_task >> evaluate_model_task
evaluate_model_task >> register_model_task
train_model_task >> generate_predictions_task >> save_predictions_task
[register_model_task, save_predictions_task] >> cleanup_task


# =============================================================================
# INFERENCE DAG DEFINITION
# =============================================================================

inference_default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

inference_dag = DAG(
    'customer_churn_inference',
    default_args=inference_default_args,
    description='Daily customer churn inference pipeline (loads model from registry)',
    schedule_interval='0 6 * * *',  # 6 AM daily (after training at 2 AM)
    catchup=False,
    tags=['ml', 'churn', 'inference'],
)

# =============================================================================
# INFERENCE TASK DEFINITIONS
# =============================================================================

load_inference_data_task = PythonOperator(
    task_id='load_inference_data',
    python_callable=load_inference_data,
    dag=inference_dag,
)

engineer_inference_features_task = PythonOperator(
    task_id='engineer_inference_features',
    python_callable=engineer_inference_features,
    dag=inference_dag,
)

load_model_task = PythonOperator(
    task_id='load_model_from_registry',
    python_callable=load_model_from_registry,
    dag=inference_dag,
)

run_inference_task = PythonOperator(
    task_id='run_batch_inference',
    python_callable=run_batch_inference,
    dag=inference_dag,
)

save_inference_task = PythonOperator(
    task_id='save_inference_predictions',
    python_callable=save_inference_predictions,
    dag=inference_dag,
)

inference_cleanup_task = PythonOperator(
    task_id='cleanup_inference',
    python_callable=cleanup_inference_temp_files,
    trigger_rule='all_done',
    dag=inference_dag,
)

# =============================================================================
# TASK DEPENDENCIES - INFERENCE DAG
# =============================================================================

# Data prep and model loading can run in parallel
load_inference_data_task >> engineer_inference_features_task
[engineer_inference_features_task, load_model_task] >> run_inference_task
run_inference_task >> save_inference_task >> inference_cleanup_task


# =============================================================================
# STANDALONE EXECUTION (for testing without Airflow)
# =============================================================================

if __name__ == "__main__":
    import sys
    import uuid

    print("=" * 60)
    print("Customer Churn Prediction Pipeline (Airflow)")
    print("=" * 60)
    print("\nNOTE: This file is designed to run as an Airflow DAG.")
    print("For standalone testing, we simulate the pipeline execution.\n")

    # Create a mock context for standalone testing
    run_id = str(uuid.uuid4())[:8]

    class MockTaskInstance:
        def __init__(self):
            self.xcom_data = {}

        def xcom_push(self, key, value):
            self.xcom_data[key] = value

        def xcom_pull(self, key, task_ids):
            return self.xcom_data.get(key)

    ti = MockTaskInstance()
    mock_context = {
        'ti': ti,
        'run_id': run_id,
        'ds': datetime.now().strftime('%Y-%m-%d')
    }

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "train":
        print("Mode: TRAINING ONLY")
        print("=" * 60)

        print("1. Loading data...")
        load_customer_data(**mock_context)

        print("\n2. Engineering features...")
        engineer_features(**mock_context)

        print("\n3. Training model...")
        train_model(**mock_context)

        print("\n4. Evaluating model...")
        evaluate_model(**mock_context)

        print("\n5. Registering model...")
        register_model(**mock_context)

        print("\n6. Generating predictions...")
        generate_predictions(**mock_context)

        print("\n7. Saving predictions...")
        save_predictions(**mock_context)

        print("\n8. Cleaning up...")
        cleanup_temp_files(**mock_context)

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Predictions saved to: {OUTPUT_PATH}")
        print(f"MLflow run ID: {ti.xcom_data.get('mlflow_run_id', 'N/A')}")
        print(f"Model version: {ti.xcom_data.get('model_version', 'N/A')}")
        print(f"Accuracy: {ti.xcom_data.get('accuracy', 0):.4f}")

    elif mode == "inference":
        print("Mode: INFERENCE ONLY")
        print("=" * 60)

        # Use a fresh context for inference
        inference_ti = MockTaskInstance()
        inference_context = {
            'ti': inference_ti,
            'run_id': run_id + "_inf",
            'ds': datetime.now().strftime('%Y-%m-%d')
        }

        print("1. Loading inference data...")
        load_inference_data(**inference_context)

        print("\n2. Engineering inference features...")
        engineer_inference_features(**inference_context)

        print("\n3. Loading model from registry...")
        load_model_from_registry(**inference_context)

        print("\n4. Running batch inference...")
        run_batch_inference(**inference_context)

        print("\n5. Saving inference predictions...")
        save_inference_predictions(**inference_context)

        print("\n6. Cleaning up...")
        cleanup_inference_temp_files(**inference_context)

        print("\n" + "=" * 60)
        print("Inference completed!")
        print("=" * 60)
        print(f"Predictions saved to: {INFERENCE_OUTPUT_PATH}")
        print(f"Model version: {inference_ti.xcom_data.get('loaded_model_version', 'N/A')}")
        print(f"Predictions: {inference_ti.xcom_data.get('inference_prediction_count', 0)}")
        print(f"High-risk: {inference_ti.xcom_data.get('inference_high_risk_count', 0)}")

    else:  # full
        print("Mode: FULL PIPELINE (train + inference)")
        print("=" * 60)

        # Training phase
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING")
        print("=" * 60)

        print("1. Loading data...")
        load_customer_data(**mock_context)

        print("\n2. Engineering features...")
        engineer_features(**mock_context)

        print("\n3. Training model...")
        train_model(**mock_context)

        print("\n4. Evaluating model...")
        evaluate_model(**mock_context)

        print("\n5. Registering model...")
        register_model(**mock_context)

        print("\n6. Generating predictions...")
        generate_predictions(**mock_context)

        print("\n7. Saving predictions...")
        save_predictions(**mock_context)

        print("\n8. Cleaning up training files...")
        cleanup_temp_files(**mock_context)

        # Inference phase
        print("\n" + "=" * 60)
        print("PHASE 2: INFERENCE")
        print("=" * 60)

        inference_ti = MockTaskInstance()
        inference_context = {
            'ti': inference_ti,
            'run_id': run_id + "_inf",
            'ds': datetime.now().strftime('%Y-%m-%d')
        }

        print("1. Loading inference data...")
        load_inference_data(**inference_context)

        print("\n2. Engineering inference features...")
        engineer_inference_features(**inference_context)

        print("\n3. Loading model from registry...")
        load_model_from_registry(**inference_context)

        print("\n4. Running batch inference...")
        run_batch_inference(**inference_context)

        print("\n5. Saving inference predictions...")
        save_inference_predictions(**inference_context)

        print("\n6. Cleaning up inference files...")
        cleanup_inference_temp_files(**inference_context)

        print("\n" + "=" * 60)
        print("Full pipeline completed!")
        print("=" * 60)
        print("\nTRAINING RESULTS:")
        print(f"  Predictions: {OUTPUT_PATH}")
        print(f"  MLflow run: {ti.xcom_data.get('mlflow_run_id', 'N/A')}")
        print(f"  Model version: {ti.xcom_data.get('model_version', 'N/A')}")
        print(f"  Accuracy: {ti.xcom_data.get('accuracy', 0):.4f}")
        print("\nINFERENCE RESULTS:")
        print(f"  Predictions: {INFERENCE_OUTPUT_PATH}")
        print(f"  Model version: {inference_ti.xcom_data.get('loaded_model_version', 'N/A')}")
        print(f"  Customers scored: {inference_ti.xcom_data.get('inference_prediction_count', 0)}")
        print(f"  High-risk: {inference_ti.xcom_data.get('inference_high_risk_count', 0)}")

    print("\n" + "=" * 60)
    print(f"View experiments at: {MLFLOW_TRACKING_URI}")
    print("=" * 60)
    print("\nUsage:")
    print("  python Airflow_ML_Pipeline.py           # Full pipeline")
    print("  python Airflow_ML_Pipeline.py train     # Training only")
    print("  python Airflow_ML_Pipeline.py inference # Inference only")
