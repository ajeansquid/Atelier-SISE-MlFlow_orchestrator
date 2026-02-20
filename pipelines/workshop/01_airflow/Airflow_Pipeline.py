# =============================================================================
# Pipeline ML Airflow - Prédiction du Churn Client
# =============================================================================
#
# Ce pipeline démontre l'approche centrée sur les tâches d'Airflow :
# - DAG (Directed Acyclic Graph) comme citoyen de première classe
# - PythonOperator pour les définitions de tâches
# - XCom pour la communication inter-tâches (limité aux petites données)
# - E/S fichier requises pour les DataFrames (trop volumineux pour XCom)
# - Nettoyage manuel des fichiers temporaires
# - **context pour accéder aux métadonnées Airflow
#
# NOTE : Ce fichier est à des fins de démonstration. En production, vous
# placeriez ce fichier dans votre dossier DAGs d'Airflow.
#
# Points de friction clés montrés :
# - XCom push/pull partout
# - Obligation de sauvegarder les DataFrames sur disque entre les tâches
# - Gestion manuelle des fichiers temporaires
# - Définitions de tâches verbeuses avec PythonOperator
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
# Par défaut, utilise le serveur MLflow Docker (démarrer avec : cd docker && docker-compose up -d)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer-churn-airflow"
MODEL_NAME = "churn-predictor-airflow"  # Pour le registre de modèles

# Obtenir la racine du projet (parent de pipelines/workshop/01_airflow/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_airflow.csv")
INFERENCE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_airflow_inference.csv")
RANDOM_SEED = 42

# Colonnes de features utilisées pour l'entraînement
FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'order_frequency',
    'support_per_order', 'rfm_score'
]


# =============================================================================
# FONCTIONS DE TÂCHES
# Note : Chaque fonction nécessite **context et doit utiliser XCom pour la communication
# =============================================================================

def load_customer_data(**context):
    """
    Charger les données clients.

    POINT DE FRICTION AIRFLOW : Impossible de retourner les DataFrames directement !
    Doit sauvegarder sur disque et passer le chemin du fichier via XCom.
    """
    print("Chargement des données clients...")

    # Essayer de charger depuis CSV d'abord
    if os.path.exists(DATA_PATH):
        print(f"Chargement depuis {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        # Générer des données synthétiques (identique aux notebooks)
        print("Génération de données clients synthétiques...")
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

        # Créer la variable cible (churn)
        churn_prob = 1 / (1 + np.exp(-(
            0.02 * df['recency_days'] -
            0.1 * df['frequency'] -
            0.001 * df['monetary_value'] +
            0.2 * df['support_tickets'] -
            2
        )))
        df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

    print(f"{len(df)} clients chargés")

    # ❌ AIRFLOW : Impossible de retourner le DataFrame - doit sauvegarder sur disque !
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"customer_data_{context['run_id']}.parquet")
    df.to_parquet(temp_path)

    # ❌ AIRFLOW : Doit utiliser XCom pour passer les données entre les tâches
    context['ti'].xcom_push(key='data_path', value=temp_path)
    context['ti'].xcom_push(key='row_count', value=len(df))


def engineer_features(**context):
    """
    Créer des features supplémentaires.

    POINT DE FRICTION AIRFLOW : Doit lire depuis le disque, puis réécrire sur disque !
    """
    print("Ingénierie des features...")

    # ❌ AIRFLOW : Récupérer le chemin du fichier depuis XCom
    ti = context['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')

    # ❌ AIRFLOW : Charger depuis le disque
    df = pd.read_parquet(data_path)

    # Ingénierie des features (même logique que les autres pipelines)
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    # Score RFM
    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Features créées. Shape : {df.shape}")

    # ❌ AIRFLOW : Sauvegarder dans un nouveau fichier temporaire
    temp_dir = tempfile.gettempdir()
    features_path = os.path.join(temp_dir, f"features_{context['run_id']}.parquet")
    df.to_parquet(features_path)

    # ❌ AIRFLOW : Pousser le chemin via XCom
    ti.xcom_push(key='features_path', value=features_path)


def train_model(**context):
    """
    Entraîner le classifieur Random Forest avec tracking MLflow.

    POINT DE FRICTION AIRFLOW : Doit sérialiser (pickle) le modèle sur disque !
    """
    print("Entraînement du modèle...")

    ti = context['ti']
    features_path = ti.xcom_pull(key='features_path', task_ids='engineer_features')

    # ❌ AIRFLOW : Charger les features depuis le disque
    df = pd.read_parquet(features_path)

    # Préparer les données
    X = df[FEATURE_COLS]
    y = df['churned']

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"airflow-{context['ds']}") as run:
        # Hyperparamètres du modèle
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 20,
            "random_state": RANDOM_SEED
        }

        # Entraîner le modèle
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        # Logger dans MLflow
        mlflow.log_params(params)
        mlflow.log_param("orchestrator", "airflow")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("training_samples", len(X_train))

        # Logger l'artefact du modèle
        mlflow.sklearn.log_model(model, "model")

        mlflow_run_id = run.info.run_id

    # ❌ AIRFLOW : Sauvegarder le modèle dans un fichier temp (impossible via XCom - trop volumineux)
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"model_{context['run_id']}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # ❌ AIRFLOW : Sauvegarder aussi les données de test pour l'évaluation
    test_data_path = os.path.join(temp_dir, f"test_data_{context['run_id']}.parquet")
    test_df = pd.DataFrame(X_test)
    test_df['churned'] = y_test.values
    test_df.to_parquet(test_data_path)

    # ❌ AIRFLOW : Pousser tous les chemins via XCom
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='mlflow_run_id', value=mlflow_run_id)
    ti.xcom_push(key='test_data_path', value=test_data_path)

    print(f"Modèle entraîné. MLflow run : {mlflow_run_id}")


def evaluate_model(**context):
    """
    Évaluer les performances du modèle et logger les métriques dans MLflow.
    """
    print("Évaluation du modèle...")

    ti = context['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    test_data_path = ti.xcom_pull(key='test_data_path', task_ids='train_model')
    mlflow_run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')

    # ❌ AIRFLOW : Charger le modèle depuis le disque
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ❌ AIRFLOW : Charger les données de test depuis le disque
    test_df = pd.read_parquet(test_data_path)
    y_test = test_df['churned']
    X_test = test_df.drop(columns=['churned'])

    # Prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
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

    # Logger les métriques dans MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(metrics)

    # ❌ AIRFLOW : Passer les métriques via XCom
    ti.xcom_push(key='accuracy', value=metrics['accuracy'])
    ti.xcom_push(key='precision', value=metrics['precision'])
    ti.xcom_push(key='recall', value=metrics['recall'])
    ti.xcom_push(key='f1', value=metrics['f1'])


def generate_predictions(**context):
    """
    Générer les prédictions de churn pour tous les clients.
    """
    print("Génération des prédictions...")

    ti = context['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    features_path = ti.xcom_pull(key='features_path', task_ids='engineer_features')

    # ❌ AIRFLOW : Charger le modèle depuis le disque
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # ❌ AIRFLOW : Charger les features depuis le disque
    df = pd.read_parquet(features_path)

    X = df[FEATURE_COLS]

    # Générer les prédictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Créer le dataframe des résultats
    predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    # ❌ AIRFLOW : Sauvegarder les prédictions dans un fichier temporaire
    temp_dir = tempfile.gettempdir()
    predictions_path = os.path.join(temp_dir, f"predictions_{context['run_id']}.parquet")
    predictions.to_parquet(predictions_path)

    ti.xcom_push(key='predictions_path', value=predictions_path)
    ti.xcom_push(key='prediction_count', value=len(predictions))

    print(f"Prédictions générées pour {len(predictions)} clients")


def save_predictions(**context):
    """
    Sauvegarder les prédictions dans un fichier CSV local.
    """
    print(f"Sauvegarde des prédictions vers {OUTPUT_PATH}...")

    ti = context['ti']
    predictions_path = ti.xcom_pull(key='predictions_path', task_ids='generate_predictions')

    # ❌ AIRFLOW : Charger les prédictions depuis le disque
    predictions = pd.read_parquet(predictions_path)

    # S'assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Sauvegarder en CSV
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"{len(predictions)} prédictions sauvegardées vers {OUTPUT_PATH}")


def register_model(**context):
    """
    Enregistrer le modèle entraîné dans le MLflow Model Registry.

    Cela permet au pipeline d'inférence de charger par nom.
    """
    print(f"Enregistrement du modèle dans le registre : {MODEL_NAME}")

    ti = context['ti']
    mlflow_run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')
    accuracy = ti.xcom_pull(key='accuracy', task_ids='evaluate_model')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{mlflow_run_id}/model"

    # Enregistrer le modèle
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"Enregistré {MODEL_NAME} version {model_version.version}")
    print(f"Accuracy: {accuracy:.4f}")

    ti.xcom_push(key='model_version', value=model_version.version)


def cleanup_temp_files(**context):
    """
    Nettoyer les fichiers temporaires.

    POINT DE FRICTION AIRFLOW : Doit nettoyer manuellement tous les fichiers temporaires !
    """
    print("Nettoyage des fichiers temporaires...")

    ti = context['ti']

    # Récupérer tous les chemins des fichiers temporaires du pipeline d'entraînement
    paths_to_delete = [
        ti.xcom_pull(key='data_path', task_ids='load_data'),
        ti.xcom_pull(key='features_path', task_ids='engineer_features'),
        ti.xcom_pull(key='model_path', task_ids='train_model'),
        ti.xcom_pull(key='test_data_path', task_ids='train_model'),
        ti.xcom_pull(key='predictions_path', task_ids='generate_predictions')
    ]

    # Supprimer les fichiers
    for path in paths_to_delete:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"Supprimé {path}")

    print("Nettoyage terminé !")


# =============================================================================
# FONCTIONS DE TÂCHES D'INFÉRENCE
# =============================================================================

def load_inference_data(**context):
    """
    Charger les données clients pour l'inférence.

    En production, cela chargerait les NOUVEAUX clients à scorer.
    """
    print("Chargement des données clients pour l'inférence...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Générer des données synthétiques
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

    print(f"{len(df)} clients chargés pour l'inférence")

    # ❌ AIRFLOW : Sauvegarder sur disque
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"inference_data_{context['run_id']}.parquet")
    df.to_parquet(temp_path)

    context['ti'].xcom_push(key='inference_data_path', value=temp_path)
    context['ti'].xcom_push(key='inference_row_count', value=len(df))


def engineer_inference_features(**context):
    """
    Créer les features pour les données d'inférence.
    """
    print("Ingénierie des features pour l'inférence...")

    ti = context['ti']
    data_path = ti.xcom_pull(key='inference_data_path', task_ids='load_inference_data')

    df = pd.read_parquet(data_path)

    # Même ingénierie des features que l'entraînement
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Shape des features d'inférence : {df.shape}")

    # ❌ AIRFLOW : Sauvegarder sur disque
    temp_dir = tempfile.gettempdir()
    features_path = os.path.join(temp_dir, f"inference_features_{context['run_id']}.parquet")
    df.to_parquet(features_path)

    ti.xcom_push(key='inference_features_path', value=features_path)


def load_model_from_registry(**context):
    """
    Charger le modèle depuis le MLflow Model Registry.

    C'est le pattern de production - charger par nom, pas par run_id.
    """
    print(f"Chargement du modèle {MODEL_NAME}/latest depuis le registre...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        # Récupérer les infos de version
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        version = latest_version.version
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger le modèle '{MODEL_NAME}' depuis le registre. "
            f"Exécutez d'abord le pipeline d'entraînement. Erreur : {e}"
        )

    print(f"Modèle version {version} chargé")

    # ❌ AIRFLOW : Sauvegarder le modèle sur disque
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"inference_model_{context['run_id']}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    context['ti'].xcom_push(key='inference_model_path', value=model_path)
    context['ti'].xcom_push(key='loaded_model_version', value=version)


def run_batch_inference(**context):
    """
    Exécuter l'inférence en batch avec le modèle chargé.
    """
    print("Exécution de l'inférence en batch...")

    ti = context['ti']
    model_path = ti.xcom_pull(key='inference_model_path', task_ids='load_model_from_registry')
    features_path = ti.xcom_pull(key='inference_features_path', task_ids='engineer_inference_features')
    model_version = ti.xcom_pull(key='loaded_model_version', task_ids='load_model_from_registry')

    # Charger le modèle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Charger les features
    df = pd.read_parquet(features_path)
    X = df[FEATURE_COLS]

    # Générer les prédictions
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
    print(f"{len(predictions)} prédictions générées ({high_risk} à haut risque)")

    # ❌ AIRFLOW : Sauvegarder sur disque
    temp_dir = tempfile.gettempdir()
    predictions_path = os.path.join(temp_dir, f"inference_predictions_{context['run_id']}.parquet")
    predictions.to_parquet(predictions_path)

    ti.xcom_push(key='inference_predictions_path', value=predictions_path)
    ti.xcom_push(key='inference_prediction_count', value=len(predictions))
    ti.xcom_push(key='inference_high_risk_count', value=int(high_risk))


def save_inference_predictions(**context):
    """
    Sauvegarder les prédictions d'inférence en CSV.
    """
    print(f"Sauvegarde des prédictions d'inférence vers {INFERENCE_OUTPUT_PATH}...")

    ti = context['ti']
    predictions_path = ti.xcom_pull(key='inference_predictions_path', task_ids='run_batch_inference')

    predictions = pd.read_parquet(predictions_path)

    os.makedirs(os.path.dirname(INFERENCE_OUTPUT_PATH), exist_ok=True)
    predictions.to_csv(INFERENCE_OUTPUT_PATH, index=False)

    print(f"{len(predictions)} prédictions sauvegardées")

    ti.xcom_push(key='inference_output_path', value=INFERENCE_OUTPUT_PATH)


def cleanup_inference_temp_files(**context):
    """
    Nettoyer les fichiers temporaires d'inférence.
    """
    print("Nettoyage des fichiers temporaires d'inférence...")

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
            print(f"Supprimé {path}")

    print("Nettoyage d'inférence terminé !")


# =============================================================================
# DÉFINITION DU DAG
# Notez toute la configuration boilerplate requise
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
    description='Pipeline quotidien de prédiction du churn client',
    schedule_interval='0 2 * * *',  # 2h du matin quotidiennement
    catchup=False,
    tags=['ml', 'churn', 'prediction'],
)

# =============================================================================
# DÉFINITIONS DES TÂCHES
# Note : Doit encapsuler chaque fonction dans un PythonOperator
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
    trigger_rule='all_done',  # Exécuter même si les tâches précédentes échouent
    dag=dag,
)

# =============================================================================
# DÉPENDANCES DES TÂCHES - DAG D'ENTRAÎNEMENT
# =============================================================================

# Définir l'ordre d'exécution
load_data_task >> engineer_features_task >> train_model_task >> evaluate_model_task
evaluate_model_task >> register_model_task
train_model_task >> generate_predictions_task >> save_predictions_task
[register_model_task, save_predictions_task] >> cleanup_task


# =============================================================================
# DÉFINITION DU DAG D'INFÉRENCE
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
    description='Pipeline quotidien d\'inférence du churn client (charge le modèle depuis le registre)',
    schedule_interval='0 6 * * *',  # 6h du matin quotidiennement (après l'entraînement à 2h)
    catchup=False,
    tags=['ml', 'churn', 'inference'],
)

# =============================================================================
# DÉFINITIONS DES TÂCHES D'INFÉRENCE
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
# DÉPENDANCES DES TÂCHES - DAG D'INFÉRENCE
# =============================================================================

# La préparation des données et le chargement du modèle peuvent s'exécuter en parallèle
load_inference_data_task >> engineer_inference_features_task
[engineer_inference_features_task, load_model_task] >> run_inference_task
run_inference_task >> save_inference_task >> inference_cleanup_task


# =============================================================================
# EXÉCUTION STANDALONE (pour tester sans Airflow)
# =============================================================================

if __name__ == "__main__":
    import sys
    import uuid

    print("=" * 60)
    print("Pipeline de Prédiction du Churn Client (Airflow)")
    print("=" * 60)
    print("\nNOTE : Ce fichier est conçu pour s'exécuter comme un DAG Airflow.")
    print("Pour les tests standalone, nous simulons l'exécution du pipeline.\n")

    # Créer un contexte mock pour les tests standalone
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
        print("Mode : ENTRAÎNEMENT UNIQUEMENT")
        print("=" * 60)

        print("1. Chargement des données...")
        load_customer_data(**mock_context)

        print("\n2. Ingénierie des features...")
        engineer_features(**mock_context)

        print("\n3. Entraînement du modèle...")
        train_model(**mock_context)

        print("\n4. Évaluation du modèle...")
        evaluate_model(**mock_context)

        print("\n5. Enregistrement du modèle...")
        register_model(**mock_context)

        print("\n6. Génération des prédictions...")
        generate_predictions(**mock_context)

        print("\n7. Sauvegarde des prédictions...")
        save_predictions(**mock_context)

        print("\n8. Nettoyage...")
        cleanup_temp_files(**mock_context)

        print("\n" + "=" * 60)
        print("Entraînement terminé !")
        print("=" * 60)
        print(f"Prédictions sauvegardées vers : {OUTPUT_PATH}")
        print(f"MLflow run ID : {ti.xcom_data.get('mlflow_run_id', 'N/A')}")
        print(f"Version du modèle : {ti.xcom_data.get('model_version', 'N/A')}")
        print(f"Accuracy : {ti.xcom_data.get('accuracy', 0):.4f}")

    elif mode == "inference":
        print("Mode : INFÉRENCE UNIQUEMENT")
        print("=" * 60)

        # Utiliser un nouveau contexte pour l'inférence
        inference_ti = MockTaskInstance()
        inference_context = {
            'ti': inference_ti,
            'run_id': run_id + "_inf",
            'ds': datetime.now().strftime('%Y-%m-%d')
        }

        print("1. Chargement des données d'inférence...")
        load_inference_data(**inference_context)

        print("\n2. Ingénierie des features d'inférence...")
        engineer_inference_features(**inference_context)

        print("\n3. Chargement du modèle depuis le registre...")
        load_model_from_registry(**inference_context)

        print("\n4. Exécution de l'inférence en batch...")
        run_batch_inference(**inference_context)

        print("\n5. Sauvegarde des prédictions d'inférence...")
        save_inference_predictions(**inference_context)

        print("\n6. Nettoyage...")
        cleanup_inference_temp_files(**inference_context)

        print("\n" + "=" * 60)
        print("Inférence terminée !")
        print("=" * 60)
        print(f"Prédictions sauvegardées vers : {INFERENCE_OUTPUT_PATH}")
        print(f"Version du modèle : {inference_ti.xcom_data.get('loaded_model_version', 'N/A')}")
        print(f"Prédictions : {inference_ti.xcom_data.get('inference_prediction_count', 0)}")
        print(f"Haut risque : {inference_ti.xcom_data.get('inference_high_risk_count', 0)}")

    else:  # full
        print("Mode : PIPELINE COMPLET (entraînement + inférence)")
        print("=" * 60)

        # Phase d'entraînement
        print("\n" + "=" * 60)
        print("PHASE 1 : ENTRAÎNEMENT")
        print("=" * 60)

        print("1. Chargement des données...")
        load_customer_data(**mock_context)

        print("\n2. Ingénierie des features...")
        engineer_features(**mock_context)

        print("\n3. Entraînement du modèle...")
        train_model(**mock_context)

        print("\n4. Évaluation du modèle...")
        evaluate_model(**mock_context)

        print("\n5. Enregistrement du modèle...")
        register_model(**mock_context)

        print("\n6. Génération des prédictions...")
        generate_predictions(**mock_context)

        print("\n7. Sauvegarde des prédictions...")
        save_predictions(**mock_context)

        print("\n8. Nettoyage des fichiers d'entraînement...")
        cleanup_temp_files(**mock_context)

        # Phase d'inférence
        print("\n" + "=" * 60)
        print("PHASE 2 : INFÉRENCE")
        print("=" * 60)

        inference_ti = MockTaskInstance()
        inference_context = {
            'ti': inference_ti,
            'run_id': run_id + "_inf",
            'ds': datetime.now().strftime('%Y-%m-%d')
        }

        print("1. Chargement des données d'inférence...")
        load_inference_data(**inference_context)

        print("\n2. Ingénierie des features d'inférence...")
        engineer_inference_features(**inference_context)

        print("\n3. Chargement du modèle depuis le registre...")
        load_model_from_registry(**inference_context)

        print("\n4. Exécution de l'inférence en batch...")
        run_batch_inference(**inference_context)

        print("\n5. Sauvegarde des prédictions d'inférence...")
        save_inference_predictions(**inference_context)

        print("\n6. Nettoyage des fichiers d'inférence...")
        cleanup_inference_temp_files(**inference_context)

        print("\n" + "=" * 60)
        print("Pipeline complet terminé !")
        print("=" * 60)
        print("\nRÉSULTATS D'ENTRAÎNEMENT :")
        print(f"  Prédictions : {OUTPUT_PATH}")
        print(f"  MLflow run : {ti.xcom_data.get('mlflow_run_id', 'N/A')}")
        print(f"  Version du modèle : {ti.xcom_data.get('model_version', 'N/A')}")
        print(f"  Accuracy : {ti.xcom_data.get('accuracy', 0):.4f}")
        print("\nRÉSULTATS D'INFÉRENCE :")
        print(f"  Prédictions : {INFERENCE_OUTPUT_PATH}")
        print(f"  Version du modèle : {inference_ti.xcom_data.get('loaded_model_version', 'N/A')}")
        print(f"  Clients scorés : {inference_ti.xcom_data.get('inference_prediction_count', 0)}")
        print(f"  Haut risque : {inference_ti.xcom_data.get('inference_high_risk_count', 0)}")

    print("\n" + "=" * 60)
    print(f"Voir les expériences sur : {MLFLOW_TRACKING_URI}")
    print("=" * 60)
    print("\nUtilisation :")
    print("  python Airflow_ML_Pipeline.py           # Pipeline complet")
    print("  python Airflow_ML_Pipeline.py train     # Entraînement uniquement")
    print("  python Airflow_ML_Pipeline.py inference # Inférence uniquement")
