# =============================================================================
# Pipeline ML Prefect - Prédiction du Churn Client
# =============================================================================
#
# Ce pipeline démontre l'approche Pythonic de Prefect pour l'orchestration :
# - Décorateurs @task et @flow sur des fonctions Python classiques
# - Passage de données en mémoire (il suffit de retourner les valeurs !)
# - Type hints Python natifs
# - Pas de XCom, pas d'I/O fichier pour les données intermédiaires
# - Retries/caching déclaratifs sur les décorateurs
#
# Deux flows disponibles :
# 1. Pipeline d'entraînement : Entraîner le modèle, évaluer, enregistrer dans MLflow
# 2. Pipeline d'inférence : Charger depuis le registry, prédire, sauvegarder les résultats
#
# Exécuter localement : python Prefect_ML_Pipeline.py
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
# Configuration MLflow
# Par défaut : serveur MLflow hébergé dans Docker (démarrer avec : cd docker && docker-compose up -d)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer-churn-prefect"
MODEL_NAME = "churn-predictor-prefect"  # For model registry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Obtenir la racine du projet (parent de pipelines/examples/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_prefect.csv")
RANDOM_SEED = 42

# Colonnes de features utilisées pour l'entraînement
FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'order_frequency',
    'support_per_order', 'rfm_score'
]


# =============================================================================
# TASKS - Remarquez comme elles sont simples comparées à Airflow !
# =============================================================================

@task(retries=3, retry_delay_seconds=10)
def load_customer_data() -> pd.DataFrame:
    """
    Charger les données clients.

    Essaie de charger depuis le CSV si disponible, sinon génère des données synthétiques.
    Cela permet d'exécuter le workshop sans dépendances externes.
    """
    print("Chargement des données clients...")

    # Essayer de charger depuis le CSV d'abord
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

        # Créer la variable cible (churn) basée sur une logique réaliste
        churn_prob = 1 / (1 + np.exp(-(
            0.02 * df['recency_days'] -
            0.1 * df['frequency'] -
            0.001 * df['monetary_value'] +
            0.2 * df['support_tickets'] -
            2
        )))
        df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

    print(f"{len(df)} clients chargés")
    return df  # ✅ Il suffit de retourner - pas d'I/O fichier nécessaire !


@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Créer des features supplémentaires pour le modèle.

    Remarque : On reçoit un DataFrame directement en entrée et on en retourne un en sortie.
    Pas de XCom, pas de chemins de fichiers, pas de pickle - juste du Python !
    """
    print("Engineering des features...")
    df = df.copy()

    # Features de ratio
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    # Score RFM (Recency, Frequency, Monetary)
    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Features engineerées. Dimensions : {df.shape}")
    return df  # ✅ Il suffit de retourner !


@task(retries=2)
def train_model(df: pd.DataFrame) -> tuple:
    """
    Entraîner un classificateur Random Forest avec tracking MLflow.

    Retourne le modèle et les données de test pour l'évaluation.
    """
    print("Entraînement du modèle...")

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

    with mlflow.start_run(run_name=f"prefect-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
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
        mlflow.log_param("orchestrator", "prefect")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("training_samples", len(X_train))

        # Logger l'artefact du modèle
        mlflow.sklearn.log_model(model, "model")

        mlflow_run_id = run.info.run_id

    print(f"Modèle entraîné. MLflow run : {mlflow_run_id}")

    # ✅ Retourner tout ce qui est nécessaire - pas d'I/O fichier !
    return model, X_test, y_test, mlflow_run_id


@task
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, mlflow_run_id: str) -> dict:
    """
    Évaluer les performances du modèle et logger les métriques dans MLflow.
    """
    print("Évaluation du modèle...")

    # Prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    print(f"Accuracy :  {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall :    {metrics['recall']:.4f}")
    print(f"F1 Score :  {metrics['f1']:.4f}")

    # Logger les métriques dans MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(metrics)

    return metrics  # ✅ Il suffit de retourner !


@task
def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Générer des prédictions de churn pour tous les clients.
    """
    print("Génération des prédictions...")

    X = df[FEATURE_COLS]

    # Générer les prédictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Créer le dataframe de résultats
    predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    print(f"Prédictions générées pour {len(predictions)} clients")
    return predictions  # ✅ Il suffit de retourner !


@task
def save_predictions(predictions: pd.DataFrame, output_path: str = OUTPUT_PATH) -> str:
    """
    Sauvegarder les prédictions dans un fichier CSV local.

    En production, cela écrirait dans une base de données ou un data warehouse.
    """
    print(f"Sauvegarde des prédictions dans {output_path}...")

    # S'assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder en CSV
    predictions.to_csv(output_path, index=False)

    print(f"{len(predictions)} prédictions sauvegardées dans {output_path}")
    return output_path


@task
def register_model(mlflow_run_id: str, metrics: dict) -> str:
    """
    Enregistrer le modèle entraîné dans le Model Registry MLflow.

    Cela permet au pipeline d'inférence de charger par nom au lieu de run_id.
    """
    print(f"Enregistrement du modèle dans le registry : {MODEL_NAME}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{mlflow_run_id}/model"

    # Enregistrer le modèle
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"{MODEL_NAME} version {model_version.version} enregistré")
    print(f"Accuracy : {metrics.get('accuracy', 0):.4f}")

    return model_version.version


@task(retries=2)
def load_model_from_registry(model_name: str = MODEL_NAME, version: str = "latest"):
    """
    Charger le modèle depuis le Model Registry MLflow.

    C'est le pattern de production - charger par nom, pas par run_id.
    """
    print(f"Chargement du modèle {model_name}/{version} depuis le registry...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{version}"

    model = mlflow.sklearn.load_model(model_uri)

    print(f"Modèle chargé avec succès")
    return model


@task
def run_batch_inference(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Exécuter l'inférence batch sur les données clients.

    Cela simule des jobs de prédiction quotidiens.
    """
    print(f"Exécution de l'inférence sur {len(df)} clients...")

    X = df[FEATURE_COLS]

    # Générer les prédictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Créer le dataframe de résultats
    predictions = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    # Statistiques résumées
    high_risk = (churn_probability > 0.7).sum()
    print(f"Clients à haut risque (>70%) : {high_risk}")

    return predictions


# =============================================================================
# FLOW - L'orchestration principale du pipeline
# =============================================================================

@flow(name="churn-training-pipeline", log_prints=True)
def training_pipeline():
    """
    Pipeline d'entraînement : Entraîner, évaluer et enregistrer le modèle dans MLflow.

    Cela s'exécuterait sur un schedule (ex: hebdomadaire) ou quand de nouvelles données arrivent.
    """

    # Charger les données
    raw_data = load_customer_data()

    # Créer les features
    features = engineer_features(raw_data)

    # Entraîner le modèle
    model, X_test, y_test, mlflow_run_id = train_model(features)

    # Évaluer le modèle
    metrics = evaluate_model(model, X_test, y_test, mlflow_run_id)

    # Enregistrer le modèle dans le registry
    model_version = register_model(mlflow_run_id, metrics)

    # Retourner le résumé
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
    Pipeline d'inférence : Charger le modèle depuis le registry et générer des prédictions.

    Cela s'exécuterait quotidiennement sur les nouvelles données clients.
    Séparer l'entraînement de l'inférence est une bonne pratique en production !
    """

    # Charger les données
    raw_data = load_customer_data()

    # Créer les features (identique à l'entraînement)
    features = engineer_features(raw_data)

    # Charger le modèle depuis le registry (pas depuis l'entraînement !)
    model = load_model_from_registry(MODEL_NAME, model_version)

    # Exécuter l'inférence batch
    predictions = run_batch_inference(model, features)

    # Sauvegarder les prédictions
    inference_output = os.path.join(PROJECT_ROOT, "data", "predictions_inference.csv")
    output_path = save_predictions(predictions, inference_output)

    # Retourner le résumé
    return {
        "customers_scored": len(predictions),
        "high_risk_customers": int((predictions['churn_probability'] > 0.7).sum()),
        "model_version": model_version,
        "output_file": output_path
    }


@flow(name="churn-full-pipeline", log_prints=True)
def full_pipeline():
    """
    Pipeline complet : Entraîner le modèle ET exécuter l'inférence.

    Démontre le workflow ML complet en un seul flow.
    En production, ce seraient typiquement des flows schedulés séparés.
    """

    # Phase 1 : Entraînement
    print("=" * 60)
    print("PHASE 1 : ENTRAÎNEMENT")
    print("=" * 60)

    raw_data = load_customer_data()
    features = engineer_features(raw_data)
    model, X_test, y_test, mlflow_run_id = train_model(features)
    metrics = evaluate_model(model, X_test, y_test, mlflow_run_id)
    model_version = register_model(mlflow_run_id, metrics)

    # Phase 2 : Inférence (utilisant le modèle qu'on vient d'enregistrer)
    print("\n" + "=" * 60)
    print("PHASE 2 : INFÉRENCE")
    print("=" * 60)

    # Charger depuis le registry (prouve que l'enregistrement a fonctionné)
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
# MAIN - Exécuter le pipeline
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Pipeline de Prédiction du Churn Client (Prefect)")
    print("=" * 60)

    # Parser l'argument de ligne de commande
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "train":
        # Entraînement seulement
        print("Mode : ENTRAÎNEMENT SEULEMENT")
        print("=" * 60)
        result = training_pipeline()

        print("\n" + "=" * 60)
        print("Entraînement terminé !")
        print("=" * 60)
        print(f"Clients entraînés :     {result['customers_trained_on']}")
        print(f"Accuracy du modèle :    {result['model_accuracy']:.4f}")
        print(f"Score F1 du modèle :    {result['model_f1']:.4f}")
        print(f"Version du modèle :     {result['model_version']}")
        print(f"MLflow run ID :         {result['mlflow_run_id']}")

    elif mode == "inference":
        # Inférence seulement (nécessite un modèle entraîné dans le registry)
        print("Mode : INFÉRENCE SEULEMENT")
        print("=" * 60)
        version = sys.argv[2] if len(sys.argv) > 2 else "latest"
        result = inference_pipeline(model_version=version)

        print("\n" + "=" * 60)
        print("Inférence terminée !")
        print("=" * 60)
        print(f"Clients scorés :        {result['customers_scored']}")
        print(f"Clients haut risque :   {result['high_risk_customers']}")
        print(f"Version modèle utilisé : {result['model_version']}")
        print(f"Prédictions dans :      {result['output_file']}")

    else:
        # Pipeline complet (train + inference)
        print("Mode : PIPELINE COMPLET (train + inference)")
        print("=" * 60)
        result = full_pipeline()

        print("\n" + "=" * 60)
        print("Pipeline complet terminé !")
        print("=" * 60)
        print("\nRÉSULTATS ENTRAÎNEMENT :")
        print(f"  Accuracy du modèle :  {result['training']['model_accuracy']:.4f}")
        print(f"  Score F1 du modèle :  {result['training']['model_f1']:.4f}")
        print(f"  Version du modèle :   {result['training']['model_version']}")
        print(f"  MLflow run ID :       {result['training']['mlflow_run_id']}")
        print("\nRÉSULTATS INFÉRENCE :")
        print(f"  Clients scorés :      {result['inference']['customers_scored']}")
        print(f"  Clients haut risque : {result['inference']['high_risk_customers']}")
        print(f"  Prédictions dans :    {result['inference']['output_file']}")

    print("\n" + "=" * 60)
    print("Voir les expériences sur : http://localhost:5000")
    print("=" * 60)
    print("\nUtilisation :")
    print("  python Prefect_ML_Pipeline.py           # Pipeline complet")
    print("  python Prefect_ML_Pipeline.py train     # Entraînement seulement")
    print("  python Prefect_ML_Pipeline.py inference # Inférence seulement")
    print("  python Prefect_ML_Pipeline.py inference 1  # Inférence avec modèle v1")
