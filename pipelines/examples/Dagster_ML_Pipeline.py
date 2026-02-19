# =============================================================================
# Pipeline ML Dagster - Prédiction du Churn Client
# =============================================================================
#
# Ce pipeline démontre l'approche centrée sur les assets de Dagster :
# - Assets (artefacts de données) comme citoyens de première classe, pas des tâches
# - Déclaratif : "quelles données devraient exister" vs "quelles tâches exécuter"
# - Inférence automatique des dépendances depuis les signatures de fonctions
# - Métadonnées riches (group_name, description, compute_kind)
# - Pensez "ce qui existe" pas "ce qui s'exécute"
#
# Exécuter localement :
#   dagster dev -f Dagster_ML_Pipeline.py
#
# Ou matérialiser les assets :
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
# Par défaut, utilise le serveur MLflow Docker (démarrer avec : cd docker && docker-compose up -d)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "customer-churn-dagster"
MODEL_NAME = "churn-predictor-dagster"  # Pour le registre de modèles

# Obtenir la racine du projet (parent de pipelines/examples/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_dagster.csv")
INFERENCE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "predictions_dagster_inference.csv")
RANDOM_SEED = 42

# Colonnes de features utilisées pour l'entraînement
FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age',
    'recency_frequency_ratio', 'monetary_per_order', 'order_frequency',
    'support_per_order', 'rfm_score'
]


# =============================================================================
# ASSETS - Pensez "quelles données devraient exister", pas "quelles tâches exécuter"
# =============================================================================

@asset(
    group_name="data_ingestion",
    description="Données clients brutes chargées depuis CSV ou générées synthétiquement",
    compute_kind="pandas"
)
def customer_data() -> pd.DataFrame:
    """
    Charger ou générer les données clients.

    CONCEPT CLÉ DAGSTER : C'est un ASSET - une donnée qui existe.
    Dagster pense en termes d'artefacts de données, pas de tâches.
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
    return df  # ✅ Retourner l'asset de données


@asset(
    group_name="feature_engineering",
    description="Features créées incluant les scores RFM et les ratios",
    compute_kind="pandas"
)
def customer_features(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Créer des features à partir des données clients brutes.

    CONCEPT CLÉ DAGSTER : Les dépendances sont inférées depuis la signature de fonction !
    Dagster sait automatiquement que cet asset dépend de `customer_data`.
    """
    print("Ingénierie des features...")
    df = customer_data.copy()

    # Features de ratio
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    # Score RFM (Récence, Fréquence, Montant)
    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Features créées. Shape : {df.shape}")
    return df  # ✅ Retourner l'asset de features


@asset(
    group_name="model_training",
    description="Modèle Random Forest entraîné avec tracking MLflow",
    compute_kind="sklearn"
)
def churn_model(customer_features: pd.DataFrame) -> dict:
    """
    Entraîner le classifieur Random Forest sur les features clients.

    CONCEPT CLÉ DAGSTER : Nous retournons un dict contenant le modèle et les métadonnées.
    Cela devient un asset dont d'autres assets peuvent dépendre.
    """
    print("Entraînement du modèle...")

    # Préparer les données
    X = customer_features[FEATURE_COLS]
    y = customer_features['churned']

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"dagster-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
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
        mlflow.log_param("orchestrator", "dagster")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("training_samples", len(X_train))

        # Logger l'artefact du modèle
        mlflow.sklearn.log_model(model, "model")

        mlflow_run_id = run.info.run_id

    print(f"Modèle entraîné. MLflow run : {mlflow_run_id}")

    # ✅ Retourner l'asset modèle comme dict (inclut les métadonnées)
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'mlflow_run_id': mlflow_run_id,
        'training_samples': len(X_train)
    }


@asset(
    group_name="model_evaluation",
    description="Métriques de performance du modèle (accuracy, precision, recall, F1)",
    compute_kind="sklearn"
)
def model_metrics(churn_model: dict) -> dict:
    """
    Évaluer les performances du modèle sur l'ensemble de test.

    CONCEPT CLÉ DAGSTER : Cet asset dépend de l'asset `churn_model`.
    Dagster infère cela depuis la signature de fonction.
    """
    print("Évaluation du modèle...")

    model = churn_model['model']
    X_test = churn_model['X_test']
    y_test = churn_model['y_test']
    mlflow_run_id = churn_model['mlflow_run_id']

    # Prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
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

    # Logger les métriques dans MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1", metrics['f1'])

    return metrics  # ✅ Retourner l'asset de métriques


@asset(
    group_name="predictions",
    description="Prédictions de probabilité de churn pour tous les clients",
    compute_kind="sklearn"
)
def customer_predictions(churn_model: dict, customer_features: pd.DataFrame) -> pd.DataFrame:
    """
    Générer les prédictions de churn pour tous les clients.

    CONCEPT CLÉ DAGSTER : Cet asset dépend À LA FOIS de `churn_model` ET de `customer_features`.
    Dagster construit automatiquement le graphe d'exécution correct.
    """
    print("Génération des prédictions...")

    model = churn_model['model']
    X = customer_features[FEATURE_COLS]

    # Générer les prédictions
    churn_probability = model.predict_proba(X)[:, 1]
    churn_predicted = model.predict(X)

    # Créer le dataframe des résultats
    predictions = pd.DataFrame({
        'customer_id': customer_features['customer_id'],
        'churn_probability': churn_probability,
        'churn_predicted': churn_predicted,
        'predicted_at': datetime.now()
    })

    print(f"Prédictions générées pour {len(predictions)} clients")
    return predictions  # ✅ Retourner l'asset de prédictions


@asset(
    group_name="predictions",
    description="Prédictions sauvegardées dans un fichier CSV local",
    compute_kind="csv"
)
def saved_predictions(customer_predictions: pd.DataFrame) -> dict:
    """
    Sauvegarder les prédictions dans un fichier CSV local.

    CONCEPT CLÉ DAGSTER : Même les opérations d'E/S sont des assets !
    Cet asset représente "les prédictions qui ont été persistées".
    """
    print(f"Sauvegarde des prédictions vers {OUTPUT_PATH}...")

    # S'assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Sauvegarder en CSV
    customer_predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"{len(customer_predictions)} prédictions sauvegardées vers {OUTPUT_PATH}")

    # ✅ Retourner les métadonnées sur ce qui a été sauvegardé
    return {
        'rows_saved': len(customer_predictions),
        'output_path': OUTPUT_PATH,
        'saved_at': datetime.now().isoformat()
    }


@asset(
    group_name="model_registry",
    description="Modèle enregistré dans le MLflow Model Registry",
    compute_kind="mlflow"
)
def registered_model(churn_model: dict, model_metrics: dict) -> dict:
    """
    Enregistrer le modèle entraîné dans le MLflow Model Registry.

    CONCEPT CLÉ DAGSTER : Cet asset dépend à la fois du modèle ET des métriques.
    Nous n'enregistrons que les modèles qui ont été évalués.
    """
    print(f"Enregistrement du modèle dans le registre MLflow : {MODEL_NAME}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{churn_model['mlflow_run_id']}/model"

    # Enregistrer le modèle
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"Enregistré {MODEL_NAME} version {model_version.version}")
    print(f"Accuracy du modèle : {model_metrics['accuracy']:.4f}")

    return {
        'model_name': MODEL_NAME,
        'model_version': model_version.version,
        'mlflow_run_id': churn_model['mlflow_run_id'],
        'accuracy': model_metrics['accuracy'],
        'registered_at': datetime.now().isoformat()
    }


# =============================================================================
# ASSETS D'INFÉRENCE - Charger depuis le registre et prédire
# =============================================================================

@asset(
    group_name="inference",
    description="Modèle chargé depuis le MLflow Model Registry pour l'inférence",
    compute_kind="mlflow"
)
def inference_model() -> dict:
    """
    Charger le modèle depuis le MLflow Model Registry.

    CONCEPT CLÉ DAGSTER : C'est un asset SOURCE pour l'inférence - pas de dépendances.
    Il charge un modèle déjà enregistré, découplé de l'entraînement.

    En production, l'entraînement et l'inférence sont des pipelines séparés.
    """
    print(f"Chargement du modèle {MODEL_NAME}/latest depuis le registre...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        # Récupérer les infos de version du modèle
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        version = latest_version.version
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger le modèle '{MODEL_NAME}' depuis le registre. "
            f"Exécutez d'abord le pipeline d'entraînement pour enregistrer un modèle. Erreur : {e}"
        )

    print(f"Modèle version {version} chargé")

    return {
        'model': model,
        'model_name': MODEL_NAME,
        'model_version': version,
        'loaded_at': datetime.now().isoformat()
    }


@asset(
    group_name="inference",
    description="Données clients fraîches pour l'inférence (séparées des données d'entraînement)",
    compute_kind="pandas"
)
def inference_customer_data() -> pd.DataFrame:
    """
    Charger les données clients pour l'inférence.

    CONCEPT CLÉ DAGSTER : C'est séparé des données d'entraînement.
    En production, cela chargerait les NOUVEAUX clients à scorer.
    """
    print("Chargement des données clients pour l'inférence...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Générer des données synthétiques
        np.random.seed(RANDOM_SEED + 1)  # Seed différent pour la variation
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
    return df


@asset(
    group_name="inference",
    description="Features créées pour les données d'inférence",
    compute_kind="pandas"
)
def inference_features(inference_customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Créer les features pour les données d'inférence.

    CONCEPT CLÉ DAGSTER : Même logique d'ingénierie des features que l'entraînement,
    mais appliquée aux données d'inférence (lignée d'assets séparée).
    """
    print("Ingénierie des features pour l'inférence...")
    df = inference_customer_data.copy()

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
    return df


@asset(
    group_name="inference",
    description="Prédictions de churn depuis le pipeline d'inférence",
    compute_kind="sklearn"
)
def inference_predictions(inference_model: dict, inference_features: pd.DataFrame) -> pd.DataFrame:
    """
    Générer les prédictions en utilisant le modèle du registre.

    CONCEPT CLÉ DAGSTER : Cela dépend de inference_model (du registre)
    et inference_features (calculées à la volée), pas du pipeline d'entraînement.
    """
    print("Exécution de l'inférence en batch...")

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
    print(f"{len(predictions)} prédictions générées ({high_risk} à haut risque)")

    return predictions


@asset(
    group_name="inference",
    description="Prédictions d'inférence sauvegardées en CSV",
    compute_kind="csv"
)
def saved_inference_predictions(inference_predictions: pd.DataFrame) -> dict:
    """
    Sauvegarder les prédictions d'inférence dans un fichier.

    CONCEPT CLÉ DAGSTER : Asset de sortie séparé des prédictions d'entraînement.
    """
    print(f"Sauvegarde des prédictions d'inférence vers {INFERENCE_OUTPUT_PATH}...")

    os.makedirs(os.path.dirname(INFERENCE_OUTPUT_PATH), exist_ok=True)
    inference_predictions.to_csv(INFERENCE_OUTPUT_PATH, index=False)

    high_risk = int((inference_predictions['churn_probability'] > 0.7).sum())
    print(f"{len(inference_predictions)} prédictions sauvegardées ({high_risk} à haut risque)")

    return {
        'rows_saved': len(inference_predictions),
        'high_risk_count': high_risk,
        'output_path': INFERENCE_OUTPUT_PATH,
        'saved_at': datetime.now().isoformat()
    }


# =============================================================================
# DÉFINITIONS DAGSTER
# Ceci enregistre tous les assets avec Dagster
# =============================================================================

# Assets d'entraînement (préparation données → entraînement → évaluation → enregistrement)
training_assets = [
    customer_data,
    customer_features,
    churn_model,
    model_metrics,
    customer_predictions,
    saved_predictions,
    registered_model,
]

# Assets d'inférence (charger depuis le registre → prédire → sauvegarder)
inference_assets = [
    inference_model,
    inference_customer_data,
    inference_features,
    inference_predictions,
    saved_inference_predictions,
]

# Tous les assets
all_assets = training_assets + inference_assets

defs = Definitions(assets=all_assets)


# =============================================================================
# EXÉCUTION STANDALONE
# =============================================================================

def run_training():
    """Matérialiser les assets d'entraînement (entraîner, évaluer, enregistrer)."""
    print("=" * 60)
    print("PIPELINE D'ENTRAÎNEMENT")
    print("=" * 60)

    result = materialize(training_assets)

    if result.success:
        metrics_result = result.output_for_node("model_metrics")
        registered = result.output_for_node("registered_model")

        print("\n" + "=" * 60)
        print("Entraînement terminé !")
        print("=" * 60)
        print(f"Performance du modèle :")
        print(f"  Accuracy :  {metrics_result['accuracy']:.4f}")
        print(f"  F1 Score :  {metrics_result['f1']:.4f}")
        print(f"\nModèle enregistré : {registered['model_name']} v{registered['model_version']}")
    else:
        print("Entraînement échoué !")

    return result


def run_inference():
    """Matérialiser les assets d'inférence (charger depuis le registre, prédire)."""
    print("=" * 60)
    print("PIPELINE D'INFÉRENCE")
    print("=" * 60)

    result = materialize(inference_assets)

    if result.success:
        saved = result.output_for_node("saved_inference_predictions")
        model_info = result.output_for_node("inference_model")

        print("\n" + "=" * 60)
        print("Inférence terminée !")
        print("=" * 60)
        print(f"Modèle utilisé : {model_info['model_name']} v{model_info['model_version']}")
        print(f"Prédictions sauvegardées : {saved['rows_saved']}")
        print(f"Clients à haut risque : {saved['high_risk_count']}")
        print(f"Fichier de sortie : {saved['output_path']}")
    else:
        print("Inférence échouée !")

    return result


def run_full():
    """Matérialiser tous les assets (entraînement + inférence)."""
    print("=" * 60)
    print("PIPELINE COMPLET (Entraînement + Inférence)")
    print("=" * 60)

    result = materialize(all_assets)

    if result.success:
        metrics_result = result.output_for_node("model_metrics")
        registered = result.output_for_node("registered_model")
        saved = result.output_for_node("saved_inference_predictions")

        print("\n" + "=" * 60)
        print("Pipeline complet terminé !")
        print("=" * 60)
        print("\nRÉSULTATS D'ENTRAÎNEMENT :")
        print(f"  Accuracy :   {metrics_result['accuracy']:.4f}")
        print(f"  F1 Score :   {metrics_result['f1']:.4f}")
        print(f"  Modèle :     {registered['model_name']} v{registered['model_version']}")
        print("\nRÉSULTATS D'INFÉRENCE :")
        print(f"  Prédictions :   {saved['rows_saved']}")
        print(f"  Haut risque :   {saved['high_risk_count']}")
        print(f"  Sortie :        {saved['output_path']}")
    else:
        print("Pipeline échoué !")

    return result


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Pipeline de Prédiction du Churn Client (Dagster)")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "train":
        run_training()
    elif mode == "inference":
        run_inference()
    elif mode == "full":
        run_full()
    else:
        print(f"\nMode inconnu : {mode}")
        print("\nUtilisation :")
        print("  python Dagster_ML_Pipeline.py           # Pipeline complet")
        print("  python Dagster_ML_Pipeline.py train     # Entraînement uniquement")
        print("  python Dagster_ML_Pipeline.py inference # Inférence uniquement")
        print("\nAvec l'UI Dagster (recommandé) :")
        print("  dagster dev -f Dagster_ML_Pipeline.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Voir les expériences sur : {MLFLOW_TRACKING_URI}")
    print("Lancer l'UI Dagster : dagster dev -f Dagster_ML_Pipeline.py")
    print("=" * 60)
    print("\nUtilisation :")
    print("  python Dagster_ML_Pipeline.py           # Pipeline complet")
    print("  python Dagster_ML_Pipeline.py train     # Entraînement uniquement")
    print("  python Dagster_ML_Pipeline.py inference # Inférence uniquement")
