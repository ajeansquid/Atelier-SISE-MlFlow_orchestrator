# =============================================================================
# Atelier Dagster - BONUS : Des Tâches aux Assets
# =============================================================================
#
# Vous avez terminé l'atelier Prefect. Voyons maintenant à quoi ressemble le MÊME pipeline
# dans Dagster - et pourquoi la différence est importante.
#
# LE CHANGEMENT CLÉ :
#   Prefect : "Exécuter load_data, puis engineer_features, puis train_model..."
#   Dagster : "Je veux que trained_model existe. Déterminez ce qui est nécessaire."
#
# Cet atelier est une TRANSFORMATION GUIDÉE :
#   1. Nous commençons avec votre pipeline Prefect terminé
#   2. Nous convertissons chaque tâche en un asset Dagster
#   3. Nous explorons l'interface Dagster et le graphe d'assets
#   4. Nous apprenons les patterns de matérialisation
#   5. Nous ajoutons des PLANIFICATIONS pour l'automatisation !
#
# -----------------------------------------------------------------------------
# CONFIGURATION (Docker - Recommandé)
# -----------------------------------------------------------------------------
#
# 1. Démarrer la stack :
#      docker-compose up -d
#
# 2. Accéder aux interfaces :
#      - Dagster: http://localhost:3000 (assets, planifications, exécutions)
#      - MLflow:  http://localhost:5000 (expérimentations, modèles)
#
# 3. Dans l'interface Dagster :
#      - Voir le graphe d'assets
#      - Matérialiser les assets manuellement
#      - Activer les planifications pour l'automatisation
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
# Configuration (identique à Prefect)
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
# PARTIE 1 : LA TRANSFORMATION - De @task à @asset
# =============================================================================
#
# Voici votre code Prefect (de l'atelier) :
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
# Dans Dagster, nous ne définissons pas de "tâches à exécuter" - nous définissons "des données qui existent".
# Transformons chaque élément :
# =============================================================================

# -----------------------------------------------------------------------------
# ASSET 1 : Données Clients Brutes
# -----------------------------------------------------------------------------
# Version Prefect :
#   @task
#   def load_data() -> pd.DataFrame:
#       return pd.read_csv(DATA_PATH)
#
# Version Dagster :

@asset(
    group_name="ingestion",
    description="Données clients brutes chargées depuis CSV ou générées synthétiquement"
)
def raw_customer_data() -> pd.DataFrame:
    """
    ASSET SOURCE : Pas de dépendances (pas de paramètres).

    Ceci est équivalent à votre tâche load_data() de Prefect, mais :
    - Elle est nommée comme DONNÉES (raw_customer_data) et non ACTION (load_data)
    - Elle a des métadonnées (group_name, description)
    - C'est une pièce de données qui EXISTE dans votre système
    """
    print("Chargement des données clients brutes...")

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

    print(f"L'asset contient {len(df)} clients")
    return df


# -----------------------------------------------------------------------------
# ASSET 2 : Features Ingéniérées
# -----------------------------------------------------------------------------
# Version Prefect :
#   @task
#   def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#       ...
#
# Version Dagster :
# REMARQUE : Le nom du paramètre EST la dépendance !

@asset(
    group_name="features",
    description="Features clients avec scores RFM et ratios"
)
def customer_features(raw_customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    ASSET DÉRIVÉ : Dépend de raw_customer_data.

    POINT CLÉ : Le nom du paramètre 'raw_customer_data' correspond au nom de l'asset
    ci-dessus. Dagster sait AUTOMATIQUEMENT que cet asset dépend de raw_customer_data.

    Pas besoin de câblage explicite ! Comparez avec Prefect :
        features = engineer_features(df)  # Vous le câblez dans le flow

    Dans Dagster, la dépendance est DÉCLARÉE dans la signature de la fonction.
    """
    print("Calcul de l'asset customer_features...")
    df = raw_customer_data.copy()

    # Feature engineering (même logique que Prefect)
    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)
    df['order_frequency'] = df['total_orders'] / (df['days_since_signup'] + 1)
    df['support_per_order'] = df['support_tickets'] / (df['total_orders'] + 1)

    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Forme de l'asset : {df.shape}")
    return df


# -----------------------------------------------------------------------------
# ASSET 3 : Modèle Entraîné
# -----------------------------------------------------------------------------
# Dans Prefect, train_model était une tâche qui retournait (model, X_test, y_test, run_id)
# Dans Dagster, nous modélisons ceci comme un asset DICT contenant tout

@asset(
    group_name="training",
    description="Modèle RandomForest entraîné avec suivi MLflow"
)
def trained_model(customer_features: pd.DataFrame) -> dict:
    """
    ASSET MODÈLE : Contient le modèle + métadonnées.

    Cet asset :
    - Dépend de customer_features (inféré depuis le paramètre)
    - S'intègre avec MLflow pour le suivi
    - Retourne un dict avec model, métriques, et run_id

    Quand vous demandez à Dagster de matérialiser trained_model, il matérialise
    automatiquement raw_customer_data et customer_features d'abord !
    """
    print("Entraînement du modèle (avec suivi MLflow)...")

    # Préparer les données
    X = customer_features[FEATURE_COLS]
    y = customer_features['churned']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Suivi MLflow
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

    # Retourner tout comme un dict (c'est ceci qui EST l'asset)
    return {
        "model": model,
        "run_id": run.info.run_id,
        "metrics": metrics,
        "feature_cols": FEATURE_COLS
    }


# -----------------------------------------------------------------------------
# ASSET 4 : Prédictions
# -----------------------------------------------------------------------------
# Remarque : Cet asset a DEUX dépendances !

@asset(
    group_name="predictions",
    description="Prédictions de churn pour tous les clients"
)
def churn_predictions(
    trained_model: dict,
    customer_features: pd.DataFrame
) -> pd.DataFrame:
    """
    ASSET MULTI-DÉPENDANCE : Dépend de trained_model ET customer_features.

    Regardez le graphe d'assets dans l'interface - vous verrez :

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

    Les deux dépendances sont inférées depuis les noms de paramètres !
    """
    print("Génération des prédictions...")

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
    print(f"Généré {len(result)} prédictions ({high_risk} à haut risque)")

    return result


# -----------------------------------------------------------------------------
# ASSET 5 : Prédictions Sauvegardées (I/O comme asset !)
# -----------------------------------------------------------------------------

@asset(
    group_name="output",
    description="Prédictions persistées dans un fichier CSV"
)
def saved_predictions(churn_predictions: pd.DataFrame) -> dict:
    """
    ASSET OUTPUT : Représente "les prédictions qui ont été sauvegardées".

    Même l'I/O fichier est modélisé comme données ! Cet asset dépend de churn_predictions
    et produit des métadonnées sur ce qui a été sauvegardé.

    Ceci est puissant car :
    - Vous pouvez voir dans l'interface quand les prédictions ont été sauvegardées pour la dernière fois
    - Vous pouvez re-matérialiser juste cet asset pour re-sauvegarder
    - La chaîne de dépendances est claire et visible
    """
    output_path = os.path.join(PROJECT_ROOT, "data", "predictions_dagster.csv")

    churn_predictions.to_csv(output_path, index=False)

    print(f"Sauvegardé {len(churn_predictions)} prédictions dans {output_path}")

    return {
        "path": output_path,
        "record_count": len(churn_predictions),
        "high_risk_count": int((churn_predictions['churn_probability'] > 0.7).sum()),
        "saved_at": datetime.now().isoformat()
    }


# =============================================================================
# PARTIE 2 : LES DÉFINITIONS - Enregistrer les Assets
# =============================================================================
#
# Dans Prefect, vous définissez un @flow qui câble les tâches ensemble.
# Dans Dagster, vous enregistrez les assets avec Definitions - pas besoin de câblage !

all_assets = [
    raw_customer_data,
    customer_features,
    trained_model,
    churn_predictions,
    saved_predictions,
]


# =============================================================================
# PARTIE 5 : AUTOMATISATION - Jobs & Planifications
# =============================================================================
#
# C'est ici que l'ORCHESTRATION devient une vraie automatisation !
#
# Dans Dagster :
# - Un JOB définit QUELS assets matérialiser
# - Une PLANIFICATION définit QUAND exécuter le job
# - Le DAEMON (exécuté dans Docker) exécute les planifications
#
# =============================================================================

# -----------------------------------------------------------------------------
# JOB : Quels assets matérialiser
# -----------------------------------------------------------------------------

training_job = define_asset_job(
    name="churn_training_job",
    selection=AssetSelection.all(),  # Matérialiser tous les assets
    description="Pipeline complet de prédiction de churn - entraîne le modèle et génère les prédictions"
)

# Alternative : jobs sélectifs
data_prep_job = define_asset_job(
    name="data_preparation_job",
    selection=[AssetKey("raw_customer_data"), AssetKey("customer_features")],
    description="Préparer les données sans entraînement"
)

# -----------------------------------------------------------------------------
# PLANIFICATION : Quand exécuter le job
# -----------------------------------------------------------------------------

training_schedule = ScheduleDefinition(
    name="churn_training_schedule",
    job=training_job,
    cron_schedule="*/2 * * * *",  # Toutes les 2 minutes (pour la démo)
    description="Réentraînement automatisé toutes les 2 minutes"
)

# Exemples de planifications alternatives (commentés) :
# daily_training = ScheduleDefinition(
#     name="daily_training",
#     job=training_job,
#     cron_schedule="0 6 * * *",  # Tous les jours à 6h du matin
# )

# -----------------------------------------------------------------------------
# DÉFINITIONS : Enregistrer tout avec Dagster
# -----------------------------------------------------------------------------

defs = Definitions(
    assets=all_assets,
    jobs=[training_job, data_prep_job],
    schedules=[training_schedule],
)


# =============================================================================
# PARTIE 3 : MATÉRIALISATION - Exécuter le Pipeline
# =============================================================================
#
# "Matérialiser" un asset signifie le calculer et stocker le résultat.
#
# Différences clés avec Prefect :
#
# Prefect :
#   training_pipeline()  # Exécuter tout le flow
#
# Dagster :
#   materialize([trained_model])  # "Je veux que trained_model existe"
#   # Dagster comprend qu'il a besoin de raw_customer_data et customer_features d'abord !
#
# Vous pouvez aussi matérialiser des SOUS-ENSEMBLES :
#   materialize([customer_features])  # Juste la préparation de données, pas d'entraînement
#   materialize([churn_predictions])  # Tout ce qui est nécessaire pour les prédictions
# =============================================================================

def run_full_pipeline():
    """Matérialiser tous les assets (équivalent à exécuter le flow Prefect complet)."""
    print("=" * 60)
    print("MATÉRIALISATION DE TOUS LES ASSETS")
    print("=" * 60)

    result = materialize(all_assets)

    # Obtenir les sorties
    saved = result.output_for_node("saved_predictions")
    model_data = result.output_for_node("trained_model")

    print("\n" + "=" * 60)
    print("MATÉRIALISATION TERMINÉE")
    print("=" * 60)
    print(f"Précision du modèle :      {model_data['metrics']['accuracy']:.4f}")
    print(f"Prédictions sauvegardées : {saved['record_count']}")
    print(f"Clients à haut risque :   {saved['high_risk_count']}")
    print(f"Fichier de sortie :        {saved['path']}")
    print(f"Exécution MLflow :         {model_data['run_id']}")

    return result


def run_data_prep_only():
    """Matérialiser seulement les assets de préparation de données (pas d'entraînement)."""
    print("=" * 60)
    print("MATÉRIALISATION DE LA PRÉPARATION DE DONNÉES UNIQUEMENT")
    print("=" * 60)

    # Matérialiser seulement jusqu'à customer_features
    result = materialize([raw_customer_data, customer_features])

    features = result.output_for_node("customer_features")
    print(f"\nDonnées préparées : {features.shape}")

    return result


def run_from_existing_features():
    """
    Démontrer la matérialisation partielle.

    Dans un scénario réel, vous pourriez avoir customer_features déjà calculé
    et vouloir juste réentraîner le modèle.
    """
    print("=" * 60)
    print("MATÉRIALISATION SÉLECTIVE")
    print("=" * 60)
    print("Ceci montre comment Dagster peut ré-exécuter seulement ce qui est nécessaire.")
    print("Dans l'interface, vous pouvez cliquer sur des assets individuels pour les matérialiser.")

    # Matérialiser tout
    result = materialize(all_assets)

    return result


# =============================================================================
# PARTIE 4 : L'INTERFACE - L'Expérience Dagster Réelle
# =============================================================================
#
# La ligne de commande est utile, mais la puissance de Dagster est dans l'interface !
#
# Démarrez avec Docker :
#   docker-compose up -d
#
# Puis ouvrez http://localhost:3000 et explorez :
#
# 1. GRAPHE D'ASSETS (Assets > View global asset lineage)
#    - Voir le graphe de dépendances visuel
#    - Cliquer sur un asset pour voir ses métadonnées
#    - Voir quels assets sont matérialisés (ont des données) vs non matérialisés
#
# 2. MATÉRIALISER (Manuel)
#    - Cliquer "Materialize all" pour tout exécuter
#    - Ou cliquer sur des assets individuels pour matérialiser juste ce sous-ensemble
#    - Regarder les journaux en temps réel
#
# 3. PLANIFICATIONS (Automatisation !)
#    - Aller dans Overview > Schedules
#    - Trouver "churn_training_schedule"
#    - L'activer (toggle ON)
#    - Regarder l'exécution toutes les 2 minutes !
#    - Vérifier l'interface MLflow pour les nouvelles expériences
#
# 4. JOBS
#    - Aller dans Overview > Jobs
#    - Voir les jobs définis (churn_training_job, data_preparation_job)
#    - Lancer les jobs manuellement ou via les planifications
#
# 5. EXÉCUTIONS
#    - Voir l'historique de toutes les matérialisations
#    - Déboguer les exécutions échouées
#    - Ré-exécuter depuis les échecs
#
# =============================================================================


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ATELIER DAGSTER - Des Tâches aux Assets")
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
ATELIER DAGSTER - BONUS
========================

Cet atelier transforme votre pipeline Prefect en assets Dagster.

CONFIGURATION (Docker - Recommandé) :
  docker-compose up -d

  Interfaces :
    Dagster: http://localhost:3000
    MLflow:  http://localhost:5000

UTILISATION EN LIGNE DE COMMANDE (pour tester) :
  python Dagster_Workshop.py full       # Matérialiser tous les assets
  python Dagster_Workshop.py data       # Seulement préparation de données (pas d'entraînement)
  python Dagster_Workshop.py selective  # Démo de matérialisation sélective

LES CONCEPTS CLÉS :

1. ASSETS vs TÂCHES
   Prefect: @task def load_data()     -> "exécuter cette fonction"
   Dagster: @asset def customer_data  -> "ces données existent"

2. DÉPENDANCES AUTOMATIQUES
   Prefect: features = engineer(df)   -> vous le câblez dans le flow
   Dagster: def features(raw_data):   -> dépendance depuis le nom de paramètre !

3. MATÉRIALISATION
   Prefect: flow()                    -> exécuter tout le pipeline
   Dagster: materialize([asset])      -> "faire exister ceci" (déps auto-résolues)

4. JOBS & PLANIFICATIONS (Automatisation !)
   Les jobs définissent QUELS assets matérialiser
   Les planifications définissent QUAND exécuter les jobs
   Le daemon (dans Docker) les exécute automatiquement

AUTOMATISATION :
  1. Ouvrir l'interface Dagster : http://localhost:3000
  2. Aller dans Overview > Schedules
  3. Activer "churn_training_schedule"
  4. Regarder les exécutions apparaître toutes les 2 minutes !
  5. Vérifier l'interface MLflow pour les nouvelles expériences

LE GRAPHE :
  Ouvrir l'interface pour voir votre linéage de données visuellement.
  Cliquer sur les assets pour les matérialiser individuellement.
""")

        print(f"Interface Dagster : http://localhost:3000")
        print(f"Interface MLflow :  {MLFLOW_TRACKING_URI}")
