# =============================================================================
# Atelier Prefect - Orchestration de Workflows ML
# =============================================================================
#
# Cet atelier enseigne l'ORCHESTRATION à travers des cas d'usage ML.
#
# SECTIONS :
#   Partie 1 : Tasks & Flows - Les bases
#   Partie 2 : Résilience - Réessais, gestion d'erreurs
#   Partie 3 : Efficacité - Cache, exécution parallèle
#   Partie 4 : Flexibilité - Paramètres, sous-flows
#   Partie 5 : Pipeline Complet - Tout avec MLflow
#   Partie 6 : AUTOMATISATION - Déployer, planifier, regarder l'exécution !
#   Partie 7 : NOTIFICATIONS - Alertes Discord/Slack sur échec
#
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
#
# 1. Démarrer la stack d'orchestration :
#      docker-compose up -d
#
# 2. Accéder aux interfaces :
#      - Prefect: http://localhost:4200 (flows, déploiements, exécutions)
#      - MLflow:  http://localhost:5000 (expérimentations, modèles)
#
# 3. Exécuter les parties de l'atelier :
#      python pipelines/workshop/02_prefect/Prefect_Workshop.py part1
#      ...
#      python pipelines/workshop/02_prefect/Prefect_Workshop.py deploy
#
# =============================================================================

import sys
import os
from pathlib import Path

from prefect import flow, task, serve
from prefect.tasks import task_input_hash
from prefect.client.orchestration import get_client
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time
import random
import os
import asyncio

# Imports pour les notifications (Partie 7)
# Note : httpx est une dépendance de Prefect, donc déjà installé
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️ httpx non disponible - les notifications webhook ne fonctionneront pas")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Charger le fichier .env pour la configuration locale (PREFECT_API_URL, MLFLOW_TRACKING_URI)
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

# Lors de l'exécution dans Docker, ces variables d'environnement sont définies par docker-compose
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(PROJECT_ROOT)))
DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")

FEATURE_COLS = [
    'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
    'days_since_signup', 'total_orders', 'support_tickets', 'age'
]


# =============================================================================
# PARTIE 1 : TASKS & FLOWS - Les Bases
# =============================================================================
#
# PROBLÈME ML :
#   Votre notebook exécute les cellules séquentiellement. Si la cellule 5 échoue,
#   vous relancez tout. Il n'y a pas de structure, pas de visibilité, pas de réutilisabilité.
#
# SOLUTION D'ORCHESTRATION :
#   Décomposer votre code en TASKS (responsabilités uniques) et FLOWS (pipelines).
#   Chaque tâche est indépendante, loggée, et peut être réessayée.
#
# =============================================================================

@task
def load_data() -> pd.DataFrame:
    """
    Une TASK est une unité de travail unique.

    Points clés :
    - Le décorateur @task transforme une fonction en tâche orchestrée
    - Les valeurs de retour circulent vers la tâche suivante (pas besoin d'I/O fichier !)
    - Prefect loggue automatiquement : heure de début, durée, succès/échec
    """
    print("Chargement des données clients...")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        # Générer des données synthétiques
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

    print(f"Chargé {len(df)} lignes")
    return df


@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tâche de feature engineering."""
    print("Ingénierie des features...")
    df = df.copy()

    df['recency_frequency_ratio'] = df['recency_days'] / (df['frequency'] + 1)
    df['monetary_per_order'] = df['monetary_value'] / (df['total_orders'] + 1)

    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['m_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

    print(f"Créé {len(df.columns)} features")
    return df


@flow(name="basic-pipeline", log_prints=True)
def basic_pipeline():
    """
    Un FLOW orchestre les tâches.

    Appelez les tâches comme des fonctions normales - les données circulent via les valeurs de retour.
    """
    raw_data = load_data()
    features = engineer_features(raw_data)
    print(f"Pipeline terminé ! Forme : {features.shape}")
    return features


# =============================================================================
# PARTIE 2 : RÉSILIENCE - Gestion des Échecs
# =============================================================================

@task(retries=3, retry_delay_seconds=5)
def load_from_unreliable_api() -> pd.DataFrame:
    """
    Les RÉESSAIS gèrent les échecs transitoires automatiquement.

    Cette tâche va :
    1. Essayer de s'exécuter
    2. Si elle échoue, attendre 5 secondes
    3. Réessayer (à nouveau jusqu'à 3 fois)
    """
    if random.random() < 0.5:
        print("API call failed! (simulated)")
        raise ConnectionError("API temporarily unavailable")

    print("Appel API réussi !")
    return load_data()


@task(retries=3, retry_delay_seconds=[10, 30, 60])
def load_with_backoff() -> pd.DataFrame:
    """
    BACKOFF EXPONENTIEL : Attendre plus longtemps entre chaque réessai.
    Premier échec : 10s, Deuxième : 30s, Troisième : 60s
    """
    if random.random() < 0.7:
        raise ConnectionError("Rate limited!")
    return load_data()


# =============================================================================
# PARTIE 3 : EFFICACITÉ - Cache & Parallélisme
# =============================================================================

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def expensive_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    CACHE : Sauter le recalcul si les entrées n'ont pas changé.
    """
    print("Exécution du feature engineering coûteux...")
    time.sleep(2)  # Simuler un calcul coûteux
    df = df.copy()
    df['expensive_feature'] = df['monetary_value'] * df['frequency']
    print("Feature engineering terminé !")
    return df


@task
def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Entraîner un modèle Random Forest."""
    print("Entraînement Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Précision Random Forest : {accuracy:.4f}")
    return {"name": "RandomForest", "model": model, "accuracy": accuracy}


@task
def train_gradient_boosting(X_train, y_train, X_test, y_test) -> dict:
    """Entraîner un modèle Gradient Boosting."""
    print("Entraînement Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Précision Gradient Boosting : {accuracy:.4f}")
    return {"name": "GradientBoosting", "model": model, "accuracy": accuracy}


@task
def select_best_model(results: list) -> dict:
    """Sélectionner le modèle avec la meilleure précision."""
    best = max(results, key=lambda x: x["accuracy"])
    print(f"Meilleur modèle : {best['name']} avec précision {best['accuracy']:.4f}")
    return best


@flow(name="parallel-training", log_prints=True)
def parallel_training_flow():
    """
    EXÉCUTION PARALLÈLE : Entraîner plusieurs modèles simultanément.
    """
    df = load_data()
    df = engineer_features(df)

    X = df[FEATURE_COLS + ['rfm_score']]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ceux-ci s'exécutent en parallèle - pas de dépendance entre eux !
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)

    best = select_best_model([rf_result, gb_result])
    return best


# =============================================================================
# PARTIE 4 : FLEXIBILITÉ - Paramètres & Sous-flows
# =============================================================================

@task
def train_with_params(df: pd.DataFrame, n_estimators: int, max_depth: int) -> dict:
    """Tâche qui accepte des hyperparamètres."""
    X = df[FEATURE_COLS + ['rfm_score']]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    print(f"Modèle : n_estimators={n_estimators}, max_depth={max_depth}, precision={accuracy:.4f}")
    return {"model": model, "params": {"n_estimators": n_estimators, "max_depth": max_depth}, "metrics": {"accuracy": accuracy, "f1": f1}}


@flow(name="parameterized-training", log_prints=True)
def parameterized_training_flow(n_estimators: int = 100, max_depth: int = 10):
    """
    FLOW PARAMÉTRÉ : Configurer sans changer le code.
    """
    df = load_data()
    df = engineer_features(df)
    result = train_with_params(df, n_estimators, max_depth)
    return result


@flow(name="data-preparation", log_prints=True)
def data_preparation_subflow() -> pd.DataFrame:
    """SOUS-FLOW : Préparation de données réutilisable."""
    df = load_data()
    df = engineer_features(df)
    return df


@flow(name="training-subflow", log_prints=True)
def training_subflow(df: pd.DataFrame, n_estimators: int = 100) -> dict:
    """Entraînement comme sous-flow séparé."""
    return train_with_params(df, n_estimators=n_estimators, max_depth=10)


# =============================================================================
# PARTIE 5 : PIPELINE COMPLET AVEC MLFLOW
# =============================================================================

@task(retries=3, retry_delay_seconds=5)
def load_data_with_retry() -> pd.DataFrame:
    """Chargement de données avec logique de réessai."""
    return load_data()


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def engineer_features_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering avec cache."""
    return engineer_features(df)


@task
def train_with_mlflow(df: pd.DataFrame, n_estimators: int, max_depth: int, experiment_name: str) -> dict:
    """
    Tâche d'entraînement avec intégration MLflow.

    Prefect gère : réessais, cache, planification
    MLflow gère : suivi des expérimentations, versionnage des modèles
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
        mlflow.sklearn.log_model(model, name="model")

        print(f"MLflow run: {run.info.run_id}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    return {"model": model, "run_id": run.info.run_id, "metrics": metrics}


@task
def register_model(run_id: str, model_name: str) -> str:
    """Enregistrer le modèle dans le registre MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Enregistré {model_name} version {result.version}")
    return result.version


@task
def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Générer des prédictions pour tous les clients."""
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
    print(f"Généré {len(result)} prédictions ({high_risk} à haut risque)")
    return result


@task
def save_predictions(predictions: pd.DataFrame, output_path: str | Path) -> str:
    """Sauvegarder les prédictions en CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Sauvegardé dans {output_path}")
    return output_path


@flow(name="churn-prediction-pipeline", log_prints=True)
def production_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    experiment_name: str = "workshop-prefect",
    model_name: str = "churn-predictor"
):
    """
    PIPELINE ML DE PRODUCTION

    Combine tous les patterns :
    - RÉESSAIS sur le chargement de données
    - CACHE sur le feature engineering
    - PARAMÈTRES pour les hyperparamètres
    - MLFLOW pour le suivi

    Ce flow peut être :
    - Exécuté manuellement : production_pipeline()
    - Déployé avec planification : Voir Partie 6
    """
    print("=" * 60)
    print("PIPELINE DE PRÉDICTION DE CHURN")
    print(f"Heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Données
    df = load_data_with_retry()
    df = engineer_features_cached(df)

    # Entraînement
    result = train_with_mlflow(df, n_estimators, max_depth, experiment_name)

    # Enregistrement
    version = register_model(result["run_id"], model_name)

    # Inférence
    predictions = generate_predictions(result["model"], df)
    output_path = PROJECT_ROOT / "data" / "predictions_workshop.csv"
    save_predictions(predictions, output_path)

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ")
    print("=" * 60)
    print(f"Modèle : {model_name} v{version}")
    print(f"Précision : {result['metrics']['accuracy']:.4f}")
    print(f"Prédictions : {len(predictions)}")

    return {"model_version": version, "metrics": result["metrics"], "predictions_count": len(predictions)}


# =============================================================================
# PARTIE 6 : AUTOMATISATION - Déployer & Planifier
# =============================================================================
#
# C'est ici que l'ORCHESTRATION devient une vraie automatisation !
#
# Jusqu'à présent, nous avons exécuté les flows manuellement. En production :
# - Les flows s'exécutent selon des PLANIFICATIONS (réentraînement quotidien, prédictions horaires)
# - Un WORKER exécute les flows (dans Docker)
# - Vous SURVEILLEZ dans l'interface (voir les exécutions, échecs, logs)
#
# =============================================================================

def deploy_with_schedule():
    """
    DÉPLOYER le pipeline avec une planification.

    Ceci crée un DÉPLOIEMENT :
    - Enregistre le flow auprès du serveur Prefect
    - Configure une planification (toutes les 2 minutes pour la démo)
    - Le Worker le récupère et l'exécute

    Après avoir exécuté ceci :
    1. Ouvrir l'interface Prefect : http://localhost:4200
    2. Aller dans Déploiements
    3. Voir "churn-prediction-pipeline/scheduled-training"
    4. Observer les exécutions apparaître toutes les 2 minutes !
    5. Vérifier l'interface MLflow : http://localhost:5000 pour les nouvelles expérimentations

    POUR ARRÊTER :
    - Terminal : Ctrl+C (arrête le processus serve())
    - Interface Prefect : Deployments > scheduled-training > Pause ou Delete
    """
    print("=" * 60)
    print("DÉPLOIEMENT DU PIPELINE AVEC PLANIFICATION")
    print("=" * 60)
    print("\nCeci va :")
    print("1. Enregistrer le flow auprès du serveur Prefect")
    print("2. Configurer une planification (toutes les 2 minutes)")
    print("3. Le Worker l'exécutera automatiquement")
    print("\nAprès le déploiement :")
    print(f"  - Interface Prefect: http://localhost:4200")
    print(f"  - Interface MLflow:  {MLFLOW_TRACKING_URI}")
    print("\nPOUR ARRÊTER :")
    print("  - Terminal : Ctrl+C")
    print("  - Interface : Deployments > Pause ou Delete")
    print("=" * 60)

    # Déployer en utilisant serve() - ceci exécute le planificateur localement
    # En production, vous utiliseriez flow.deploy() avec un work pool
    production_pipeline.serve(
        name="scheduled-training",
        cron="*/2 * * * *",  # Toutes les 2 minutes (pour la démo)
        tags=["workshop", "ml", "churn"],
        description="Réentraînement automatisé du modèle de churn - s'exécute toutes les 2 minutes",
        parameters={
            "n_estimators": 100,
            "max_depth": 10,
            "experiment_name": "workshop-automated",
            "model_name": "churn-predictor-auto"
        }
    )


def run_single():
    """Exécuter le pipeline une fois (pour tester avant le déploiement)."""
    print("Exécution du pipeline une fois...")
    result = production_pipeline(
        n_estimators=100,
        max_depth=10,
        experiment_name="workshop-prefect",
        model_name="churn-predictor"
    )
    return result


def deploy_to_worker():
    """
    DÉPLOYER le pipeline vers le WORKER DOCKER.

    DIFFÉRENCE avec serve() :
    - serve() : Le flow s'exécute dans VOTRE processus Python (local)
    - deploy() : Le flow s'exécute dans le WORKER DOCKER (production)

    Après avoir exécuté ceci :
    1. Votre terminal est LIBÉRÉ (pas de processus bloquant)
    2. Le worker Docker va exécuter le flow
    3. Ouvrir l'interface Prefect : http://localhost:4200
    4. Aller dans Deployments > worker-training
    5. Cliquer sur "Quick Run" pour déclencher manuellement
    6. Ou attendre la planification (toutes les 5 minutes)

    POUR VOIR LES LOGS DU WORKER :
        docker-compose logs -f prefect-worker

    POUR SUPPRIMER LE DÉPLOIEMENT :
        Interface Prefect > Deployments > Delete
    """
    print("=" * 60)
    print("DÉPLOIEMENT VERS LE WORKER DOCKER")
    print("=" * 60)
    print("""
DIFFÉRENCE serve() vs deploy() :

  serve()  → Le flow s'exécute ICI (votre terminal)
             Ctrl+C arrête tout
             Simple, bon pour apprendre

  deploy() → Le flow s'exécute dans le WORKER DOCKER
             Votre terminal est libéré
             Architecture de production

""")
    print("Déploiement en cours...")
    print()

    # Créer le déploiement vers le work pool Docker
    # Le worker dans docker-compose écoute "default-pool"
    # Note: Pour un work pool de type "process", on utilise from_source() pour
    # indiquer où se trouve le code que le worker doit exécuter.
    deployment_id = production_pipeline.from_source(
        source=str(PROJECT_ROOT),  # Chemin vers le code source
        entrypoint="pipelines/workshop/02_prefect/Prefect_Workshop.py:production_pipeline"
    ).deploy(
        name="worker-training",
        work_pool_name="default-pool",
        cron="*/5 * * * *",  # Toutes les 5 minutes
        tags=["workshop", "ml", "worker-demo"],
        description="Pipeline exécuté par le worker Docker - toutes les 5 minutes",
        parameters={
            "n_estimators": 100,
            "max_depth": 10,
            "experiment_name": "workshop-worker",
            "model_name": "churn-predictor-worker"
        }
    )

    print()
    print("=" * 60)
    print("✅ DÉPLOIEMENT RÉUSSI !")
    print("=" * 60)
    print(f"""
Le flow est maintenant déployé vers le worker Docker.

PROCHAINES ÉTAPES :

1. Ouvrir l'interface Prefect : http://localhost:4200
   → Aller dans Deployments
   → Trouver "churn-prediction-pipeline/worker-training"

2. Déclencher manuellement :
   → Cliquer sur "Quick Run"
   → Observer l'exécution dans l'onglet "Runs"

3. Voir les logs du worker :
   docker-compose logs -f prefect-worker

4. Vérifier MLflow : http://localhost:5000
   → Expérience "workshop-worker"

5. Pour supprimer le déploiement :
   → Interface Prefect > Deployments > Delete

COMPARAISON :
  - 'deploy' (serve)  : Exécution locale, terminal bloqué
  - 'worker-demo'     : Exécution Docker, terminal libre ← VOUS ÊTES ICI
""")

    return deployment_id


# =============================================================================
# PARTIE 7 : NOTIFICATIONS - Alertes Discord/Slack
# =============================================================================
#
# PROBLÈME :
#   Vous ne voulez pas surveiller vos pipelines 24h/24. Vous voulez être
#   ALERTÉ quand quelque chose échoue (ou réussit) !
#
# SOLUTION :
#   Prefect supporte les notifications via webhooks : Discord, Slack, Teams, etc.
#   Recevez un message sur votre serveur Discord quand un flow échoue !
#
# =============================================================================

def send_discord_notification(webhook_url: str, message: str, username: str = "Prefect Bot"):
    """
    Envoyer une notification Discord via webhook.

    Pour créer un webhook Discord :
    1. Aller dans les paramètres de votre serveur Discord
    2. Intégrations > Webhooks > Nouveau Webhook
    3. Copier l'URL du webhook

    L'URL ressemble à : https://discord.com/api/webhooks/<id>/<token>
    """
    if not HTTPX_AVAILABLE:
        print("⚠️ httpx non disponible - notification non envoyée")
        return

    payload = {
        "username": username,
        "content": message
    }

    try:
        response = httpx.post(webhook_url, json=payload)
        response.raise_for_status()
        print(f"✅ Notification Discord envoyée")
    except Exception as e:
        print(f"❌ Erreur envoi Discord : {e}")


def send_slack_notification(webhook_url: str, message: str):
    """
    Envoyer une notification Slack via webhook.

    Pour créer un webhook Slack :
    1. Aller sur https://api.slack.com/apps
    2. Créer une app > Incoming Webhooks > Activer
    3. Ajouter à un channel et copier l'URL
    """
    if not HTTPX_AVAILABLE:
        print("⚠️ httpx non disponible - notification non envoyée")
        return

    payload = {"text": message}

    try:
        response = httpx.post(webhook_url, json=payload)
        response.raise_for_status()
        print(f"✅ Notification Slack envoyée")
    except Exception as e:
        print(f"❌ Erreur envoi Slack : {e}")


def create_failure_handler(webhook_url: str, service: str = "discord"):
    """
    Créer un handler d'échec pour un flow.

    Ce handler sera appelé automatiquement quand le flow échoue.
    """
    def handle_failure(flow, flow_run, state):
        message = f"""🚨 **ÉCHEC DU PIPELINE**

📋 Flow : `{flow.name}`
🔢 Run ID : `{flow_run.id}`
❌ État : `{state.name}`
⏰ Heure : `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

Consultez l'interface Prefect pour plus de détails : http://localhost:4200
"""
        if service == "discord":
            send_discord_notification(webhook_url, message)
        elif service == "slack":
            send_slack_notification(webhook_url, message)

    return handle_failure


def create_success_handler(webhook_url: str, service: str = "discord"):
    """
    Créer un handler de succès pour un flow.

    Optionnel : être notifié aussi des succès (utile pour le monitoring).
    """
    def handle_success(flow, flow_run, state):
        message = f"""✅ **PIPELINE RÉUSSI**

📋 Flow : `{flow.name}`
🔢 Run ID : `{flow_run.id}`
✅ État : `{state.name}`
⏰ Heure : `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
        if service == "discord":
            send_discord_notification(webhook_url, message)
        elif service == "slack":
            send_slack_notification(webhook_url, message)

    return handle_success


# Exemple de flow avec notifications
# NOTE : Remplacez VOTRE_WEBHOOK_URL par votre vrai webhook Discord/Slack !
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


@task
def task_that_might_fail(fail_probability: float = 0.5):
    """Tâche qui peut échouer (pour tester les notifications)."""
    print(f"Exécution de la tâche (probabilité d'échec : {fail_probability})")
    if random.random() < fail_probability:
        raise ValueError("💥 Échec simulé ! Vérifiez vos notifications.")
    print("✅ Tâche réussie !")
    return "success"


def run_notification_demo():
    """
    Démonstration des notifications.

    INSTRUCTIONS :
    1. Créez un webhook Discord ou Slack
    2. Définissez la variable d'environnement :
       - Discord : export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
       - Slack : export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    3. Exécutez cette démo avec : python Prefect_Workshop.py part7
    """
    print("=" * 60)
    print("PARTIE 7 : NOTIFICATIONS")
    print("=" * 60)

    # Vérifier si un webhook est configuré
    webhook_url = DISCORD_WEBHOOK_URL or SLACK_WEBHOOK_URL
    service = "discord" if DISCORD_WEBHOOK_URL else "slack"

    if not webhook_url:
        print("""
⚠️  AUCUN WEBHOOK CONFIGURÉ !

Pour tester les notifications :

1. Créer un webhook Discord :
   - Paramètres serveur > Intégrations > Webhooks > Nouveau
   - Copier l'URL

2. Définir la variable d'environnement :
   - Windows : set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   - Linux/Mac : export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

3. Relancer : python Prefect_Workshop.py part7

ALTERNATIVE : Mode simulation (sans vrai webhook)
""")
        # Mode simulation sans vrai webhook
        print("Mode simulation : on affiche ce qui serait envoyé\n")

        @flow(name="notification-demo-simulation", log_prints=True)
        def demo_flow_simulation():
            print("🔔 [SIMULATION] Notification de démarrage serait envoyée")
            result = task_that_might_fail(fail_probability=0.5)
            print("🔔 [SIMULATION] Notification de succès serait envoyée")
            return result

        try:
            demo_flow_simulation()
        except Exception as e:
            print(f"🔔 [SIMULATION] Notification d'échec serait envoyée : {e}")

        return

    # Webhook configuré - créer les handlers
    failure_handler = create_failure_handler(webhook_url, service)
    success_handler = create_success_handler(webhook_url, service)

    @flow(
        name="notification-demo",
        log_prints=True,
        on_failure=[failure_handler],
        on_completion=[success_handler]  # Appelé sur succès ET échec
    )
    def demo_flow_with_notifications():
        """Flow avec notifications automatiques."""
        result = task_that_might_fail(fail_probability=0.3)
        return result

    print(f"\nWebhook {service.upper()} configuré !")
    print("Exécution du flow de démonstration...")
    print("Surveillez votre channel pour les notifications !\n")

    try:
        demo_flow_with_notifications()
        print("\n✅ Flow terminé avec succès - notification envoyée !")
    except Exception as e:
        print(f"\n❌ Flow échoué - notification d'échec envoyée !")


# =============================================================================
# NOTE PÉDAGOGIQUE : PATTERNS DE PRÉTRAITEMENT
# =============================================================================
#
# Les patterns de preprocessing (scaler.pkl, sklearn Pipeline) sont
# couverts en détail dans les notebooks :
#   - 01b_mlflow_transition.ipynb (section 2.3)
#   - 02_mlflow_organized.ipynb (sections 8.1-8.6, comparaison complète)
#
# Ce fichier se concentre sur l'ORCHESTRATION avec Prefect.
#
# RÉSUMÉ DES MEILLEURES PRATIQUES :
#   - Utilisez sklearn Pipeline pour combiner scaler + modèle en UN artefact
#   - Loggez le pipeline complet dans MLflow : mlflow.sklearn.log_model(pipeline)
#   - À l'inférence : pipeline.predict(X) applique automatiquement le scaling
#
# =============================================================================


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
        print("\nPARTIE 3 : Efficacité (Cache & Parallélisme)")
        print("-" * 40)
        result = parallel_training_flow()

    elif mode == "part4":
        print("\nPARTIE 4 : Flexibilité (Paramètres & Sous-flows)")
        print("-" * 40)
        result = parameterized_training_flow(n_estimators=50)

    elif mode == "part5" or mode == "full":
        print("\nPARTIE 5 : Pipeline Complet avec MLflow")
        print("-" * 40)
        result = run_single()

    elif mode == "deploy":
        print("\nPARTIE 6 : Déployer avec Planification (AUTOMATISATION !)")
        print("-" * 40)
        print("\nCeci va démarrer un processus de longue durée qui :")
        print("1. Enregistre le flow")
        print("2. L'exécute toutes les 2 minutes")
        print("3. Loggue dans MLflow")
        print("\nAppuyez sur Ctrl+C pour arrêter.\n")
        deploy_with_schedule()

    elif mode == "worker-demo" or mode == "worker":
        print("\nBONUS : Déployer vers le Worker Docker")
        print("-" * 40)
        deploy_to_worker()

    elif mode == "part7" or mode == "notifications":
        run_notification_demo()

    else:
        print("""
ATELIER PREFECT - Orchestration ML
====================================

CONFIGURATION :
  docker-compose up -d

  Interfaces :
    Prefect: http://localhost:4200
    MLflow:  http://localhost:5000

MODES :
  part1           Tasks & Flows (bases)
  part2           Résilience (réessais)
  part3           Efficacité (cache, parallélisme)
  part4           Flexibilité (paramètres, sous-flows)
  part5           Pipeline Complet avec MLflow
  deploy          AUTOMATISATION - Exécution locale avec serve()
  worker-demo     PRODUCTION - Exécution dans le worker Docker !
  part7           NOTIFICATIONS - Alertes Discord/Slack

DÉROULEMENT DE L'ATELIER :
  1. Exécuter part1-part5 pour apprendre les patterns d'orchestration
  2. Exécuter 'deploy' pour voir l'automatisation réelle
  3. Exécuter 'worker-demo' pour déployer vers le worker Docker
  4. Exécuter 'part7' pour configurer les alertes
  5. Ouvrir l'interface Prefect pour observer les exécutions
  6. Ouvrir l'interface MLflow pour voir les expérimentations

NOTE : Les patterns de preprocessing (scaler.pkl, sklearn Pipeline) sont
       couverts dans les notebooks (01b, 02_mlflow_organized).

NOTIFICATIONS (mode part7) :
  - Créer un webhook Discord ou Slack
  - Définir la variable : DISCORD_WEBHOOK_URL ou SLACK_WEBHOOK_URL
  - Recevoir des alertes automatiques sur échec !

AUTOMATISATION :
  deploy       → serve() : exécution locale, terminal bloqué
  worker-demo  → deploy() : exécution Docker, terminal libre (production)

  Comparaison :
    - 'deploy' utilise serve() - le flow tourne dans VOTRE terminal
    - 'worker-demo' utilise deploy() - le flow tourne dans le WORKER DOCKER

Exemple :
  python Prefect_Workshop.py part1       # Apprendre les bases
  python Prefect_Workshop.py part5       # Pipeline complet avec MLflow
  python Prefect_Workshop.py deploy      # Local (Ctrl+C pour arrêter)
  python Prefect_Workshop.py worker-demo # Docker (terminal libéré)
  python Prefect_Workshop.py part7       # Notifications Discord/Slack
""")
