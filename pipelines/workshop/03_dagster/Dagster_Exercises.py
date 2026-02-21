# =============================================================================
# Atelier Dagster - EXERCICES : Le Changement de Paradigme
# =============================================================================
#
# 🎯 OBJECTIF : Comprendre l'approche "Asset-oriented" vs "Task-oriented"
#
# Après 2 heures de Prefect (Task-oriented), découvrez un autre paradigme :
#   - Prefect : "Fais ceci, puis fais cela" (verbes, actions)
#   - Dagster : "Je veux que cette donnée existe" (noms, états)
#
# LANCEMENT :
#   Avec Docker (recommandé) : docker-compose --profile dagster up -d
#   Sans Docker : dagster dev -f Dagster_Exercises.py
#
# Puis ouvrez http://localhost:3000 et observez le graphe se dessiner !
#
# IMPORTANT : L'objectif est de passer 20% du temps à coder et 80% du temps
# dans l'interface web pour comprendre visuellement le paradigme.
#
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Charger le fichier .env pour MLflow
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Imports Dagster
from dagster import asset, Config, MaterializeResult, MetadataValue, Definitions

# =============================================================================
# 📊 COMPRENDRE LE PARADIGME : TASK vs ASSET
# =============================================================================
#
# PREFECT (Task-oriented) - Ce que vous avez fait :
# ------------------------------------------------
#   @task
#   def load_data(): ...
#
#   @task
#   def preprocess(data): ...
#
#   @flow
#   def my_pipeline():
#       data = load_data()           # Vous appelez explicitement
#       features = preprocess(data)  # Vous passez les données à la main
#
# DAGSTER (Asset-oriented) - Ce que vous allez découvrir :
# --------------------------------------------------------
#   @asset
#   def raw_data(): ...
#
#   @asset
#   def features(raw_data):  # Le NOM de l'argument = la dépendance !
#       ...                  # Dagster câble automatiquement
#
# La différence clé : Dans Dagster, vous déclarez ce qui DOIT EXISTER,
# pas ce qui doit S'EXÉCUTER. Dagster déduit le graphe tout seul !
#
# =============================================================================


# =============================================================================
# DÉFI 1 : LA MAGIE DE L'AUTO-CÂBLAGE
# =============================================================================
#
# Ouvrez http://localhost:3000 après avoir lancé `dagster dev -f Dagster_Exercises.py`
# Regardez le graphe : il est CASSÉ !
#
# Dagster cherche un asset nommé "donnees_entree" qui n'existe pas.
# -> Corrigez l'argument pour qu'il corresponde EXACTEMENT au nom de l'asset.
# -> Sauvegardez (Ctrl+S) et regardez l'UI se mettre à jour en temps réel !
#
# =============================================================================

@asset(
    description="Données clients brutes",
    compute_kind="pandas"
)
def raw_customer_data() -> pd.DataFrame:
    """Simule le chargement des données brutes."""
    print("📂 Chargement des données brutes...")
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 45, 22, 50],
        "monetary": [100, 200, 500, 50, 1000],
        "frequency": [2, 5, 10, 1, 20],
        "churn": [0, 1, 0, 0, 1]
    })


# TODO 1 : Renommez "donnees_entree" en "raw_customer_data"
#          Sauvegardez et observez le graphe se connecter dans l'UI !
@asset(
    description="Features préparées pour l'entraînement",
    compute_kind="pandas"
)
def preprocessed_features(donnees_entree: pd.DataFrame) -> pd.DataFrame:  # <-- MODIFIEZ ICI
    """Nettoie et normalise les données."""
    print("🔧 Preprocessing des données...")

    # Simulation de preprocessing
    df = donnees_entree.copy()
    df["age_normalized"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["monetary_normalized"] = (df["monetary"] - df["monetary"].mean()) / df["monetary"].std()

    return df


# =============================================================================
# DÉFI 2 : AJOUTER UN ASSET AU GRAPHE
# =============================================================================
#
# La fonction ci-dessous n'est PAS encore un asset Dagster.
# -> Ajoutez le décorateur @asset
# -> Sauvegardez et regardez un 3ème nœud apparaître dans le graphe !
#
# REMARQUE IMPORTANTE : Le code Python NE S'EXÉCUTE PAS quand vous sauvegardez.
# Dagster analyse juste les décorateurs pour dessiner le plan d'architecture.
# L'exécution ne se fait que quand vous cliquez sur "Materialize" !
#
# =============================================================================

class ModelConfig(Config):
    """Configuration du modèle (modifiable dans l'UI Dagster)."""
    n_estimators: int = 100
    max_depth: int = 5


# TODO 2 : Ajoutez @asset au-dessus de cette fonction
#          N'oubliez pas : description="...", compute_kind="sklearn"
def trained_model(preprocessed_features: pd.DataFrame, config: ModelConfig):
    """Entraîne le modèle et retourne des métadonnées."""
    from sklearn.ensemble import RandomForestClassifier

    print(f"🎯 Entraînement avec n_estimators={config.n_estimators}, max_depth={config.max_depth}")

    # Préparation des données
    feature_cols = ["age_normalized", "monetary_normalized"]
    feature_cols = [c for c in feature_cols if c in preprocessed_features.columns]

    if not feature_cols:
        feature_cols = ["age", "monetary", "frequency"]

    X = preprocessed_features[feature_cols].fillna(0)
    y = preprocessed_features["churn"] if "churn" in preprocessed_features.columns else np.zeros(len(X))

    # Entraînement
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=42
    )
    model.fit(X, y)

    # Métriques simulées
    accuracy = 0.82 + np.random.random() * 0.1

    # -------------------------------------------------------------------------
    # MÉTADONNÉES DAGSTER : S'affichent dans l'interface !
    # -------------------------------------------------------------------------
    # Au lieu de juste "return model", on retourne un MaterializeResult
    # qui dit à l'interface Dagster d'afficher les métriques.
    # Cliquez sur l'asset dans l'UI pour voir ces valeurs !

    return MaterializeResult(
        metadata={
            "accuracy": MetadataValue.float(accuracy),
            "n_estimators": MetadataValue.int(config.n_estimators),
            "max_depth": MetadataValue.int(config.max_depth),
            "lignes_entrainees": MetadataValue.int(len(X)),
            "features_utilisees": MetadataValue.text(", ".join(feature_cols))
        }
    )


# =============================================================================
# DÉFI 3 : INTÉGRATION MLFLOW (Bonus)
# =============================================================================
#
# Décommentez le code MLflow dans cet asset pour logger dans MLflow
# ET dans les métadonnées Dagster en même temps !
#
# =============================================================================

# TODO 3 : Ajoutez @asset et décommentez le code MLflow
def trained_model_with_mlflow(preprocessed_features: pd.DataFrame, config: ModelConfig):
    """Entraîne le modèle avec tracking MLflow."""
    from sklearn.ensemble import RandomForestClassifier

    print(f"🎯 Entraînement avec MLflow tracking...")

    # Préparation
    X = preprocessed_features[["age", "monetary", "frequency"]].fillna(0)
    y = preprocessed_features["churn"] if "churn" in preprocessed_features.columns else np.zeros(len(X))

    # Entraînement
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=42
    )
    model.fit(X, y)
    accuracy = 0.82 + np.random.random() * 0.1

    # -------------------------------------------------------------------------
    # TODO : Décommentez le bloc MLflow ci-dessous
    # -------------------------------------------------------------------------
    # import mlflow
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment("dagster-churn-experiment")
    #
    # with mlflow.start_run(run_name="dagster-asset-run"):
    #     mlflow.log_params({
    #         "n_estimators": config.n_estimators,
    #         "max_depth": config.max_depth
    #     })
    #     mlflow.log_metric("accuracy", accuracy)
    #     print(f"✅ Loggé dans MLflow : {MLFLOW_TRACKING_URI}")

    return MaterializeResult(
        metadata={
            "accuracy": MetadataValue.float(accuracy),
            "mlflow_experiment": MetadataValue.text("dagster-churn-experiment"),
        }
    )


# =============================================================================
# DÉFI 4 : EXPÉRIMENTATION DEPUIS L'INTERFACE (Le vrai pouvoir !)
# =============================================================================
#
# 🚀 Ne codez plus rien ! Allez dans l'interface http://localhost:3000 :
#
# 1. MATÉRIALISER TOUT :
#    - Cliquez sur "Materialize All" (en haut à droite)
#    - Observez les assets se calculer un par un
#    - Vérifiez les métadonnées en cliquant sur chaque asset
#
# 2. VOIR LES MÉTADONNÉES :
#    - Cliquez sur l'asset "trained_model" dans le graphe
#    - Regardez le panneau de droite : l'accuracy s'affiche !
#    - C'est le MaterializeResult qui fait ça
#
# 3. LE TEST ULTIME - Matérialisation Partielle :
#    - Vous voulez tester n_estimators = 500
#    - Dans Prefect, vous auriez dû relancer TOUT le pipeline
#      (ou configurer un cache complexe avec task_input_hash)
#    - Dans Dagster :
#      a) Cliquez UNIQUEMENT sur l'asset "trained_model" dans le graphe
#      b) Cliquez sur "Materialize selected"
#      c) Dans le Launchpad, changez la configuration (n_estimators: 500)
#      d) Lancez !
#    - OBSERVEZ : Dagster NE RELANCE PAS raw_customer_data ni preprocessed_features
#      Il réutilise les données déjà matérialisées !
#
# C'est LA killer-feature de Dagster pour l'expérimentation ML :
# On peut modifier les paramètres du modèle et relancer SANS recharger
# les térabytes de données du début du pipeline.
#
# =============================================================================


# =============================================================================
# DÉFINITIONS DAGSTER (obligatoire pour dagster dev)
# =============================================================================

# Liste des assets à exposer
# NOTE: Ajoutez trained_model à cette liste après l'avoir décoré avec @asset
defs = Definitions(
    assets=[
        raw_customer_data,
        preprocessed_features,
        # trained_model,           # <-- Décommentez après TODO 2
        # trained_model_with_mlflow,  # <-- Décommentez après TODO 3
    ]
)


# =============================================================================
# MODE STANDALONE (pour tester sans l'interface)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎓 DAGSTER EXERCISES - Le Changement de Paradigme")
    print("=" * 70)
    print("""
LANCEMENT RECOMMANDÉ :
  docker-compose --profile dagster up -d

Puis ouvrez http://localhost:3000

MODE STANDALONE (si dagster dev ne fonctionne pas) :
  python Dagster_Exercises.py

PROGRESSION :
  1. Corrigez le graphe cassé (TODO 1)
  2. Ajoutez l'asset trained_model (TODO 2)
  3. Ajoutez MLflow (TODO 3 - bonus)
  4. Expérimentez dans l'interface ! (DÉFI 4)

RÉFÉRENCE :
  Consultez Dagster_Workshop.py pour les solutions complètes.
""")

    # Exécution standalone pour test
    from dagster import materialize

    print("\n📊 Matérialisation des assets...\n")

    result = materialize(
        assets=[raw_customer_data, preprocessed_features],
    )

    if result.success:
        print("\n✅ Assets matérialisés avec succès !")
        print("   Pour voir le graphe interactif, utilisez : dagster dev -f Dagster_Exercises.py")
    else:
        print("\n❌ Erreur lors de la matérialisation")
