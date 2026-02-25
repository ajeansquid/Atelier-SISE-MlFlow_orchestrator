# =============================================================================
# Atelier Prefect - EXERCICES (Approche Fil Rouge)
# =============================================================================
#
# 🎯 OBJECTIF : Transformer un script ML classique en pipeline de production
#
# Vous allez progressivement "blinder" un vrai pipeline de prédiction de Churn
# avec tous les patterns d'orchestration professionnels :
#
#   ÉTAPE 1 : Tasks & Flows (transformer les fonctions en tâches orchestrées)
#   ÉTAPE 2 : Résilience (ajouter des réessais sur le chargement)
#   ÉTAPE 3 : Efficacité (cacher le preprocessing coûteux + parallélisme)
#   ÉTAPE 4 : MLflow + sklearn Pipeline (tracker avec la meilleure pratique)
#   ÉTAPE 5 : Orchestration (organiser en sous-flows)
#   ÉTAPE 6 : Déploiement (planifier l'exécution automatique)
#   ÉTAPE 7 : Notifications (alertes Discord/Slack en cas d'échec)
#
# PRÉREQUIS (à faire UNE SEULE FOIS avant de commencer) :
#
#   1. Vérifiez que Docker est lancé :
#      docker-compose ps
#
#   2. Copiez le fichier .env.example et renommez-le en .env
#      (ou en terminal : cp .env.example .env sur Mac/Linux, copy .env.example .env sur Windows)
#
#   C'est tout ! Le fichier .env configure automatiquement la connexion
#   aux serveurs Prefect et MLflow dans Docker.
#
# EXÉCUTION :
#   python Prefect_Exercises.py etape1    # Commencer ici !
#   python Prefect_Exercises.py etape2    # Puis continuer...
#   python Prefect_Exercises.py etape3
#   python Prefect_Exercises.py etape4
#   python Prefect_Exercises.py etape5
#   python Prefect_Exercises.py deploy    # Le clou du spectacle !
#   python Prefect_Exercises.py notif     # Bonus : alertes
#
# INTERFACES :
#   Prefect : http://localhost:4200
#   MLflow  : http://localhost:5000
#
# RÉFÉRENCE : Consultez Prefect_Workshop.py pour voir les solutions complètes.
#
# =============================================================================

import sys
import os
import time
from pathlib import Path
from datetime import timedelta, datetime

import mlflow
import pandas as pd
import numpy as np

# Charger automatiquement le fichier .env (PREFECT_API_URL, MLFLOW_TRACKING_URI)
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_exercises.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# =============================================================================
# LE PIPELINE DE CHURN - Code de départ (sans orchestration)
# =============================================================================
#
# Voici un pipeline ML classique. Votre mission : le transformer en pipeline
# de production robuste avec Prefect !
#
# =============================================================================


def load_churn_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Charger les données clients.

    Dans un contexte réel, cela pourrait être une requête vers une base de
    données ou une API qui peut échouer temporairement.
    """
    print(f"📂 Chargement des données depuis {file_path}...")

    if Path(file_path).exists():
        df = pd.read_csv(file_path)
    else:
        # Données synthétiques si le fichier n'existe pas
        print("⚠️  Fichier non trouvé, génération de données synthétiques...")
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 70, n_samples),
            'recency_days': np.random.randint(1, 365, n_samples),
            'frequency': np.random.randint(1, 50, n_samples),
            'monetary_value': np.random.uniform(10, 1000, n_samples),
            'churned': np.random.binomial(1, 0.3, n_samples)
        })

    print(f"✅ Chargé {len(df)} clients")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Prétraitement des données.

    Cette étape est souvent coûteuse (feature engineering, scaling, encoding).
    C'est un candidat idéal pour le cache !
    """
    print("🔧 Preprocessing des données...")
    time.sleep(1)  # Simule un preprocessing coûteux

    # Feature engineering
    feature_cols = ['recency_days', 'frequency', 'monetary_value', 'age']
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in ['customer_id', 'churned']]

    X = df[feature_cols].fillna(0)
    y = df['churned'] if 'churned' in df.columns else pd.Series(np.zeros(len(df)))

    # Normalisation simple
    X = (X - X.mean()) / (X.std() + 1e-8)

    print(f"✅ Features : {list(X.columns)}, Shape : {X.shape}")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, max_depth: int = 10) -> dict:
    """
    Entraîner un modèle RandomForest.

    Les hyperparamètres devraient être configurables !
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    print(f"🎯 Entraînement du modèle (n_estimators={n_estimators}, max_depth={max_depth})...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    print(f"✅ Accuracy : {metrics['accuracy']:.4f}, F1 : {metrics['f1']:.4f}")
    return {"model": model, "metrics": metrics, "feature_cols": list(X.columns)}


def save_predictions(df: pd.DataFrame, model, feature_cols: list) -> str:
    """Sauvegarder les prédictions."""
    print("💾 Sauvegarde des prédictions...")

    X_pred = df[feature_cols].fillna(0)
    X_pred = (X_pred - X_pred.mean()) / (X_pred.std() + 1e-8)

    predictions = model.predict(X_pred)
    df_out = df[['customer_id']].copy()
    df_out['predicted_churn'] = predictions

    df_out.to_csv(PREDICTIONS_PATH, index=False)
    print(f"✅ Prédictions sauvegardées : {PREDICTIONS_PATH}")
    return PREDICTIONS_PATH


# =============================================================================
# ÉTAPE 1 : TASKS & FLOWS - Les Bases
# =============================================================================
#
# OBJECTIF : Transformer les fonctions ci-dessus en tâches Prefect
#
# CONCEPTS :
#   - @task : Transforme une fonction en tâche orchestrée (logs, UI, etc.)
#   - @flow : Orchestre plusieurs tâches, point d'entrée du pipeline
#   - Les données circulent via les valeurs de retour (pas de fichiers !)
#
# =============================================================================

def run_etape1():
    """
    ÉTAPE 1 : Créer votre premier flow Prefect

    Transformez le code métier en tâches orchestrées :
    1. Décorer load_churn_data comme une @task
    2. Décorer preprocess_data comme une @task
    3. Décorer train_model comme une @task
    4. Créer un @flow qui les orchestre
    """
    from prefect import flow, task

    print("=" * 70)
    print("ÉTAPE 1 : TASKS & FLOWS - Transformer le code en pipeline Prefect")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TODO : Ajoutez le décorateur @task à cette fonction
    # INDICE : @task se place juste au-dessus de "def"
    # -------------------------------------------------------------------------
    @task(name="chargement-donnees")
    def load_data() -> pd.DataFrame:
        """Charger les données clients."""
        return load_churn_data(DATA_PATH)

    # -------------------------------------------------------------------------
    # TODO : Ajoutez le décorateur @task à cette fonction
    # -------------------------------------------------------------------------
    @task(name="Preparation-features")
    def prepare_features(df: pd.DataFrame) -> tuple:
        """Préparer les features."""
        return preprocess_data(df)

    # -------------------------------------------------------------------------
    # TODO : Ajoutez le décorateur @task à cette fonction
    # -------------------------------------------------------------------------
    @task(name="entrainement-modele")
    def train(X: pd.DataFrame, y: pd.Series) -> dict:
        """Entraîner le modèle."""
        return train_model(X, y)
    
    @task(name="sauvegarde-predictions")
    def save_pred(df: pd.DataFrame, model, feature_cols: list) -> str:
        """Sauvegarder les prédictions."""
        return save_predictions(df, model, feature_cols)

    # -------------------------------------------------------------------------
    # TODO : Créez un flow qui orchestre ces tâches
    # INDICE : @flow(name="churn-pipeline-v1", log_prints=True)
    # -------------------------------------------------------------------------
    @flow(name="churn-pipeline-v1", log_prints=True)
    def churn_pipeline_v1():
        """
        Pipeline de prédiction de Churn - Version 1

        TODO : Appelez les tâches dans l'ordre :
        1. data = load_data()
        2. X, y = prepare_features(data)
        3. result = train(X, y)
        4. Retournez result
        """
        # TODO : Complétez le pipeline
        data = load_data()       # <-- Remplacez par load_data()
        X, y = prepare_features(data)  # <-- Remplacez par prepare_features(data)
        result = train(X, y)     # <-- Remplacez par train(X, y)
        save_pred(data, result['model'], result['feature_cols'])
        if result:
            print(f"🎉 Pipeline terminé ! Accuracy : {result['metrics']['accuracy']:.4f}")

        return result

    # Exécuter
    print("\n📋 Instructions :")
    print("   1. Ajoutez @task aux 3 fonctions (load_data, prepare_features, train)")
    print("   2. Ajoutez @flow à churn_pipeline_v1")
    print("   3. Complétez les appels dans le flow\n")

    try:
        result = churn_pipeline_v1()
        if result and result.get('metrics', {}).get('accuracy', 0) > 0:
            print("\n✅ ÉTAPE 1 RÉUSSIE !")
            print("   Vous avez créé votre premier pipeline Prefect !")
            print("   Passez à l'étape 2 : python Prefect_Exercises.py etape2")
        else:
            print("\n❌ Le flow n'a pas retourné de résultat.")
            print("   Vérifiez que vous avez remplacé les None par les appels de fonctions.")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("   INDICE : Avez-vous ajouté les décorateurs @task et @flow ?")
        
    # Rappel : Pour éxécuter cette étape, utilisez la commande :
    # uv run .\pipelines\workshop\02_prefect\Prefect_Exercises.py etape1 (depuis le terminal à la racine du projet)

    # -------------------------------------------------------------------------
    # MINI-DÉFIS (après avoir complété l'exercice) :
    # -------------------------------------------------------------------------
    #
    # DÉFI 1.1 : Ouvrez http://localhost:4200 et retrouvez votre flow.
    #            Explorez les détails de l'exécution (tâches, durées, logs).
    #
    # DÉFI 1.2 : Ajoutez name="chargement-donnees" au décorateur @task de
    #            load_data. Regardez comment ça change l'affichage dans l'UI.
    #
    # DÉFI 1.3 : Ajoutez une 4ème tâche 'save_results' qui sauvegarde les
    #            métriques dans un fichier JSON.
    # -------------------------------------------------------------------------


# =============================================================================
# ÉTAPE 2 : RÉSILIENCE - Réessais Automatiques
# =============================================================================
#
# OBJECTIF : Rendre le chargement de données robuste aux échecs transitoires
#
# CONCEPTS :
#   - retries : Nombre de tentatives après échec
#   - retry_delay_seconds : Délai entre les tentatives
#   - Backoff exponentiel : [5, 15, 30] = attendre de plus en plus longtemps
#
# SCÉNARIO RÉEL : Une base de données temporairement indisponible, une API
# avec rate limiting, un fichier réseau inaccessible...
#
# =============================================================================

def run_etape2():
    """
    ÉTAPE 2 : Ajouter la résilience au chargement de données

    Le chargement de données peut échouer (réseau, base de données...).
    Configurez des réessais automatiques !
    """
    from prefect import flow, task
    import random

    print("=" * 70)
    print("ÉTAPE 2 : RÉSILIENCE - Réessais automatiques sur le chargement")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TODO : Configurez cette tâche pour réessayer 3 fois avec backoff [5, 10, 20]
    # INDICE : @task(retries=3, retry_delay_seconds=[5, 10, 20])
    # -------------------------------------------------------------------------
    @task(retries=3, retry_delay_seconds=[2, 8, 32])
    def load_data_with_retry() -> pd.DataFrame:
        """
        Charger les données avec simulation d'échecs.

        Dans un vrai scénario, remplacez random.random() par une vraie
        requête vers une base de données ou une API.
        """
        # Simulation d'échec transitoire (40% de chance d'échec)
        if random.random() < 0.8:
            print("❌ Connexion échouée ! (simulation)")
            raise ConnectionError("Database temporarily unavailable")

        return load_churn_data(DATA_PATH)

    @task
    def prepare_features(df: pd.DataFrame) -> tuple:
        return preprocess_data(df)

    @task
    def train(X: pd.DataFrame, y: pd.Series) -> dict:
        return train_model(X, y)

    @flow(name="churn-pipeline-v2-resilient", log_prints=True)
    def churn_pipeline_v2():
        """Pipeline avec résilience."""
        data = load_data_with_retry()
        X, y = prepare_features(data)
        result = train(X, y)
        return result

    # Exécuter
    print("\n📋 Instructions :")
    print("   1. Ajoutez retries=3 et retry_delay_seconds=[5, 10, 20] à load_data_with_retry")
    print("   2. Exécutez plusieurs fois pour voir les réessais en action\n")

    random.seed(None)  # Seed aléatoire pour voir les échecs

    try:
        result = churn_pipeline_v2()
        print("\n✅ ÉTAPE 2 RÉUSSIE !")
        print("   Votre pipeline est maintenant résilient aux échecs transitoires !")
        print("   Passez à l'étape 3 : python Prefect_Exercises.py etape3")
    except Exception as e:
        print(f"\n⚠️  Le pipeline a échoué après tous les réessais : {e}")
        print("   C'est normal si la malchance s'acharne ! Réexécutez.")
        print("   Vérifiez que vous avez bien configuré retries=3.")

    # -------------------------------------------------------------------------
    # MINI-DÉFIS :
    # -------------------------------------------------------------------------
    #
    # DÉFI 2.1 : Changez la probabilité d'échec de 0.4 à 0.8.
    #            Combien de fois voyez-vous "Connexion échouée !" avant succès ?
    #
    # DÉFI 2.2 : Ajoutez un backoff plus agressif : [2, 8, 32] secondes.
    #            Chronométrez le temps total en cas d'échecs multiples.
    #            (Globalement, vous devriez voir des temps d'attente de plus en plus longs : 2s, puis 8s, puis 32s
    #            (et avec peu de retry c'est possible que le Flux s'arrête)
    #
    # DÉFI 2.3 : Dans un vrai projet, où mettriez-vous les réessais ?
    #            (Indice : pas sur le preprocessing !)
    # -------------------------------------------------------------------------
    
    # Rappel : Pour éxécuter cette étape, utilisez la commande :
    # uv run .\pipelines\workshop\02_prefect\Prefect_Exercises.py etape2 (depuis le terminal à la racine du projet)


# =============================================================================
# ÉTAPE 3 : EFFICACITÉ - Cache du Preprocessing + Parallélisme
# =============================================================================
#
# OBJECTIF : Éviter de refaire le preprocessing si les données n'ont pas changé,
#            et entraîner plusieurs modèles simultanément
#
# CONCEPTS CACHE :
#   - cache_key_fn : Fonction qui génère une clé de cache (hash des inputs)
#   - cache_expiration : Durée de validité du cache
#   - task_input_hash : Fonction prête à l'emploi pour hasher les inputs
#
# CONCEPTS PARALLÉLISME :
#   - task()        : Exécution séquentielle (attend la fin avant de continuer)
#   - task.submit() : Exécution parallèle (retourne un Future immédiatement)
#   - future.result() : Récupère le résultat quand la tâche est terminée
#
# SCÉNARIO RÉEL : Feature engineering coûteux (30 min), données qui ne changent
# que quotidiennement → cacher pendant 1h évite 90% des recalculs !
#
# =============================================================================

def run_etape3():
    """
    ÉTAPE 3 : Cacher le preprocessing coûteux

    Le preprocessing prend 1 seconde (simulé). Sur de vraies données,
    ça peut prendre 30 minutes ! Configurez le cache.
    """
    from prefect import flow, task
    from prefect.tasks import task_input_hash

    print("=" * 70)
    print("ÉTAPE 3 : EFFICACITÉ - Cache du preprocessing")
    print("=" * 70)

    @task(retries=3, retry_delay_seconds=[5, 10, 20])
    def load_data() -> pd.DataFrame:
        return load_churn_data(DATA_PATH)

    # -------------------------------------------------------------------------
    # TODO : Activez le cache sur cette tâche
    # INDICE : @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
    # -------------------------------------------------------------------------
    @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
    def prepare_features_cached(df: pd.DataFrame) -> tuple:
        """
        Preprocessing avec cache.

        La première exécution prendra ~1 seconde.
        Les suivantes seront instantanées (cache hit) !

        NOTE: Lors d'un cache hit, Prefect n'exécute PAS le code de la fonction.
        Le print ci-dessous n'apparaîtra donc pas la 2ème fois !
        """
        print("⏳ Preprocessing en cours (1 seconde)...")
        return preprocess_data(df)

    @task
    def train(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100) -> dict:
        return train_model(X, y, n_estimators=n_estimators)

    @flow(name="churn-pipeline-v3-cached", log_prints=True)
    def churn_pipeline_v3():
        """Pipeline avec cache."""
        data = load_data()

        print("\n=== Première exécution du preprocessing ===")
        start = time.time()
        X, y = prepare_features_cached(data)
        time1 = time.time() - start
        print(f"⏱️  Temps : {time1:.2f}s")

        print("\n=== Deuxième exécution (devrait utiliser le cache) ===")
        start = time.time()
        X2, y2 = prepare_features_cached(data)  # Mêmes données = cache hit !
        time2 = time.time() - start
        print(f"⏱️  Temps : {time2:.2f}s")

        # Vérification du cache
        if time2 < 0.5 and time1 > 0.8:
            print("\n✅ CACHE FONCTIONNE ! La 2ème exécution était instantanée.")
            print("   (Notez que le print 'Preprocessing en cours' n'est pas apparu !)")
        else:
            print("\n⚠️  Le cache ne semble pas actif. Vérifiez les paramètres.")

        result = train(X, y)
        return result

    # Exécuter
    print("\n📋 Instructions :")
    print("   1. Ajoutez cache_key_fn=task_input_hash à prepare_features_cached")
    print("   2. Ajoutez cache_expiration=timedelta(hours=1)")
    print("   3. Observez les temps d'exécution\n")

    try:
        result = churn_pipeline_v3()
        print("\n✅ ÉTAPE 3 TERMINÉE !")
        print("   Passez à l'étape 4 : python Prefect_Exercises.py etape4")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")

    # -------------------------------------------------------------------------
    # MINI-DÉFIS :
    # -------------------------------------------------------------------------
    #
    # DÉFI 3.1 : Changez cache_expiration à timedelta(seconds=10).
    #            Attendez 15 secondes et réexécutez. Le cache est-il invalidé ?
    #
    # DÉFI 3.2 : Modifiez légèrement le DataFrame (ajoutez une colonne).
    #            Que se passe-t-il avec le cache ?
    #
    # DÉFI 3.3 : Où NE PAS mettre de cache ? (Indice : l'entraînement du modèle
    #            avec des hyperparamètres différents ne devrait PAS être caché)
    #
    # DÉFI 3.4 (BONUS) : PARALLÉLISME - Entraîner deux modèles simultanément
    # -------------------------------------------------------------------------
    # Dans Prefect, appeler une tâche normalement = séquentiel.
    # Utiliser .submit() = parallèle : les deux tâches démarrent en même temps !
    # -------------------------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    @task
    def train_rf(X_train, y_train, X_test, y_test) -> dict:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        return {"name": "RandomForest", "accuracy": accuracy_score(y_test, model.predict(X_test))}

    @task
    def train_gb(X_train, y_train, X_test, y_test) -> dict:
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        return {"name": "GradientBoosting", "accuracy": accuracy_score(y_test, model.predict(X_test))}

    @flow(name="parallel-training-demo", log_prints=True)
    def parallel_training_demo():
        data = load_churn_data(DATA_PATH)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # VERSION PARALLÈLE avec .submit() :
        rf_future = train_rf.submit(X_train, y_train, X_test, y_test)  # démarre immédiatement
        gb_future = train_gb.submit(X_train, y_train, X_test, y_test)  # démarre aussi !
        rf, gb = rf_future.result(), gb_future.result()                 # attend les deux

        best = max([rf, gb], key=lambda x: x["accuracy"])
        print(f"Meilleur modèle : {best['name']} ({best['accuracy']:.4f})")

    print("\n" + "=" * 60)
    print("DÉFI 3.4 : PARALLÉLISME")
    print("=" * 60)
    print("SÉQUENTIEL → rf démarre, finit, PUIS gb démarre")
    print("PARALLÈLE  → rf ET gb démarrent en même temps (.submit())")
    print("Observez dans l'UI Prefect : les deux tâches s'exécutent simultanément !\n")
    parallel_training_demo()
    # -------------------------------------------------------------------------
    # Rappel : Pour éxecuter cette étape :


# =============================================================================
# ÉTAPE 4 : MLFLOW - Tracking des Expérimentations
# =============================================================================
#
# OBJECTIF : Logger les paramètres, métriques et modèles dans MLflow
#
# CONCEPTS :
#   - Prefect gère : réessais, cache, planification, logs d'exécution
#   - MLflow gère : paramètres, métriques, artefacts, versioning des modèles
#   - Les appels MLflow se font DANS les tâches Prefect
#   - sklearn Pipeline : combine scaler + modèle en UN SEUL artefact (best practice !)
#
# =============================================================================

def run_etape4():
    """
    ÉTAPE 4 : Intégrer MLflow pour le tracking

    Ajoutez le tracking MLflow avec sklearn Pipeline.
    """
    from prefect import flow, task
    from prefect.tasks import task_input_hash
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler

    # Vérifier si MLflow est disponible
    try:
        import mlflow
        import mlflow.sklearn
        MLFLOW_AVAILABLE = True
    except ImportError:
        MLFLOW_AVAILABLE = False
        print("⚠️  MLflow non installé. Exercice en mode simulation.")

    print("=" * 70)
    print("ÉTAPE 4 : MLFLOW + SKLEARN PIPELINE")
    print("=" * 70)
    print("""
POURQUOI SKLEARN PIPELINE ?

Au lieu de gérer scaler et modèle séparément :
   scaler.fit_transform(X)
   model.fit(X_scaled, y)
   mlflow.log_artifact("scaler.pkl")  # Artefact 1
   mlflow.log_model(model)            # Artefact 2
   # → 2 artefacts, risque d'oublier le scaler à l'inférence !

On combine tout dans un Pipeline :
   pipeline = Pipeline([('scaler', StandardScaler()), ('model', RF())])
   pipeline.fit(X, y)
   mlflow.log_model(pipeline)  # UN SEUL artefact !
   # → À l'inférence : pipeline.predict(X) fait TOUT
""")

    @task(retries=3, retry_delay_seconds=[5, 10, 20])
    def load_data() -> pd.DataFrame:
        return load_churn_data(DATA_PATH)

    @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
    def prepare_features(df: pd.DataFrame) -> tuple:
        return preprocess_data(df)

    # -------------------------------------------------------------------------
    # TODO : Complétez l'intégration MLflow avec sklearn Pipeline
    # -------------------------------------------------------------------------
    @task
    def train_with_mlflow(
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int,
        max_depth: int,
        experiment_name: str
    ) -> dict:
        """Entraîner avec sklearn Pipeline et logger dans MLflow."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ---------------------------------------------------------------------
        # TODO : Créez le sklearn Pipeline
        # INDICE :
        #   pipeline = SklearnPipeline([
        #       ('scaler', StandardScaler()),
        #       ('model', RandomForestClassifier(n_estimators=..., max_depth=..., random_state=42))
        #   ])
        # ---------------------------------------------------------------------
        pipeline = SklearnPipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            ))
        ])

        # Entraîner le pipeline (scaler + modèle en une seule ligne)
        pipeline.fit(X_train, y_train)

        # Prédire (le pipeline applique automatiquement le scaling)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if not MLFLOW_AVAILABLE:
            print(f"📊 [SIMULATION] Pipeline entraîné : Accuracy={accuracy:.4f}")
            return {"pipeline": pipeline, "accuracy": accuracy, "run_id": "simulation"}

        # ---------------------------------------------------------------------
        # TODO : Configurez MLflow
        # INDICE : mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        #          mlflow.set_experiment(experiment_name='prefect-training')
        # ---------------------------------------------------------------------
        pass  # <-- Remplacez par la configuration MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name='prefect-training')

        # ---------------------------------------------------------------------
        # TODO : Démarrez un run MLflow et loggez le pipeline
        # INDICE : with mlflow.start_run(run_name="prefect-training"):
        # ---------------------------------------------------------------------
        # Décommentez et complétez :
        
        with mlflow.start_run(run_name=f"training-{datetime.now().strftime('%H%M%S')}"):
        
            # TODO : Loggez les paramètres
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        
            # TODO : Loggez les métriques
            mlflow.log_metrics({"accuracy": accuracy, "f1": f1})
        
            # TODO : Loggez le PIPELINE (scaler + modèle en UN artefact !)
            mlflow.sklearn.log_model(pipeline, name="model")
            # NOTE : Par défaut, MLflow sérialise en pickle (risque de sécurité).
            # Alternatives possibles selon le contexte :
            #
            # - skops (recommandé en production, Python uniquement) :
            #   mlflow.sklearn.log_model(pipeline, name="model", serialization_format="skops")
            #   ✅ Plus sûr (pas d'exécution de code arbitraire à la désérialisation)
            #   ✅ Supporté nativement par MLflow
            #   ❌ Plus récent → ecosystème moins mature, moins de ressources en ligne
            #   ❌ Moins flexible (certains objets custom non supportés)
            #   ❌ Python uniquement, comme pickle
            #
            # - ONNX (interopérable, multi-langages) :
            #   ✅ Permet de servir le modèle depuis Java, C#, JavaScript, etc.
            #   ✅ Idéal si l'inférence se fait hors Python (API Java, appli mobile...)
            #   ✅ Performances d'inférence souvent meilleures (runtime optimisé)
            #   ❌ Conversion parfois complexe, tous les modèles ne sont pas supportés
            #   ❌ Le Pipeline sklearn peut poser des problèmes à la conversion
            #   import onnxmltools; mlflow.onnx.log_model(...)
        
            # ---------------------------------------------------------------------
            # TODO : Loggez la matrice de confusion avec mlflow.log_figure()
            #
            # La figure est déjà créée ci-dessous, il vous reste à la logger !
            # INDICE : mlflow.log_figure(fig, "confusion_matrix.png")
            # ---------------------------------------------------------------------
            from sklearn.metrics import ConfusionMatrixDisplay
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            ax.set_title("Matrice de confusion")

            # TODO : Loggez la figure dans MLflow ici
            mlflow.log_figure(fig, "confusion_matrix.png")

            plt.close(fig)  # Libérer la mémoire

            run_id = mlflow.active_run().info.run_id
            print(f"✅ MLflow Run ID : {run_id}")
            print(f"   Accuracy : {accuracy:.4f}, F1 : {f1:.4f}")
            print(f"   ⭐ Pipeline loggé (scaler + model en UN artefact)")
        
        return {"pipeline": pipeline, "accuracy": accuracy, "f1": f1, "run_id": run_id}

        # En attendant que vous complétiez, version sans tracking :
        print(f"⚠️  MLflow non configuré. Accuracy : {accuracy:.4f}")
        return {"pipeline": pipeline, "accuracy": accuracy, "run_id": None}

    # -------------------------------------------------------------------------
    # TODO : Ajoutez des paramètres au flow avec des valeurs par défaut
    # INDICE : def churn_pipeline_v4(n_estimators: int = 100, max_depth: int = 10):
    # -------------------------------------------------------------------------
    @flow(name="churn-pipeline-v4-mlflow", log_prints=True)
    def churn_pipeline_v4(n_estimators: int = 100, max_depth: int = 10):  # <-- Ajoutez les paramètres ici
        """Pipeline avec intégration MLflow."""
        data = load_data()
        X, y = prepare_features(data)
        
        # TODO : Défi 4.3 : Enregistrez le modèle dans le Model Registry avec mlflow.register_model()
        # INDICE : mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
        mlflow.register_model(f"runs:/{train_with_mlflow(X, y, n_estimators, max_depth, 'prefect-churn-exercises')['run_id']}/model", "churn-model")

        # TODO : Utilisez les paramètres du flow au lieu de valeurs fixes
        result = train_with_mlflow(
            X, y,
            n_estimators=n_estimators,  # <-- Remplacez par le paramètre du flow
            max_depth=max_depth,      # <-- Remplacez par le paramètre du flow
            experiment_name="prefect-churn-exercises"
        )

        return result

    # Exécuter
    print("\n📋 Instructions :")
    print("   1. Décommentez le bloc MLflow dans train_with_mlflow")
    print("   2. Ajoutez des paramètres au flow (n_estimators, max_depth)")
    print("   3. Vérifiez les résultats dans http://localhost:5000\n")

    if MLFLOW_AVAILABLE:
        print(f"📊 MLflow tracking : {MLFLOW_TRACKING_URI}\n")

    try:
        # result = churn_pipeline_v4(n_estimators=100, max_depth=10)
        for n_est, depth in [(50, 5), (200, 20), (150, 15)]:
            result = churn_pipeline_v4(n_estimators=n_est, max_depth=depth)
        print("\n✅ ÉTAPE 4 TERMINÉE !")
        print("   Passez à l'étape 5 : python Prefect_Exercises.py etape5")
        if MLFLOW_AVAILABLE and result.get("run_id"):
            print(f"   Voir le run : {MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")

    # -------------------------------------------------------------------------
    # MINI-DÉFIS :
    # -------------------------------------------------------------------------
    #
    # DÉFI 4.1 : Exécutez le flow 3 fois avec différents hyperparamètres.
    #            Comparez les résultats dans l'interface MLflow.
    #
    # DÉFI 4.2 : La matrice de confusion est déjà générée dans train_with_mlflow.
    #            Loggez-la dans MLflow avec mlflow.log_figure() et vérifiez qu'elle apparaît dans l'UI.
    #            Elle devrait se trouver dans l'onglet "Artifacts" de MLflow !
    #
    # DÉFI 4.3 : Utilisez mlflow.register_model() pour enregistrer le modèle
    #            dans le Model Registry avec le nom "churn-model".
    # -------------------------------------------------------------------------


# =============================================================================
# ÉTAPE 5 : ORCHESTRATION - Organisation en Sous-Flows
# =============================================================================
#
# OBJECTIF : Structurer le pipeline en sous-flows réutilisables
#
# CONCEPTS :
#   - Sous-flows : Des flows appelés par d'autres flows
#   - Modularité : Réutiliser le preprocessing dans différents pipelines
#   - Lisibilité : Séparer les responsabilités
#
# =============================================================================

def run_etape5():
    """
    ÉTAPE 5 : Organiser en sous-flows

    Structurez le pipeline en modules réutilisables :
    - data_preparation_flow : Chargement + preprocessing
    - training_flow : Entraînement + évaluation
    - main_pipeline : Orchestre les sous-flows
    """
    from prefect import flow, task
    from prefect.tasks import task_input_hash

    print("=" * 70)
    print("ÉTAPE 5 : ORCHESTRATION - Organisation en sous-flows")
    print("=" * 70)

    # === TÂCHES ===

    @task(retries=3, retry_delay_seconds=[5, 10, 20])
    def load_data() -> pd.DataFrame:
        return load_churn_data(DATA_PATH)

    @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
    def prepare_features(df: pd.DataFrame) -> tuple:
        return preprocess_data(df)

    @task
    def train(X, y, n_estimators: int, max_depth: int) -> dict:
        return train_model(X, y, n_estimators=n_estimators, max_depth=max_depth)

    @task
    def save_preds(df: pd.DataFrame, model, feature_cols: list) -> str:
        return save_predictions(df, model, feature_cols)

    # -------------------------------------------------------------------------
    # TODO : Créez un sous-flow pour la préparation des données
    # INDICE : @flow(name="data-preparation")
    # -------------------------------------------------------------------------
    @flow(name="data-preparation", log_prints=True)
    def data_preparation_flow() -> tuple:
        """
        Sous-flow : Préparation des données

        Charge les données et applique le preprocessing.
        Retourne (data, X, y) pour être utilisé par d'autres flows.
        """
        # TODO : Appeler load_data() puis prepare_features()
        data = load_data()
        X, y = prepare_features(data)
        return data, X, y

    # -------------------------------------------------------------------------
    # TODO : Créez un sous-flow pour l'entraînement
    # INDICE : @flow(name="model-training")
    # -------------------------------------------------------------------------
    @flow(name="model-training", log_prints=True)
    def training_flow(X, y, n_estimators: int = 100, max_depth: int = 10, experiment_name: str = "prefect-churn-exercises") -> dict:
        """
        Sous-flow : Entraînement du modèle

        Entraîne et évalue le modèle avec les hyperparamètres donnés.
        """
        # TODO : Décorer cette fonction avec @flow
        # TODO : Appeler train()
        result = train(X, y, n_estimators=n_estimators, max_depth=max_depth)
        return result
    
    # -------------------------------------------------------------------------
    # Défi 5.1 : Flow inference_pipeline
    # TODO : Créez le flow 'inference_pipeline' ici qui réutilise data_preparation_flow
    #        mais charge un modèle existant (au lieu de l'entraîner).
    # -------------------------------------------------------------------------
    @flow(name="inference-pipeline", log_prints=True)
    def inference_pipeline():
        """
        Pipeline d'inférence - Réutilise la préparation des données

        Ce flow pourrait être utilisé pour faire des prédictions sur de nouvelles données
        en réutilisant data_preparation_flow, mais en chargeant un modèle pré-entraîné
        au lieu de l'entraîner à nouveau.
        """
        # TODO : Appeler data_preparation_flow pour obtenir data, X, y
        data, X, y = data_preparation_flow()

        # TODO : Charger un modèle existant (par exemple depuis MLflow ou un fichier)
        #       et faire des prédictions avec save_preds()
        #       (Vous pouvez simuler le chargement du modèle si vous n'avez pas encore de modèle enregistré)
        model = mlflow.sklearn.load_model("models:/churn-model/latest")  # Nom enregistré au Défi 4.3 via mlflow.register_model()
        feature_cols = X.columns.tolist() if X is not None else []
        
        if model:
            save_preds(data, model, feature_cols)
            print("✅ Prédictions sauvegardées avec le modèle chargé.")
        else:
            print("⚠️  Aucun modèle chargé. Simulez le chargement pour tester save_preds.")

    # -------------------------------------------------------------------------
    # TODO : Créez le flow principal qui orchestre les sous-flows
    # -------------------------------------------------------------------------
    @flow(name="churn-pipeline-v5-modular", log_prints=True)
    def churn_pipeline_v5(n_estimators: int = 100, max_depth: int = 10, save_predictions: bool = True, experiment_name: str = "prefect-churn-exercises") -> dict:
        """
        Pipeline principal - Orchestre les sous-flows.

        Cette structure permet de :
        - Réutiliser data_preparation_flow dans d'autres pipelines (inférence)
        - Tester training_flow indépendamment
        - Avoir une vue claire de l'architecture
        """
        print("🚀 Démarrage du pipeline modulaire...")

        # Étape 1 : Préparation des données (sous-flow)
        data, X, y = data_preparation_flow()

        if X is None:
            print("❌ La préparation des données a échoué.")
            print("   Vérifiez que data_preparation_flow est bien un @flow")
            return None

        # Étape 2 : Entraînement (sous-flow)
        result = training_flow(X, y, n_estimators, max_depth, experiment_name=experiment_name)

        if result is None:
            print("❌ L'entraînement a échoué.")
            return None

        # Étape 3 : Sauvegarde des prédictions (optionnel)
        if save_predictions and result.get("model"):
            save_preds(data, result["model"], result["feature_cols"])

        print(f"🎉 Pipeline terminé ! Accuracy : {result['metrics']['accuracy']:.4f}")

        # ---------------------------------------------------------------------
        # DÉFI 5.1 : Lancez le pipeline d'inférence à la suite de l'entraînement !
        # Décommentez la ligne ci-dessous une fois qu'un modèle est enregistré
        # dans MLflow (via mlflow.register_model() au Défi 4.3).
        #
        inference_pipeline()
        # ---------------------------------------------------------------------

        return result

    # Exécuter
    print("\n📋 Instructions :")
    print("   1. Ajoutez @flow à data_preparation_flow")
    print("   2. Ajoutez @flow à training_flow")
    print("   3. Complétez les appels dans chaque sous-flow\n")

    try:
        result = churn_pipeline_v5()
        if result:
            print("\n✅ ÉTAPE 5 RÉUSSIE !")
            print("   Votre pipeline est maintenant modulaire et réutilisable !")
            print("   Passez à l'étape finale : python Prefect_Exercises.py deploy")


    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("   Vérifiez les décorateurs @flow sur les sous-flows.")

    # -------------------------------------------------------------------------
    # MINI-DÉFIS :
    # -------------------------------------------------------------------------
    #
    # DÉFI 5.1 : Créez un flow 'inference_pipeline' qui réutilise
    #            data_preparation_flow mais charge un modèle existant.
    #
    # DÉFI 5.2 : Dans l'interface Prefect, trouvez le graphe d'exécution.
    #            Voyez-vous les sous-flows imbriqués ? 
    #            (Il y a un bouton permettant de cacher/montrer les sous-flows disponible pour mieux les repérer)
    #
    # DÉFI 5.3 : Ajoutez un paramètre 'experiment_name' au flow principal
    #            et propagez-le jusqu'à training_flow.
    #            (Notez qu'il est déjà présent dans training_flow, mais il n'est pas encore utilisé dans le flow principal)
    # -------------------------------------------------------------------------


# =============================================================================
# ÉTAPE 6 : DÉPLOIEMENT - Automatisation avec Planification
# =============================================================================
#
# 🎯 C'EST LE CLOU DU SPECTACLE !
#
# OBJECTIF : Déployer le pipeline pour qu'il s'exécute automatiquement
#
# CONCEPTS :
#   - flow.serve() : Démarre un serveur qui écoute les planifications
#   - cron : Expression de planification (comme les crontabs Linux)
#      - " * * * * * " : Chaque étoile représente une unité de temps (minute, heure, jour du mois, mois, jour de la semaine)
#      - Exemples :
#         - "0 12 * * *" : Tous les jours à 12h
#         - "*/2 * * * *" : Toutes les 2 minutes
#         - "0 0 * * 0" : Tous les dimanches à minuit
#   - Prefect UI : Visualiser les exécutions automatiques
#
# SCÉNARIO RÉEL : Réentraîner le modèle tous les jours à 6h du matin
#
# =============================================================================

def run_deploy():
    """
    ÉTAPE 6 : Déployer le pipeline planifié

    Le pipeline va s'exécuter automatiquement toutes les 2 minutes !
    Observez dans l'interface Prefect.
    """
    from prefect import flow, task
    from prefect.tasks import task_input_hash

    print("=" * 70)
    print("ÉTAPE 6 : DÉPLOIEMENT - Le pipeline s'exécute tout seul !")
    print("=" * 70)

    # Pipeline complet avec tous les patterns
    @task(retries=3, retry_delay_seconds=[5, 10, 20])
    def load_data() -> pd.DataFrame:
        return load_churn_data(DATA_PATH)

    @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
    def prepare_features(df: pd.DataFrame) -> tuple:
        return preprocess_data(df)

    @task
    def train(X, y, n_estimators: int, max_depth: int) -> dict:
        return train_model(X, y, n_estimators=n_estimators, max_depth=max_depth)

    @flow(name="churn-production-pipeline", log_prints=True)
    def churn_production_pipeline(n_estimators: int = 100, max_depth: int = 10):
        """
        Pipeline de production - Prêt pour le déploiement !

        Ce pipeline sera exécuté automatiquement selon la planification.
        """
        print(f"🚀 Exécution planifiée à {datetime.now().strftime('%H:%M:%S')}")

        data = load_data()
        X, y = prepare_features(data)
        result = train(X, y, n_estimators, max_depth)

        print(f"✅ Terminé ! Accuracy : {result['metrics']['accuracy']:.4f}")
        return result

    print("\n📋 Instructions :")
    print("   1. Le pipeline va être déployé avec une planification cron")
    print("   2. Il s'exécutera toutes les 2 minutes automatiquement")
    print("   3. Ouvrez http://localhost:4200 pour observer les exécutions")
    print("   4. Appuyez sur Ctrl+C pour arrêter\n")

    print("=" * 70)
    print("🎯 DÉPLOIEMENT EN COURS...")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TODO : Déployez le flow avec une planification
    # INDICE : churn_production_pipeline.serve(
    #              name="churn-scheduled",
    #              cron="*/2 * * * *"  # Toutes les 2 minutes
    #          )
    # -------------------------------------------------------------------------
    
    churn_production_pipeline.serve(
        name="churn-scheduled",
        cron="*/2 * * * *",  # Toutes les 2 minutes
        tags=["production", "churn", "exercises"] # Tags 
    )

    # En attendant que vous décommentiez, exécution simple :
    print("\n⚠️  Pour activer le déploiement planifié, décommentez le bloc")
    print("   churn_production_pipeline.serve(...) dans le code.\n")
    print("   Exécution unique en attendant...\n")

    churn_production_pipeline()

    print("\n" + "=" * 70)
    print("✅ Pour voir la VRAIE automatisation :")
    print("   1. Décommentez le bloc .serve() dans le code")
    print("   2. Réexécutez : python Prefect_Exercises.py deploy")
    print("   3. Observez les runs apparaître dans http://localhost:4200")
    print("=" * 70)
    # Note : Le serveur Prefect continuera de tourner et d'exécuter le pipeline selon la planification.
    # Appuyez sur Ctrl+C pour l'arrêter quand vous avez fini de tester.
    # Votre terminal est utilisé pour le serveur Prefect, donc vous ne verrez pas les prints du pipeline tant que le serveur tourne. C'est normal !

    # -------------------------------------------------------------------------
    # MINI-DÉFIS :
    # -------------------------------------------------------------------------
    #
    # DÉFI 6.1 : Changez la planification pour "0 6 * * *" (tous les jours à 6h).
    #            C'est un pattern commun pour le réentraînement quotidien !
    #
    # DÉFI 6.2 : Ajoutez des tags pour organiser vos déploiements
    #            (ex: "production", "ml", "churn").
    #
    # DÉFI 6.3 : Créez un 2ème déploiement pour l'inférence avec une planification
    #            différente (ex: toutes les heures pour des prédictions batch).
    # -------------------------------------------------------------------------


# =============================================================================
# ÉTAPE 7 : NOTIFICATIONS - Alertes en cas d'échec
# =============================================================================
#
# OBJECTIF : Recevoir une notification Discord/Slack quand le pipeline échoue
#
# CONCEPTS :
#   - on_failure : Liste de handlers appelés en cas d'échec
#   - Webhooks : URLs pour envoyer des messages à Discord/Slack
#   - Monitoring proactif : Être alerté avant que les utilisateurs ne le soient
#
# =============================================================================

def run_notifications():
    """
    ÉTAPE 7 (BONUS) : Notifications Discord/Slack

    Configurez des alertes automatiques en cas d'échec du pipeline.
    """
    from prefect import flow, task
    import json

    print("=" * 70)
    print("ÉTAPE 7 : NOTIFICATIONS - Alertes Discord/Slack")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Configuration des webhooks (à remplacer par vos URLs)
    # -------------------------------------------------------------------------
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

    # -------------------------------------------------------------------------
    # Handler de notification Discord
    # -------------------------------------------------------------------------
    def notify_discord_on_failure(flow, flow_run, state):
        """Envoie une notification Discord en cas d'échec."""
        if not DISCORD_WEBHOOK_URL:
            print("⚠️  DISCORD_WEBHOOK_URL non configuré")
            return

        import urllib.request

        message = {
            "embeds": [{
                "title": "❌ Pipeline Échoué !",
                "description": f"**Flow:** {flow.name}\n**État:** {state.name}",
                "color": 15158332,  # Rouge
                "fields": [
                    {"name": "Run ID", "value": str(flow_run.id)[:8], "inline": True},
                    {"name": "Heure", "value": datetime.now().strftime("%H:%M:%S"), "inline": True}
                ]
            }]
        }

        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=json.dumps(message).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )

        try:
            urllib.request.urlopen(req)
            print("📨 Notification Discord envoyée !")
        except Exception as e:
            print(f"❌ Erreur Discord : {e}")

    # -------------------------------------------------------------------------
    # TODO : Ajoutez on_failure=[notify_discord_on_failure] au flow
    # -------------------------------------------------------------------------
    @task
    def failing_task():
        """Tâche qui échoue volontairement pour tester les notifications."""
        raise Exception("Échec simulé pour tester les notifications !")

    @flow(name="notification-test-flow", log_prints=True, on_failure=[notify_discord_on_failure])
    def notification_test_flow():
        """Flow de test pour les notifications."""
        print("🧪 Test des notifications...")
        failing_task()

    # Exécuter
    print("\n📋 Configuration des notifications :")
    print("")
    print("   DISCORD :")
    print("   1. Créez un serveur Discord ou utilisez un existant")
    print("   2. Paramètres du serveur > Intégrations > Webhooks > Nouveau")
    print("   3. Copiez l'URL du webhook")
    print("   4. Définissez : export DISCORD_WEBHOOK_URL='votre-url'")
    print("")
    print("   SLACK :")
    print("   1. Allez sur api.slack.com/apps et créez une app")
    print("   2. Activez 'Incoming Webhooks'")
    print("   3. Ajoutez un webhook pour votre channel")
    print("   4. Définissez : export SLACK_WEBHOOK_URL='votre-url'")
    print("")

    if DISCORD_WEBHOOK_URL or SLACK_WEBHOOK_URL:
        print("✅ Webhook configuré ! Lancement du test...\n")
        try:
            notification_test_flow()
        except Exception:
            print("\n✅ Le flow a échoué (comme prévu).")
            print("   Vérifiez Discord/Slack pour la notification !")
    else:
        print("⚠️  Aucun webhook configuré.")
        print("   Définissez DISCORD_WEBHOOK_URL ou SLACK_WEBHOOK_URL")
        print("   puis réexécutez pour tester.\n")

        print("Exemple avec variable d'environnement :")
        print("   export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'")
        print("   python Prefect_Exercises.py notif")

    # -------------------------------------------------------------------------
    # MINI-DÉFIS :
    # -------------------------------------------------------------------------
    #
    # DÉFI 7.1 : Ajoutez aussi on_completion pour être notifié des succès.
    #
    # DÉFI 7.2 : Personnalisez le message Discord avec plus de détails
    #            (paramètres utilisés, métriques obtenues...).
    #
    # DÉFI 7.3 : Créez un handler qui envoie un email (avec smtplib).
    # -------------------------------------------------------------------------


# =============================================================================
# ÉTAPE 8 : DÉPLOIEMENT DOCKER - Exécution dans le Worker
# =============================================================================
#
# OBJECTIF : Déployer le pipeline pour qu'il s'exécute dans le worker Docker
#
# CONCEPTS :
#   - flow.serve() : Exécute le flow LOCALEMENT (dans votre terminal)
#   - flow.deploy() : Enregistre le flow pour exécution dans un WORKER DOCKER
#
# DIFFÉRENCE CLÉ :
#   - serve() bloque votre terminal (Ctrl+C pour arrêter)
#   - deploy() libère votre terminal (le worker Docker exécute le flow)
#
# ARCHITECTURE PRODUCTION :
#   Votre code → Prefect Server → Work Pool → Worker Docker → Exécution
#
# =============================================================================

def run_worker_demo():
    """
    ÉTAPE 8 : Déployer vers le worker Docker

    Cette étape démontre la différence entre serve() et deploy().
    Le flow sera exécuté par le worker Docker, pas localement !
    """
    from prefect import flow, task
    from prefect.tasks import task_input_hash

    print("=" * 70)
    print("ÉTAPE 8 : DÉPLOIEMENT DOCKER - Le worker exécute le flow")
    print("=" * 70)

    print("""
📋 DIFFÉRENCE serve() vs deploy() :

  serve()  → Le flow s'exécute ICI (votre terminal)
             Terminal BLOQUÉ jusqu'à Ctrl+C
             Simple, bon pour apprendre

  deploy() → Le flow s'exécute dans le WORKER DOCKER
             Terminal LIBÉRÉ immédiatement
             Architecture de production

""")

    print("=" * 70)
    print("🎯 DÉPLOIEMENT VERS LE WORKER DOCKER...")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # SOLUTION : Utiliser from_source() pour déployer vers un work pool
    #
    # Pour un work pool de type "process", on doit indiquer où se trouve
    # le code que le worker doit exécuter. On utilise le flow
    # production_pipeline du fichier Prefect_Workshop.py qui est défini
    # au niveau module.
    # -------------------------------------------------------------------------
    from prefect import flow

    # Charger le flow depuis le Workshop via from_source()
    deployed_flow = flow.from_source(
        source=str(PROJECT_ROOT),
        entrypoint="pipelines/workshop/02_prefect/Prefect_Workshop.py:production_pipeline"
    )

    deployment_id = deployed_flow.deploy(
        name="worker-training-exercises",
        work_pool_name="default-pool",
        cron="*/5 * * * *",  # Toutes les 5 minutes
        tags=["workshop", "ml", "worker-demo", "exercises"],
        description="Pipeline exécuté par le worker Docker - toutes les 5 minutes",
        parameters={
            "n_estimators": 100,
            "max_depth": 10,
            "experiment_name": "workshop-exercises-worker",
            "model_name": "churn-predictor-exercises"
        }
    )

    print()
    print("=" * 70)
    print("✅ DÉPLOIEMENT RÉUSSI !")
    print("=" * 70)
    print(f"""
Le flow est maintenant déployé vers le worker Docker.

PROCHAINES ÉTAPES :

1. Ouvrir l'interface Prefect : http://localhost:4200
   → Aller dans Deployments
   → Trouver "churn-prediction-pipeline/worker-training-exercises"

2. Déclencher manuellement :
   → Cliquer sur "Quick Run"
   → Observer l'exécution dans l'onglet "Runs"

3. Voir les logs du worker :
   docker-compose logs -f prefect-worker

4. Vérifier MLflow : http://localhost:5000

5. Pour supprimer le déploiement :
   → Interface Prefect > Deployments > Delete

COMPARAISON :
  - 'deploy' (serve)  : Exécution locale, terminal bloqué
  - 'worker-demo'     : Exécution Docker, terminal libre ← VOUS ÊTES ICI
""")

    return deployment_id


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎓 PREFECT EXERCISES - Approche Fil Rouge")
    print("   Transformez un script ML en pipeline de production !")
    print("=" * 70)

    etape = sys.argv[1] if len(sys.argv) > 1 else "help"

    if etape == "etape1":
        run_etape1()

    elif etape == "etape2":
        run_etape2()

    elif etape == "etape3":
        run_etape3()

    elif etape == "etape4":
        run_etape4()

    elif etape == "etape5":
        run_etape5()

    elif etape == "deploy":
        run_deploy()

    elif etape == "notif":
        run_notifications()

    elif etape == "worker-demo":
        run_worker_demo()

    else:
        print("""
🎯 OBJECTIF : Transformer un script ML classique en pipeline de production

Le code de départ (fonctions load_churn_data, preprocess_data, train_model)
est un pipeline ML typique. Votre mission : le "blinder" progressivement
avec tous les patterns d'orchestration professionnels !

PROGRESSION :
  etape1      Tasks & Flows      → Transformer les fonctions en tâches
  etape2      Résilience         → Ajouter des réessais sur le chargement
  etape3      Efficacité         → Cacher le preprocessing coûteux
  etape4      MLflow + Pipeline  → Tracker avec sklearn Pipeline ⭐
  etape5      Sous-flows         → Organiser en modules réutilisables
  deploy      Automatisation     → Planifier l'exécution automatique (serve)
  worker-demo Déploiement Docker → Exécuter dans le worker Docker (deploy)
  notif       Notifications      → Alertes Discord/Slack (bonus)

PRÉREQUIS :
  1. docker-compose ps                  # Vérifier que Docker tourne
  2. Copier .env.example vers .env      # Config (une seule fois)

UTILISATION :
  python Prefect_Exercises.py etape1      # Commencez ici !
  python Prefect_Exercises.py etape2      # Puis continuez...
  python Prefect_Exercises.py etape3
  python Prefect_Exercises.py etape4      # MLflow + sklearn Pipeline
  python Prefect_Exercises.py etape5
  python Prefect_Exercises.py deploy      # Automatisation avec serve()
  python Prefect_Exercises.py worker-demo # Exécution dans le worker Docker
  python Prefect_Exercises.py notif       # Bonus : alertes

RÉFÉRENCE :
  Consultez Prefect_Workshop.py pour voir les solutions complètes.

INTERFACES :
  Prefect : http://localhost:4200
  MLflow  : http://localhost:5000
""")
