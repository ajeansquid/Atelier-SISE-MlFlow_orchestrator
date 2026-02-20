# Orchestration des Workflows ML : Guide Pratique

## Table des Matières

1. [Pourquoi des Orchestrateurs ?](#1-pourquoi-des-orchestrateurs-)
2. [Les Trois Orchestrateurs](#2-les-trois-orchestrateurs)
3. [Patterns d'Orchestration pour le ML](#3-patterns-dorchestration-pour-le-ml)
4. [Où MLflow S'Intègre](#4-où-mlflow-sintègre)
5. [Structure de l'Atelier](#5-structure-de-latelier)
6. [Exécuter les Ateliers](#6-exécuter-les-ateliers)
7. [Référence Rapide](#7-référence-rapide)

---

## 1. Pourquoi des Orchestrateurs ?

### Le Problème avec les Notebooks

Vous avez construit un excellent modèle ML dans un notebook. Et maintenant ?

```
Votre workflow actuel :
────────────────────────────────────────────────────────────────
Lundi :      Exécuter le notebook manuellement, modèle entraîné
Mardi :      Oublié de l'exécuter
Mercredi :   "Pourquoi le modèle est obsolète ?"
Jeudi :      L'exécuter, l'API échoue à la cellule 5, tout relancer
Vendredi :   Manager : "Pourquoi ce n'est pas automatisé ?"
```

### Ce que les Orchestrateurs Résolvent

| Problème ML | Solution d'Orchestration |
|------------|------------------------|
| L'API échoue aléatoirement | **Réessais** avec backoff |
| Le feature engineering est lent | **Cache** - sauter si inchangé |
| Comparer 5 modèles prend 5x plus de temps | **Exécution parallèle** |
| Différents hyperparamètres = changements de code | **Paramètres** - configurer à l'exécution |
| Entraînement et inférence partagent du code | **Sous-flows** - composer les pipelines |
| Pas de visibilité sur l'état du pipeline | **Monitoring** - logs, UI, alertes |

### Le Rôle de l'Orchestrateur

```
                    ORCHESTRATEUR
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐  │
    │   │Charger│ → │ Feat │ →  │Entraîner│→│ Enreg│  │
    │   │Données│   │ Eng  │    │ Modèle │ │Modèle│  │
    │   └──────┘    └──────┘    └──────┘    └──────┘  │
    │      │           │           │           │      │
    │   Réessai?    Cache?      Track?      Version?  │
    │   3 fois      1 heure     MLflow      Registre  │
    │                                                  │
    └──────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         Prefect:                MLflow:
         - Réessais              - Expérimentations
         - Cache                 - Métriques
         - Planification         - Versions de modèles
         - Monitoring            - Artefacts
```

**Point clé** : Les orchestrateurs gèrent COMMENT votre pipeline s'exécute. MLflow gère CE QUI est suivi.

---

## 2. Les Trois Orchestrateurs

### Vue d'Ensemble

| Outil | Philosophie | Idéal Pour |
|------|------------|----------|
| **Airflow** | "Tout configurer explicitement" | Enterprise, infra existante |
| **Prefect** | "Juste écrire du Python" | Équipes Python, démarrage rapide |
| **Dagster** | "Penser aux données, pas aux tâches" | Plateformes data, lignage |

### Airflow : Standard de l'Industrie

- Créé par Airbnb (2014)
- Testé au combat à grande échelle
- Écosystème riche (Spark, bases de données, cloud)

**Points de friction pour le ML :**
- XCom ne peut pas gérer les DataFrames (limite de 48KB)
- Doit sauvegarder/charger des fichiers entre les tâches
- Nettoyage manuel des fichiers temporaires
- Configuration lourde avant toute logique

### Prefect : Approche Pythonique

- Créé en 2018 comme alternative native Python
- Décorateurs sur des fonctions normales
- Valeurs de retour = flux de données
- Infrastructure minimale

**Avantage clé** : Ressemble à l'écriture de Python normal.

### Dagster : Centré sur les Assets

- Créé en 2018, focus sur les "assets de données"
- Dépendances inférées depuis les paramètres de fonction
- Graphe de lignage de données intégré
- Ré-exécuter seulement ce qui a changé

**Avantage clé** : Penser "quelles données existent" et non "quelles tâches s'exécutent".

---

## 3. Patterns d'Orchestration pour le ML

Ces patterns résolvent de vrais problèmes ML. Chacun est couvert dans l'atelier Prefect.

### Pattern 1 : Résilience (Réessais)

**Problème ML** : Les APIs de données échouent aléatoirement - limites de débit, timeouts, problèmes réseau.

```python
# Votre job nocturne échoue à 3h du matin. Vous le découvrez à 9h.

@task(retries=3, retry_delay_seconds=60)
def load_from_api() -> pd.DataFrame:
    """Réessayer automatiquement en cas d'échec."""
    return requests.get(API_URL).json()

# Backoff exponentiel pour les limites de débit
@task(retries=3, retry_delay_seconds=[10, 30, 60])
def load_with_backoff() -> pd.DataFrame:
    """Attendre plus longtemps entre chaque réessai : 10s, 30s, 60s."""
    ...
```

### Pattern 2 : Efficacité (Cache)

**Problème ML** : Le feature engineering prend 30 minutes. Le pipeline échoue à l'entraînement. Maintenant vous ré-exécutez le feature engineering à nouveau.

```python
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sauter si la même entrée a été vue dans la dernière heure."""
    # Calcul coûteux ici...
    return df
```

### Pattern 3 : Efficacité (Parallélisme)

**Problème ML** : Comparer 5 modèles mais les exécuter séquentiellement = 5x plus lent.

```python
@flow
def compare_models():
    # Préparation des données (séquentiel)
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # Entraînement (parallèle - pas de dépendances entre ceux-ci !)
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)

    # Sélection (attend tous les résultats ci-dessus)
    best = select_best([rf_result, gb_result, xgb_result])
    return best
```

### Pattern 4 : Flexibilité (Paramètres)

**Problème ML** : Différents hyperparamètres = changer le code à chaque fois.

```python
@flow
def training_flow(
    n_estimators: int = 100,
    max_depth: int = 10,
    experiment_name: str = "default"
):
    """Configurer sans changer le code."""
    ...

# Utilisation :
training_flow()  # Utiliser les valeurs par défaut
training_flow(n_estimators=200)  # Remplacer une valeur
training_flow(n_estimators=50, max_depth=5)  # Remplacer plusieurs valeurs
```

### Pattern 5 : Flexibilité (Sous-flows)

**Problème ML** : L'entraînement et l'inférence sont séparés mais partagent la logique de préparation des données.

```python
@flow
def data_preparation() -> pd.DataFrame:
    """Préparation de données réutilisable."""
    df = load_data()
    df = engineer_features(df)
    return df

@flow
def training_pipeline():
    df = data_preparation()  # Réutiliser !
    model = train(df)
    register(model)

@flow
def inference_pipeline():
    df = data_preparation()  # Réutiliser !
    model = load_from_registry()
    predictions = predict(model, df)
    save(predictions)
```

---

## 4. Où MLflow S'Intègre

### Répartition des Responsabilités

| Préoccupation | Orchestrateur (Prefect) | Tracker (MLflow) |
|---------|------------------------|------------------|
| Réessayer en cas d'échec | ✅ | |
| Cacher les calculs | ✅ | |
| Exécuter les tâches en parallèle | ✅ | |
| Planifier les pipelines | ✅ | |
| Logger les paramètres | | ✅ |
| Logger les métriques | | ✅ |
| Stocker les artefacts de modèles | | ✅ |
| Versionner les modèles | | ✅ |
| Servir les modèles | | ✅ |

### Point d'Intégration : La Tâche d'Entraînement

```python
@task(retries=2, cache_expiration=timedelta(hours=1))
def train_model(df: pd.DataFrame, n_estimators: int) -> dict:
    """
    Prefect gère :
    - Réessayer si l'entraînement échoue
    - Cacher si mêmes données/paramètres

    MLflow gère :
    - Suivre l'expérimentation
    - Stocker le modèle
    - Versionner
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        return {"model": model, "run_id": mlflow.active_run().info.run_id}
```

### Bonnes Pratiques

| Pratique | Pourquoi |
|----------|-----|
| Séparer les flows train/inference | Planifications différentes, modes d'échec différents |
| Charger les modèles depuis le registre | `models:/name/latest` et non `runs:/id/model` |
| Logger `orchestrator` comme param | Savoir quel outil a exécuté cette expérimentation |
| Enregistrer après validation | N'enregistrer que les modèles qui passent le seuil de qualité |

---

## 5. Structure de l'Atelier

### Atelier Prefect (Focus Principal)

Situé dans `02_prefect/`

- `Prefect_Workshop.py` : Référence complète (7 parties)
- `Prefect_Exercises.py` : **Exercices interactifs** (approche Fil Rouge)
  - Un vrai pipeline Churn que vous "blindez" progressivement
  - Étapes : etape1 → etape2 → ... → deploy → notif

| Partie | Pattern d'Orchestration | Problème ML Résolu |
|------|----------------------|-------------------|
| Partie 1 | Tasks & Flows | Chaos des notebooks → pipeline structuré |
| Partie 2 | Réessais, Backoff | Échecs d'API, limites de débit |
| Partie 3 | Cache, Parallélisme | Features coûteuses, comparaison lente de modèles |
| Partie 4 | Paramètres, Sous-flows | Tuning d'hyperparamètres, composants réutilisables |
| Partie 5 | Pipeline Complet | Tous les patterns + intégration MLflow |

### Vue d'Ensemble Airflow (Lecture Guidée)

Situé dans `01_airflow/`

- Pourquoi Airflow existe (standard de l'industrie)
- Points de friction pour le ML (limites XCom, I/O fichiers)
- `airflow_overview.md` : Explication des concepts
- `Airflow_Pipeline.py` : Pipeline complet de référence

### Atelier Dagster (Bonus)

Situé dans `03_dagster/`

- `Dagster_Workshop.py` : Apprendre la pensée centrée sur les assets
- `Dagster_Pipeline.py` : Pipeline complet de référence
- Utiliser l'interface Dagster pour la visualisation

---

## 6. Exécuter les Ateliers

### Prérequis

```bash
# 1. Démarrer le serveur MLflow
docker-compose up -d

# 2. Installer les dépendances
pip install prefect dagster dagster-webserver mlflow scikit-learn pandas

# 3. Générer les données d'exemple
python generate_sample_data.py
```

### Atelier Prefect

```bash
# Partie 1 : Tasks & Flows
python pipelines/workshop/02_prefect/Prefect_Workshop.py part1

# Partie 2 : Résilience (Réessais)
python pipelines/workshop/02_prefect/Prefect_Workshop.py part2

# Partie 3 : Efficacité (Cache, Parallèle)
python pipelines/workshop/02_prefect/Prefect_Workshop.py part3

# Partie 4 : Flexibilité (Paramètres, Sous-flows)
python pipelines/workshop/02_prefect/Prefect_Workshop.py part4

# Partie 5 : Pipeline Complet avec MLflow
python pipelines/workshop/02_prefect/Prefect_Workshop.py part5

# Partie 6 : Déploiement avec planification
python pipelines/workshop/02_prefect/Prefect_Workshop.py deploy

# Partie 7 : Notifications (Discord/Slack)
python pipelines/workshop/02_prefect/Prefect_Workshop.py part7
```

### Atelier Dagster (Bonus)

```bash
# Ligne de commande
python pipelines/workshop/03_dagster/Dagster_Workshop.py full

# Pipeline complet de référence
python pipelines/workshop/03_dagster/Dagster_Pipeline.py

# Avec interface (recommandé)
dagster dev -f pipelines/workshop/03_dagster/Dagster_Pipeline.py
# Ouvrir http://localhost:3000
```

### Pipeline Airflow (Référence)

```bash
# Lire le code pour comprendre les points de friction
# pipelines/workshop/01_airflow/Airflow_Pipeline.py

# Exécution standalone (simulation sans Airflow)
python pipelines/workshop/01_airflow/Airflow_Pipeline.py
python pipelines/workshop/01_airflow/Airflow_Pipeline.py train
python pipelines/workshop/01_airflow/Airflow_Pipeline.py inference
```

### Visualiser les Résultats

- **Interface MLflow** : http://localhost:5000
- **Interface Dagster** : http://localhost:3000
- **Interface Prefect** : `prefect server start` → http://localhost:4200

---

## 7. Référence Rapide

### Patterns Prefect

```python
# Tâche basique
@task
def my_task(df: pd.DataFrame) -> pd.DataFrame:
    return process(df)

# Avec réessais
@task(retries=3, retry_delay_seconds=60)
def resilient_task():
    ...

# Avec cache
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def cached_task(df: pd.DataFrame):
    ...

# Flow avec paramètres
@flow(log_prints=True)
def my_flow(n_estimators: int = 100):
    ...

# Composition de sous-flows
@flow
def parent_flow():
    data = data_prep_subflow()
    result = training_subflow(data)
```

### Patterns Dagster

```python
# Définition d'asset
@asset(group_name="features", description="Features ingéniérées")
def customer_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Dépendance inférée depuis le nom du paramètre !
    return process(raw_data)

# Matérialiser les assets
materialize([raw_data, customer_features, model])
```

### MLflow dans les Orchestrateurs

```python
# Dans une tâche Prefect
@task
def train_with_mlflow(df, params):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("my-experiment")

    with mlflow.start_run():
        mlflow.log_params(params)
        model = train(df)
        mlflow.log_metrics(evaluate(model))
        mlflow.sklearn.log_model(model, "model")

    return model
```

### Tableau de Comparaison

| Aspect | Prefect | Airflow | Dagster |
|--------|---------|---------|---------|
| Définir l'unité | `@task` | `PythonOperator` | `@asset` |
| Définir le pipeline | `@flow` | `DAG(...)` | `Definitions` |
| Passage de données | Valeurs de retour | XCom + fichiers | Paramètres de fonction |
| Réessais | `@task(retries=3)` | `default_args` | Politiques |
| Cache | `cache_key_fn=...` | Manuel | Intégré |
| Dépendances | Implicite (appels) | Explicite (`>>`) | Inférées |
| Philosophie | Centré sur les tâches | Centré sur les tâches | Centré sur les assets |

---

## Prochaines Étapes

1. **Commencer par Airflow** : Lire `01_airflow/airflow_overview.md` pour comprendre le standard
2. **Focus Prefect** : `python pipelines/workshop/02_prefect/Prefect_Workshop.py part1`
3. **Progresser à travers toutes les parties** : part1 → part2 → part3 → part4 → part5 → deploy → part7
4. **Essayer le bonus Dagster** : Transformer vos connaissances vers la pensée centrée sur les assets
5. **Explorer les pipelines de référence** : Voir les patterns prêts pour la production
