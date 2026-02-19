# Airflow: The Industry Standard

## Why Look at Airflow?

**Airflow** is the most widely deployed orchestrator in industry:
- Created by Airbnb in 2014, now an Apache project
- Used by: Uber, Slack, Twitter, Adobe, Lyft, and thousands more
- Battle-tested at massive scale

Understanding Airflow helps you:
1. Work with existing enterprise systems
2. Appreciate why alternatives like Prefect and Dagster were created
3. Make informed decisions about which tool to use

---

## Airflow Concepts

```
Airflow Vocabulary
─────────────────────────────────────────────
DAG         Directed Acyclic Graph - your pipeline definition
Task        One step in the pipeline
Operator    Type of task (PythonOperator, BashOperator, etc.)
XCom        Cross-communication between tasks (key-value store)
```

---

## The Pain Points (Why We Focus on Prefect)

### Point Douloureux 1 : Limitations de XCom

XCom est la façon dont les tâches partagent des données, mais il a une **limite de 48KB** par défaut.

**Problème** : Vous ne pouvez pas passer des DataFrames via XCom.

```python
# Ce que vous VOULEZ faire (mais ne pouvez pas) :
def load_data(**context):
    df = pd.read_csv('data.csv')
    return df  # Trop gros pour XCom !

# Ce que vous DEVEZ faire :
def load_data(**context):
    df = pd.read_csv('data.csv')

    # Sauvegarder sur disque
    temp_path = f"/tmp/data_{context['run_id']}.parquet"
    df.to_parquet(temp_path)

    # Pousser le CHEMIN (pas les données)
    context['ti'].xcom_push(key='data_path', value=temp_path)
```

### Point Douloureux 2 : I/O Fichier Partout

Chaque tâche qui a besoin de données doit :
1. Tirer le chemin du fichier depuis XCom
2. Charger les données depuis le disque
3. Les traiter
4. Sauvegarder dans un nouveau fichier
5. Pousser le nouveau chemin vers XCom

```python
def process_data(**context):
    # Tirer le chemin de la tâche précédente
    path = context['ti'].xcom_pull(key='data_path', task_ids='load_data')

    # Charger depuis le disque
    df = pd.read_parquet(path)

    # Traiter
    df = df.dropna()

    # Sauvegarder dans un nouveau fichier
    new_path = f"/tmp/processed_{context['run_id']}.parquet"
    df.to_parquet(new_path)

    # Pousser le nouveau chemin
    context['ti'].xcom_push(key='data_path', value=new_path)
```

**Comparez avec Prefect :**
```python
@task
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()  # C'est tout !
```

### Point Douloureux 3 : Nettoyage Manuel

Tous ces fichiers temporaires ? Vous devez les nettoyer :

```python
def cleanup_temp_files(**context):
    """Supprimer tous les fichiers temporaires créés pendant l'exécution du DAG."""
    paths = []
    paths.append(context['ti'].xcom_pull(key='data_path', task_ids='load_data'))
    paths.append(context['ti'].xcom_pull(key='data_path', task_ids='process'))
    paths.append(context['ti'].xcom_pull(key='model_path', task_ids='train'))
    # ... chaque tâche qui a créé un fichier

    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)

# Doit s'exécuter après tout, même en cas d'échec
cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_temp_files,
    trigger_rule='all_done',
    dag=dag
)
```

### Point Douloureux 4 : Configuration Verbeuse

Avant toute logique métier, vous avez besoin de ~30 lignes de code passe-partout :

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['team@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Train customer churn model',
    schedule_interval='0 2 * * *',
    catchup=False,
    tags=['ml', 'training'],
)

# MAINTENANT vous pouvez définir les tâches...
load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)
```

### Point Douloureux 5 : Dépendances Déclarées Séparément

Les dépendances entre tâches sont définies séparément des fonctions :

```python
# Définir les tâches
load = PythonOperator(task_id='load', ...)
process = PythonOperator(task_id='process', ...)
train = PythonOperator(task_id='train', ...)
evaluate = PythonOperator(task_id='evaluate', ...)

# Dépendances déclarées séparément (facile de se tromper)
load >> process >> train >> evaluate
```

**Comparez avec Prefect :**
```python
@flow
def pipeline():
    data = load()
    processed = process(data)  # La dépendance est évidente !
    model = train(processed)
    evaluate(model)
```

---

## Implémentation de Référence

Voir l'implémentation Airflow complète :

```
pipelines/examples/Airflow_ML_Pipeline.py
```

Ce fichier implémente le même pipeline ML que Prefect et Dagster, montrant :
- Les patterns de push/pull XCom
- L'I/O fichier pour les DataFrames
- L'implémentation de la tâche de nettoyage
- L'intégration MLflow avec Airflow
- L'utilisation des kwargs `**context`

### Sections Clés à Revoir

| Plage de Lignes | Ce Qu'Elle Montre |
|------------------|-------------------|
| 44-48 | Configuration et chemins |
| 60-100 | `load_customer_data` avec push XCom |
| 100-140 | `engineer_features` avec pull/push XCom |
| 200-250 | `train_model` avec sérialisation pickle |
| 350-380 | Tâche `cleanup_temp_files` |
| 400-450 | Définition du DAG et câblage des tâches |

---

## Quand Airflow a du Sens

Malgré la complexité, Airflow est le bon choix quand :

| Scénario | Pourquoi Airflow |
|----------|------------------|
| Infrastructure existante | Votre entreprise l'utilise déjà |
| Orchestration inter-systèmes | Besoin de coordonner Spark, bases de données, services cloud |
| Exigences d'entreprise | Journaux d'audit, RBAC, conformité |
| Équipe plateforme dédiée | Peut gérer la complexité opérationnelle |
| Workflows non-Python | Besoin de BashOperator, KubernetesOperator, etc. |

---

## Résumé : Airflow vs Prefect

| Aspect | Airflow | Prefect |
|--------|---------|---------|  
| Passage de données | XCom + fichiers temp | Valeurs de retour |
| Configuration | 30+ lignes passe-partout | 2 décorateurs |
| Nettoyage | Tâche manuelle requise | Automatique |
| Dépendances | `task1 >> task2` | Appels de fonction |
| Annotations de type | Non supportées | Python natif |
| Courbe d'apprentissage | Raide | Douce |

---

## Prochaines Étapes

1. **Revoir l'exemple** : Ouvrir `pipelines/examples/Airflow_ML_Pipeline.py`
2. **Comparer avec Prefect** : Regarder les mêmes opérations dans `pipelines/examples/Prefect_ML_Pipeline.py`
3. **Comprendre les compromis** : La complexité d'Airflow vous apporte des fonctionnalités d'entreprise et un écosystème
