# Atelier MLflow + Orchestrateurs

## Contexte

**Durée :** 3 heures
**Public :** Étudiants Master 2 SISE (profils mixtes : data scientists, ML engineers, data engineers)
**Objectif :** Apprendre les pratiques MLOps - orchestrer des pipelines ML avec planification et automatisation, suivre les expérimentations avec MLflow

---

## Progression de l'Atelier

Le parcours pédagogique couvre le **pipeline ML complet** :

```
DÉVELOPPER → SUIVRE → ENREGISTRER → SERVIR → AUTOMATISER
```

### Phase 1 : Fondamentaux MLflow (Notebooks)

| Notebook | Objectif |
|----------|---------|
| `01_messy_notebook.ipynb` | Point de départ - chaos typique de data science |
| `01b_mlflow_transition.ipynb` | Transition guidée interactive vers MLflow |
| `02_mlflow_organized.ipynb` | Solution de référence MLflow complète |

**Structure de la Transition MLflow :**
- Partie 1 : Guidage complet (bases du tracking)
- Partie 2 : Découverte progressive (modèles, artefacts, recherche)
- Partie 3 : Exercices à compléter
- Partie 4 : Serving & inférence (registre, chargement, service, inférence batch)

### Phase 2 : Orchestration avec Automatisation (Focus Principal)

**Atelier Prefect** (`pipelines/workshop/02_prefect/`) :
- `Prefect_Workshop.py` : Référence complète (7 parties)
- `Prefect_Exercises.py` : **Exercices interactifs** (approche Fil Rouge - un vrai pipeline Churn à blinder progressivement !)
- Partie 1 : Tasks & Flows (bases)
- Partie 2 : Résilience (réessais, backoff exponentiel)
- Partie 3 : Efficacité (cache, entraînement de modèles en parallèle)
- Partie 4 : Flexibilité (paramètres, sous-flows)
- Partie 5 : Pipeline complet avec intégration MLflow
- **Partie 6 : AUTOMATISATION - Déployer, planifier, regarder l'exécution !**
- Partie 7 : Notifications (Discord/Slack)

### Phase 3 : Bonus Dagster

**Atelier Dagster** (`pipelines/workshop/03_dagster/`) :
- `Dagster_Exercises.py` : **Exercices interactifs** (paradigme Asset vs Task - 80% UI, 20% code)
- `Dagster_Workshop.py` : Référence complète
- Comprendre l'auto-câblage des dépendances
- Matérialisation partielle (la killer-feature !)
- Métadonnées et intégration MLflow

---

## Architecture : Orchestration Basée sur Docker

```
+-------------------------------------------------------------------------+
|                          DOCKER COMPOSE                                  |
|                                                                          |
|  +------------+   +----------------------+   +-------------------+      |
|  |            |   |      PREFECT         |   |                   |      |
|  |   MLFLOW   |   |  +------+ +-------+  |   |    DAGSTER        |      |
|  |   SERVER   |<--|  |Server| |Worker |  |   |  +----------+     |      |
|  |            |   |  |:4200 | |       |  |   |  |Webserver |     |      |
|  |  :5000     |   |  +------+ +---+---+  |   |  |  :3000   |     |      |
|  |            |   |              |       |   |  +----------+     |      |
|  | Experiments|   |    Executes--+       |   |  +----------+     |      |
|  | Models     |   |    scheduled         |   |  | Daemon   |     |      |
|  | Artifacts  |   |    flows             |   |  |(schedules)|    |      |
|  |            |   |                      |   |  +----------+     |      |
|  +------------+   +----------------------+   +-------------------+      |
|        ^                    |                         |                  |
|        +--------------------+-------------------------+                  |
|                    Logs experiments to MLflow                            |
+-------------------------------------------------------------------------+

VOTRE MACHINE :
  - VS Code avec notebooks (apprentissage)
  - Déployer les flows via CLI
  - Observer l'automatisation dans les interfaces
```

**Point Clé :** Il s'agit d'une VRAIE orchestration. Les flows s'exécutent automatiquement selon des plannings. Les workers les exécutent. Vous observez dans les interfaces.

---

## Prérequis

| Outil | Version | Objectif |
|------|---------|---------|
| **Python** | >= 3.10 | Tous les pipelines et notebooks |
| **Docker** & **Docker Compose** | Récente | Serveur MLflow, serveur/worker Prefect, Dagster |
| **VS Code** | Récente | Notebooks, édition de code |
| **Git** | Toute version | Cloner le dépôt |

---

## Démarrage Rapide

### 1. Démarrer la Stack

```bash
# Démarrer MLflow + Prefect (serveur + worker)
docker-compose up -d

# Vérifier les services
docker-compose ps
# Attendre le statut "healthy"

# Optionnel : Ajouter Dagster
docker-compose --profile dagster up -d
```

### 2. Installer les Dépendances Locales

```bash
pip install -r requirements.txt
```

### 3. Générer les Données d'Exemple

```bash
python generate_sample_data.py
```

Cela crée `data/customer_data.csv` utilisé par tous les pipelines.

### 4. Accéder aux Interfaces

| Service | URL | Objectif |
|---------|-----|---------|
| **Prefect** | http://localhost:4200 | Déploiements de flows, exécutions, planifications |
| **MLflow** | http://localhost:5000 | Expérimentations, modèles, artefacts |
| **Dagster** | http://localhost:3000 | Graphe d'assets, matérialisations (bonus) |

### 5. Exécuter l'Atelier

**Apprendre les patterns (Parties 1-7) :**
```bash
python pipelines/workshop/02_prefect/Prefect_Workshop.py part1  # Tasks & Flows
python pipelines/workshop/02_prefect/Prefect_Workshop.py part2  # Réessais
python pipelines/workshop/02_prefect/Prefect_Workshop.py part3  # Cache & Parallélisme
python pipelines/workshop/02_prefect/Prefect_Workshop.py part4  # Paramètres
python pipelines/workshop/02_prefect/Prefect_Workshop.py part5  # Pipeline Complet + MLflow
python pipelines/workshop/02_prefect/Prefect_Workshop.py part7  # Notifications Discord/Slack
```

**Voir l'automatisation réelle (Partie 6) :**
```bash
python pipelines/workshop/02_prefect/Prefect_Workshop.py deploy
```

Cela déploie un flow planifié qui s'exécute toutes les 2 minutes. Observez :
- **Interface Prefect** : Voir les déploiements et les exécutions automatiques
- **Interface MLflow** : Voir les expérimentations apparaître automatiquement

**Bonus Dagster :**
```bash
docker-compose --profile dagster up -d
# Ouvrir http://localhost:3000 et suivre les TODOs dans Dagster_Exercises.py
# Le graphe est "cassé" volontairement - corrigez-le en modifiant le fichier !

# Référence (si besoin de voir la solution) :
python pipelines/workshop/03_dagster/Dagster_Workshop.py full
```

**Vue d'ensemble Airflow (lecture guidée) :**

Lire `pipelines/workshop/01_airflow/airflow_overview.md` pour comprendre les points de friction d'Airflow pour les workflows ML. Ensuite comparer avec l'implémentation de référence dans `pipelines/workshop/01_airflow/Airflow_Pipeline.py`.

---

## Structure du Projet

```
├── notebooks/                        # Notebooks Jupyter
│   ├── 01_messy_notebook.ipynb       # Point de départ - chaos
│   ├── 01b_mlflow_transition.ipynb   # Transition MLflow guidée
│   └── 02_mlflow_organized.ipynb     # Solution MLflow complète
│
├── pipelines/workshop/               # Matériel d'apprentissage
│   ├── orchestrator_guide.md         # Guide des concepts
│   │
│   ├── 01_airflow/                   # LECTURE GUIDÉE (voir le standard)
│   │   ├── airflow_overview.md       # Points de friction expliqués
│   │   └── Airflow_Pipeline.py       # Pipeline complet de référence
│   │
│   ├── 02_prefect/                   # FOCUS PRINCIPAL
│   │   ├── Prefect_Workshop.py       # Référence complète (7 parties)
│   │   └── Prefect_Exercises.py      # Exercices interactifs (approche Fil Rouge)
│   │
│   └── 03_dagster/                   # BONUS
│       ├── Dagster_Exercises.py      # Exercices interactifs (chargé par Docker)
│       └── Dagster_Workshop.py       # Référence complète (solution)
│
├── docs/
│   ├── mlflow_cheatsheet.md          # Référence rapide
│   ├── README_DOCKER.md              # Instructions de configuration Docker
│   ├── mlflow/                       # PDFs de cours
│   └── airflow/                      # PDFs de cours
│
├── data/                             # Données générées
│   └── customer_data.csv
│
├── docker-compose.yml                # MLflow + Prefect + Dagster
├── Dockerfile.prefect                # Image worker Prefect
├── Dockerfile.dagster                # Image de base Dagster (webserver + daemon)
├── requirements.txt
└── generate_sample_data.py
```

---

## Services Docker

| Service | Port | Objectif | Par défaut |
|---------|------|---------|---------|
| **mlflow** | 5000 | Tracking d'expérimentations, registre de modèles | Oui |
| **prefect-server** | 4200 | Monitoring de flows, déploiements, planifications | Oui |
| **prefect-worker** | - | Exécute les flows planifiés | Oui |
| **dagster-webserver** | 3000 | Interface du graphe d'assets (bonus) | `--profile dagster` |
| **dagster-daemon** | - | Exécute les jobs planifiés (bonus) | `--profile dagster` |

### Commandes

```bash
# Atelier principal (MLflow + Prefect)
docker-compose up -d

# Avec bonus Dagster
docker-compose --profile dagster up -d

# Tout arrêter
docker-compose down

# Table rase (supprimer les données)
docker-compose down -v

# Voir les logs
docker-compose logs -f prefect-worker
docker-compose logs -f mlflow
docker-compose logs -f dagster-daemon
```

Voir `docs/README_DOCKER.md` pour les instructions Docker détaillées.

---

## Orchestrateurs Couverts

| Outil | Couverture | Rôle |
|------|----------|------|
| **Prefect** | Focus principal | Pratique hands-on, approche Pythonique, automatisation réelle |
| **Dagster** | Bonus | Alternative moderne, paradigme centré sur les assets |
| **Airflow** | Vue d'ensemble | Référence standard de l'industrie, montre la complexité |

---

## Patterns d'Orchestration Clés

L'atelier enseigne les patterns d'orchestration à travers des cas d'usage ML :

| Pattern | Problème ML | Solution |
|---------|-----------|----------|
| **Réessais** | Échecs aléatoires d'API | `@task(retries=3, retry_delay_seconds=60)` |
| **Backoff Exponentiel** | Limites de débit | `retry_delay_seconds=[10, 30, 60]` |
| **Cache** | Feature engineering coûteux | `cache_key_fn=task_input_hash` |
| **Parallélisme** | Comparer plusieurs modèles | Exécuter les tâches d'entraînement en parallèle |
| **Paramètres** | Tuning d'hyperparamètres | `@flow` avec arguments typés |
| **Planifications** | Réentraînement quotidien | `cron="0 6 * * *"` |

---

## Comparaison des Trois Orchestrateurs

### Airflow - Infrastructure Lourde, Centré sur les Tâches

**Philosophie :** "Je définis un graphe de tâches à exécuter"

**Points de friction pour le ML :**
- XCom limité à 48KB - doit sérialiser les DataFrames sur disque
- Nécessite des tâches de nettoyage pour les fichiers temporaires
- Infrastructure lourde

### Prefect - Pythonique, Centré sur les Tâches

**Philosophie :** "J'écris des fonctions Python, Prefect gère l'orchestration"

**Pourquoi c'est mieux pour le ML :**
```python
@task(retries=3)
def my_task(input_df: pd.DataFrame) -> pd.DataFrame:
    return result_df  # Juste le retourner ! Pas d'I/O !

@flow
def my_pipeline():
    data = load_data()
    result = my_task(data)  # Appel de fonction normal
```

### Dagster - Déclaratif, Centré sur les Assets

**Philosophie :** "Je définis les assets de données que je veux qui existent"

**Le changement de paradigme :**
```python
@asset
def customer_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Dagster infère la dépendance depuis le nom du paramètre !
    return processed_df
```

---

## Répartition des Responsabilités

| Préoccupation | Orchestrateur | MLflow |
|---------|--------------|--------|
| Réessayer en cas d'échec | X | |
| Cacher les calculs | X | |
| Exécuter les tâches en parallèle | X | |
| Planifier les pipelines | X | |
| Logger les paramètres | | X |
| Logger les métriques | | X |
| Stocker les artefacts de modèles | | X |
| Versionner les modèles | | X |
| Servir les modèles | | X |

**Point clé :** Les orchestrateurs gèrent COMMENT votre pipeline s'exécute. MLflow gère CE QUI est suivi.

---

## Déroulement de l'Atelier

### Configuration & Docker (~15 min)
1. Démarrer la stack Docker, vérifier que les services sont sains
2. Installer les dépendances Python
3. Générer les données d'exemple

### Partie 1 : Fondamentaux MLflow (~45 min)
1. Commencer avec le notebook désordonné - identifier les points de friction
2. Travailler sur le notebook de transition MLflow
3. Apprendre le tracking, les modèles, le registre, le serving

### Partie 2 : Patterns d'Orchestration (~50 min)
1. Travailler sur les Parties 1-5 de l'Atelier Prefect
2. Apprendre les réessais, le cache, le parallélisme, les paramètres
3. Construire un pipeline ML complet avec MLflow

### Partie 3 : Automatisation Réelle (~20 min)
1. Déployer un flow planifié (Partie 6)
2. Le regarder s'exécuter automatiquement
3. Voir les expérimentations apparaître dans MLflow

### Partie 4 : Bonus Dagster (si le temps le permet, ~10 min)
1. Explorer le paradigme centré sur les assets
2. Activer les planifications dans l'interface
3. Comparer avec l'approche Prefect

---

## Acquis des Étudiants

### Compréhension Technique :
1. **Composants MLflow :** Tracking, Modèles, Registre, Serving
2. **Patterns d'orchestration :** Réessais, cache, parallélisme, paramètres, planifications
3. **Automatisation réelle :** Déployer des flows, définir des planifications, les regarder s'exécuter
4. **Compromis de conception :** Centré sur les tâches vs centré sur les assets

### Compétences Pratiques :
- Construire des pipelines ML de production avec tracking MLflow
- Implémenter des patterns d'orchestration dans Prefect
- Déployer et planifier des pipelines automatisés
- Utiliser Docker pour l'infrastructure ML

### Langage CV/Entretien :
> "J'ai implémenté des pipelines ML en utilisant MLflow pour le suivi des expérimentations et le registre de modèles, avec Prefect pour l'orchestration et la planification. Je peux déployer des pipelines de réentraînement automatisés et je comprends les compromis entre les différentes approches d'orchestration."

---

## Ressources Supplémentaires

- **Documentation MLflow :** https://mlflow.org/docs/latest/index.html
- **Documentation Prefect :** https://docs.prefect.io/
- **Documentation Dagster :** https://docs.dagster.io/
- **Documentation Airflow :** https://airflow.apache.org/docs/
