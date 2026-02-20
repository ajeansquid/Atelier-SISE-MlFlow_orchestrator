# Configuration Docker pour l'Atelier MLflow + Orchestrateurs

## Vue d'Ensemble de l'Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DOCKER COMPOSE                                  │
│                                                                          │
│  ┌──────────────┐   ┌────────────────────────┐   ┌───────────────────┐  │
│  │              │   │      PREFECT           │   │      DAGSTER      │  │
│  │   MLFLOW     │   │  ┌────────┐ ┌───────┐  │   │  ┌───────────┐   │  │
│  │   SERVER     │◀──│  │ Server │ │Worker │  │   │  │ Webserver │   │  │
│  │              │   │  │ :4200  │ │       │  │   │  │  :3000    │   │  │
│  │  :5000       │   │  └────────┘ └───┬───┘  │   │  └───────────┘   │  │
│  │              │   │                 │      │   │  ┌───────────┐   │  │
│  │  Experiments │   │    Executes ────┘      │   │  │  Daemon   │   │  │
│  │  Models      │   │    scheduled           │   │  │ (executes │   │  │
│  │  Artifacts   │   │    flows               │   │  │ schedules)│   │  │
│  │              │   │                        │   │  └─────┬─────┘   │  │
│  └──────────────┘   └────────────────────────┘   └────────┼─────────┘  │
│         ▲                      │                          │             │
│         │                      │                          │             │
│         └──────────────────────┴──────────────────────────┘             │
│                    Logs experiments to MLflow                            │
└─────────────────────────────────────────────────────────────────────────┘

VOTRE MACHINE :
  - VS Code avec notebooks (apprentissage)
  - Déployer des flows via CLI
  - Observer l'automatisation dans les interfaces
```

**Point Clé :** Tout s'exécute dans Docker. Vous déployez des flows et les regardez s'exécuter automatiquement. C'est de la VRAIE orchestration.

---

## Démarrage Rapide

### Atelier Principal : MLflow + Prefect

```bash
# Démarrer MLflow + Prefect (serveur + worker)
docker-compose up -d

# Vérifier que les services fonctionnent
docker-compose ps

# Accéder aux interfaces
# Prefect: http://localhost:4200
# MLflow:  http://localhost:5000
```

### Bonus : Ajouter Dagster

```bash
# Démarrer tout incluant Dagster
docker-compose --profile dagster up -d

# Accéder à l'interface Dagster
# http://localhost:3000
```

---

## Détails des Services

| Service | Port | Objectif |
|---------|------|----------|
| **mlflow** | 5000 | Suivi d'expérimentations, registre de modèles |
| **prefect-server** | 4200 | Monitoring de flows, déploiements, planifications |
| **prefect-worker** | - | Exécute les flows planifiés |
| **dagster-webserver** (bonus) | 3000 | Interface graphe d'assets, matérialisations |
| **dagster-daemon** (bonus) | - | Exécute les jobs planifiés |

---

## Flux de Travail de l'Atelier

### 1. Démarrer la Stack

```bash
docker-compose up -d
```

Attendre que les services soient en bonne santé :
```bash
docker-compose ps
# Chercher le statut "healthy" sur mlflow et prefect-server
```

### 2. Apprendre les Patterns (Parties 1-5)

Exécuter les parties de l'atelier localement pour apprendre les patterns d'orchestration :

```bash
# Installer les dépendances
pip install -r requirements.txt

# Exécuter les parties de l'atelier
python pipelines/workshop/02_prefect/Prefect_Workshop.py part1  # Tasks & Flows
python pipelines/workshop/02_prefect/Prefect_Workshop.py part2  # Réessais
python pipelines/workshop/02_prefect/Prefect_Workshop.py part3  # Cache & Parallélisme
python pipelines/workshop/02_prefect/Prefect_Workshop.py part4  # Paramètres
python pipelines/workshop/02_prefect/Prefect_Workshop.py part5  # Pipeline Complet + MLflow
python pipelines/workshop/02_prefect/Prefect_Workshop.py part7  # Notifications Discord/Slack
```

### 3. Voir l'Automatisation Réelle (Partie 6)

Déployer un flow planifié :

```bash
python pipelines/workshop/02_prefect/Prefect_Workshop.py deploy
```

Ceci va :
1. Enregistrer le flow auprès du serveur Prefect
2. Créer une planification (toutes les 2 minutes)
3. Continuer de s'exécuter jusqu'à ce que vous appuyiez sur Ctrl+C

Maintenant observez :
- **Interface Prefect** (http://localhost:4200) : Voir les déploiements et exécutions
- **Interface MLflow** (http://localhost:5000) : Voir les expérimentations apparaître automatiquement

### 4. Bonus : Dagster

```bash
# Démarrer Dagster
docker-compose --profile dagster up -d
```

Ouvrir http://localhost:3000 :
1. **Graphe d'Assets** : Voir le linéage de données
2. **Matérialiser** : Cliquer pour exécuter des assets
3. **Planifications** : Activer `churn_training_schedule`
4. **Observer** : Les exécutions apparaissent toutes les 2 minutes !

---

## Commandes Courantes

```bash
# Démarrer l'atelier principal (MLflow + Prefect)
docker-compose up -d

# Démarrer avec le bonus Dagster
docker-compose --profile dagster up -d

# Arrêter tous les services
docker-compose down

# Arrêter et supprimer les données (repartir de zéro)
docker-compose down -v

# Voir les logs
docker-compose logs -f mlflow
docker-compose logs -f prefect-server
docker-compose logs -f prefect-worker
docker-compose logs -f dagster-webserver
docker-compose logs -f dagster-daemon

# Reconstruire les images après modifications des Dockerfile
docker-compose build prefect-worker
docker-compose --profile dagster build

# Redémarrer un service
docker-compose restart prefect-worker
```

---

## Dépannage

### Les services ne démarrent pas

```bash
# Vérifier le statut
docker-compose ps

# Vérifier les logs pour les erreurs
docker-compose logs mlflow
docker-compose logs prefect-server
docker-compose logs prefect-worker
```

### Port déjà utilisé

Copier `.env.example` vers `.env` et ajuster les ports :

```bash
cp .env.example .env
```

Ensuite éditer `.env` avec vos ports souhaités :

```bash
MLFLOW_PORT=5001
PREFECT_PORT=4201
DAGSTER_PORT=3001
```

### Le worker Prefect n'exécute pas les flows

```bash
# Vérifier les logs du worker
docker-compose logs -f prefect-worker

# Redémarrer le worker
docker-compose restart prefect-worker
```

### Les planifications Dagster ne s'exécutent pas

S'assurer que le daemon tourne :
```bash
docker-compose logs -f dagster-daemon
```

### MLflow non accessible depuis les conteneurs

Les conteneurs utilisent le réseau Docker. Depuis les conteneurs :
- MLflow : `http://mlflow:5000`
- Prefect : `http://prefect-server:4200/api`

Depuis votre machine :
- MLflow : `http://localhost:5000`
- Prefect : `http://localhost:4200`

---

## Persistance des Données

Les données sont stockées dans des volumes Docker :

| Volume | Contenu |
|--------|----------|
| `workshop-mlflow-data` | Expérimentations, exécutions, modèles |
| `workshop-prefect-data` | Historique des exécutions de flows, déploiements |
| `workshop-dagster-data` | Matérialisations d'assets |

Pour réinitialiser toutes les données :
```bash
docker-compose down -v
```

---

## Pour les Formateurs

### Avant l'Atelier

```bash
# État propre
docker-compose down -v

# Télécharger et construire les images
docker-compose pull
docker-compose build
docker-compose --profile dagster build

# Démarrer et vérifier
docker-compose up -d
docker-compose ps  # Tous doivent être en bonne santé (healthy)
```

### Pendant l'Atelier

Les étudiants exécutent :
```bash
docker-compose up -d
pip install -r requirements.txt
```

Puis suivent le flux de l'atelier :
1. Parties 1-5 : Apprendre les patterns localement
2. Partie 6 : Déployer et observer l'automatisation
3. Bonus : Explorer Dagster

### Points Pédagogiques Clés

1. **Pourquoi Docker ?** L'orchestration réelle nécessite de l'infrastructure (serveurs, workers, daemons)
2. **Architecture Prefect** : Serveur (API + UI) + Worker (exécute les flows)
3. **Architecture Dagster** : Webserver (UI) + Daemon (exécute les planifications)
4. **Intégration MLflow** : Les deux orchestrateurs loguent vers le même serveur MLflow
5. **Automatisation** : Les flows s'exécutent selon la planification sans intervention manuelle
