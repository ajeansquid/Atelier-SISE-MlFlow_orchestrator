# Approches de Déploiement Prefect

Ce document compare les différentes approches pour déployer des flows Prefect, du plus simple au plus avancé.

## Vue d'ensemble

| Approche | Complexité | Cas d'usage | Infrastructure |
|----------|------------|-------------|----------------|
| `flow.serve()` | Simple | Développement, petites équipes | Local ou VM unique |
| `prefect deployment build` | Moyenne | Production, CI/CD | Agent + serveur Prefect |
| Work pools + agents | Avancée | Production scale | Kubernetes, Docker, Cloud |
| Prefect Cloud | Managé | Équipes sans ops | Aucune (SaaS) |

---

## 1. `flow.serve()` - Approche Simple (Notre Atelier)

C'est l'approche que nous enseignons dans l'atelier. Elle est parfaite pour :
- Apprentissage des concepts Prefect
- Développement local
- Petites équipes avec infrastructure simple

### Comment ça marche

```python
from prefect import flow

@flow
def my_pipeline():
    # ... votre code ML
    pass

if __name__ == "__main__":
    # Démarre un serveur local qui exécute le flow selon le cron
    my_pipeline.serve(
        name="mon-deployment",
        cron="*/2 * * * *",  # Toutes les 2 minutes
        tags=["ml", "production"]
    )
```

### Avantages

- **Zéro infrastructure** : Pas besoin de serveur Prefect séparé
- **Un seul processus** : Le script fait tout (scheduler + worker)
- **Facile à comprendre** : Ressemble à du Python normal
- **Parfait pour l'apprentissage** : Moins de concepts à assimiler

### Inconvénients

- **Un seul processus** : Si le script s'arrête, tout s'arrête
- **Pas de distribution** : Un seul worker exécute tout
- **Pas de haute disponibilité** : Pas de failover automatique

### Quand l'utiliser

- Développement local
- POC et prototypage
- Petits pipelines de données
- Environnements simples (une seule VM)

---

## 2. `prefect deployment build` - Approche CLI (DataCamp)

C'est l'approche utilisée dans le [tutoriel DataCamp](datacamp_tutorial.md). Plus robuste que `serve()`.

### Comment ça marche

```bash
# 1. Construire le déploiement (génère un YAML)
prefect deployment build main.py:my_pipeline \
    -n 'mon-deployment' \
    -a \
    --tag production

# 2. Démarrer l'agent dans un terminal séparé
prefect agent start -p 'default-agent-pool'

# 3. Exécuter le déploiement (ou laisser le scheduler le faire)
prefect deployment run 'my-pipeline/mon-deployment'
```

### Avantages

- **Séparation des rôles** : CLI pour le déploiement, agent pour l'exécution
- **Fichier YAML** : Configuration versionnée et auditable
- **Intégration CI/CD** : Facile à automatiser avec GitHub Actions
- **Scaling** : Plusieurs agents peuvent traiter les flows

### Inconvénients

- **Plus de commandes** : Build, agent, run séparés
- **Infrastructure** : Besoin d'un agent qui tourne en permanence
- **Courbe d'apprentissage** : Plus de concepts à comprendre

### Quand l'utiliser

- Production avec CI/CD
- Équipes avec processus DevOps établis
- Besoin de scaling horizontal

---

## 3. Work Pools - Approche Avancée

Les Work Pools permettent de contrôler l'infrastructure d'exécution.

### Types de Work Pools

| Type | Infrastructure | Cas d'usage |
|------|----------------|-------------|
| Process | Local | Développement |
| Docker | Conteneurs | Environnements isolés |
| Kubernetes | Cluster K8s | Production scale |
| prefect:managed | Prefect Cloud | Zero ops |

### Exemple avec Docker

```python
from prefect import flow
from prefect.infrastructure import DockerContainer

docker_block = DockerContainer(
    image="my-ml-image:latest",
    image_pull_policy="ALWAYS",
    auto_remove=True
)

my_pipeline.deploy(
    name="docker-deployment",
    work_pool_name="my-docker-pool",
    image="my-ml-image:latest"
)
```

### Quand l'utiliser

- Production à grande échelle
- Besoin d'isolation (conteneurs)
- Infrastructure Kubernetes existante

---

## 4. Prefect Cloud - Approche Managée

Prefect propose un service cloud avec :
- 10 heures de calcul gratuites/mois
- Serveur Prefect managé
- Work pools managés
- Pas d'infrastructure à gérer

### Connexion

```bash
# Se connecter à Prefect Cloud
prefect cloud login

# Créer un work pool managé
prefect work-pool create my-pool --type prefect:managed
```

### Déploiement vers le Cloud

```python
from prefect import flow

@flow
def my_pipeline():
    pass

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/mon-repo/mon-projet.git",
        entrypoint="main.py:my_pipeline"
    ).deploy(
        name="cloud-deployment",
        work_pool_name="my-managed-pool",
        tags=["production"],
        job_variables={"pip_packages": ["pandas", "scikit-learn"]}
    )
```

### Avantages

- **Zéro ops** : Pas d'infrastructure à maintenir
- **Haute disponibilité** : SLA garanti
- **UI avancée** : Dashboards, événements, automatisations

### Inconvénients

- **Coût** : Plans payants pour usage intensif
- **Dépendance** : Vendor lock-in
- **Données** : Le code doit être accessible (GitHub)

---

## Pourquoi nous enseignons `serve()` d'abord

### Philosophie pédagogique

1. **Moins de concepts** : Un seul fichier Python, pas de CLI complexe
2. **Résultat immédiat** : Le flow s'exécute tout de suite
3. **Focus sur l'orchestration** : Pas sur l'infrastructure
4. **Progression naturelle** : Simple → Complexe

### Progression recommandée

```
Débutant          Intermédiaire        Avancé
─────────────────────────────────────────────────►
  serve()     →   build + agent   →   Work pools
                                         │
                                         v
                                   Prefect Cloud
```

### Dans l'atelier

| Partie | Ce qu'on enseigne | Approche |
|--------|-------------------|----------|
| 1-4 | Patterns Prefect | Exécution directe |
| 5 | Pipeline MLflow | Exécution directe |
| 6 | Automatisation | `serve()` |
| 7 | Notifications | Webhooks |

---

## Comparaison : serve() vs build + agent

### Code minimal avec serve()

```python
# main.py - Un seul fichier !
from prefect import flow

@flow
def my_pipeline():
    print("Hello!")

if __name__ == "__main__":
    my_pipeline.serve(name="simple", cron="0 * * * *")
```

```bash
# Une seule commande
python main.py
```

### Code avec build + agent

```python
# main.py
from prefect import flow

@flow
def my_pipeline():
    print("Hello!")

if __name__ == "__main__":
    my_pipeline()
```

```bash
# Plusieurs étapes
prefect server start  # Terminal 1
prefect deployment build main.py:my_pipeline -n "simple" -a  # Terminal 2
prefect agent start -p 'default-agent-pool'  # Terminal 3
prefect deployment run 'my-pipeline/simple'  # Terminal 4 (ou via UI)
```

### Verdict

| Critère | serve() | build + agent |
|---------|---------|---------------|
| Lignes de commande | 1 | 4+ |
| Terminaux ouverts | 1 | 3 |
| Fichiers générés | 0 | 1 (YAML) |
| Temps de démarrage | Immédiat | ~2 min |
| Production-ready | Non | Oui |

---

## Conclusion

Pour l'atelier Master 2 SISE, `flow.serve()` est le choix optimal :
- Focus sur les **concepts d'orchestration**, pas l'infrastructure
- Résultat **visible immédiatement**
- Transition vers `build + agent` **facile** une fois les bases acquises

Pour la production, évaluez vos besoins :
- **Petite équipe, scripts simples** → `serve()` suffit
- **CI/CD, scaling** → `build + agent`
- **Entreprise, SLA** → Work pools ou Prefect Cloud
