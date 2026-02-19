# Aide-mémoire MLflow

Référence rapide pour l'atelier. Gardez ceci ouvert pendant que vous travaillez sur les notebooks !

---

## Configuration & Connexion

```python
import mlflow
import mlflow.sklearn

# Se connecter au serveur de tracking
mlflow.set_tracking_uri("http://localhost:5000")

# Créer/sélectionner une expérience
mlflow.set_experiment("my-experiment")
```

---

## Structure de Base d'un Run

```python
with mlflow.start_run(run_name="my-run"):
    # Votre code d'entraînement ici
    mlflow.log_param("key", value)
    mlflow.log_metric("accuracy", 0.95)
```

---

## Logger les Paramètres

```python
# Paramètre unique
mlflow.log_param("learning_rate", 0.01)

# Plusieurs paramètres (recommandé)
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.01
})
```

---

## Logger les Métriques

```python
# Métrique unique
mlflow.log_metric("accuracy", 0.95)

# Plusieurs métriques (recommandé)
mlflow.log_metrics({
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.89,
    "f1": 0.90
})

# Métrique avec step (pour les courbes d'entraînement)
mlflow.log_metric("loss", 0.5, step=1)
mlflow.log_metric("loss", 0.3, step=2)
```

---

## Logger les Modèles

```python
# Modèle sklearn
mlflow.sklearn.log_model(model, name="model")

# Charger le modèle plus tard
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Ou depuis le registre
loaded_model = mlflow.sklearn.load_model("models:/model-name/latest")
```

---

## Logger les Artefacts

### Figures (matplotlib)
```python
fig, ax = plt.subplots()
ax.plot(data)
mlflow.log_figure(fig, "plot.png")
plt.close()
```

### Fichiers (pickles, CSVs, etc.)
```python
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "data.csv")
    df.to_csv(path)
    mlflow.log_artifact(path)  # Racine des artefacts
    # OU
    mlflow.log_artifact(path, artifact_path="data")  # Dans un sous-dossier
```

---

## Tags

```python
# Tag unique
mlflow.set_tag("author", "your-name")

# Plusieurs tags
mlflow.set_tags({
    "author": "your-name",
    "model_type": "RandomForest",
    "stage": "experimentation"
})
```

---

## Rechercher des Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("my-experiment")

# Trouver les meilleurs runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.f1 DESC"],
    max_results=10
)

for run in runs:
    print(f"{run.info.run_name}: {run.data.metrics}")
```

---

## Registre de Modèles

```python
# Enregistrer un modèle
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="my-model"
)

# Charger depuis le registre
model = mlflow.sklearn.load_model("models:/my-model/latest")
# OU version spécifique
model = mlflow.sklearn.load_model("models:/my-model/1")
```

---

## Autologging (Alternative au Manuel)

```python
# Activer autolog pour sklearn
mlflow.sklearn.autolog()

# Maintenant entraînez simplement - MLflow logge tout automatiquement !
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Désactiver quand terminé
mlflow.sklearn.autolog(disable=True)
```

**Note** : Les métriques d'autolog sont sur les données d'ENTRAÎNEMENT. Ajoutez les métriques de test manuellement pour une comparaison équitable.

---

## Patterns Courants

### Run d'Entraînement Complet
```python
with mlflow.start_run(run_name="full-example"):
    # Tags
    mlflow.set_tag("author", "workshop")

    # Paramètres
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # Entraînement
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métriques
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

    # Modèle
    mlflow.sklearn.log_model(model, name="model")

    # Graphique
    fig, ax = plt.subplots()
    # ... créer le graphique ...
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()
```

### Sauvegarder le Prétraitement (Scaler)
```python
import joblib
import tempfile
import os

with mlflow.start_run():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sauvegarder le scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scaler.pkl")
        joblib.dump(scaler, path)
        mlflow.log_artifact(path, artifact_path="preprocessing")

    mlflow.set_tag("requires_scaling", "true")
```

### Charger le Prétraitement Plus Tard
```python
run_id = "your-run-id"
artifact_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
scaler = joblib.load(artifact_path)
```

---

## Débogage Rapide

```python
# URI de tracking actuel
print(mlflow.get_tracking_uri())

# Expérience actuelle
print(mlflow.get_experiment_by_name("name"))

# Lister les artefacts d'un run
artifacts = client.list_artifacts(run_id)
for a in artifacts:
    print(a.path)
```

---

## Accès à l'Interface

- **Interface MLflow** : http://localhost:5000
- **Expériences** : Barre latérale gauche
- **Runs** : Cliquez sur le nom de l'expérience
- **Artefacts** : Cliquez sur le run, puis onglet "Artifacts"
- **Comparer** : Sélectionnez les runs avec les cases, cliquez "Compare"

---

## Résumé des Concepts Clés

| Concept | Ce que c'est | Exemple |
|---------|--------------|---------|
| **Experiment** | Groupe de runs liés | "churn-prediction" |
| **Run** | Une seule exécution | "rf-100-trees" |
| **Parameter** | Entrée du modèle | n_estimators=100 |
| **Metric** | Sortie/résultat | accuracy=0.95 |
| **Artifact** | Fichier (modèle, graphique) | model.pkl |
| **Tag** | Métadonnée | author="me" |
| **Registry** | Versionnage de modèles | "prod-model" v1, v2 |
| **Projects** | Code reproductible | fichier MLproject |

---

## Support LLM/GenAI (Bonus)

MLflow supporte maintenant les applications LLM :

```python
# Tracer les appels LLM
@mlflow.trace
def chat_with_llm(prompt):
    response = openai_client.chat(prompt)
    return response

# Logger un modèle LLM
mlflow.openai.log_model(...)

# Évaluer les sorties LLM
mlflow.evaluate(model, data, model_type="text")
```

**Cas d'usage** : Pipelines RAG, débogage de chatbots, monitoring d'agents
