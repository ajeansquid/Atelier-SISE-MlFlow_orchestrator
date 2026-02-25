"""
Génère des données clients synthétiques pour l'atelier MLflow + Orchestrateurs.
Ce script crée des données clients synthétiques qui peuvent être utilisées par :
- Les notebooks Jupyter (01_messy_notebook.ipynb, 02_mlflow_organized.ipynb)
- Les pipelines d'orchestration (Prefect, Airflow, Dagster)

Exécutez ce script pour créer data/customer_data.csv :
    python generate_sample_data.py
Les pipelines utiliseront automatiquement ce CSV s'il existe,
sinon ils génèrent des données synthétiques à la volée.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_PATH = Path("data/customer_data.csv")
RANDOM_SEED = 42
N_CUSTOMERS = 5000


def generate_customer_data(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Génère des données clients synthétiques pour la prédiction de churn.

    Features :
    - customer_id: Identifiant unique du client
    - recency_days: Jours depuis le dernier achat
    - frequency: Nombre d'achats dans la dernière période
    - monetary_value: Dépense totale
    - avg_order_value: Valeur moyenne de commande
    - days_since_signup: Ancienneté du client
    - total_orders: Nombre total de commandes
    - support_tickets: Nombre de tickets support ouverts
    - age: Âge du client
    - churned: Variable cible (1 = churné, 0 = actif)

    Les caractéristiques sont générées avec des corrélations réalistes :
    - Plus d'ancienneté → plus de commandes totales
    - Plus de fréquence → plus de valeur monétaire
    - Clients actifs (faible récence) → plus d'engagement
    - Clients à valeur élevée → potentiellement plus de tickets support
    """
    np.random.seed(seed)

    # Commencer par l'ancienneté client (caractéristique fondamentale)
    days_since_signup = np.random.gamma(shape=2.5, scale=200, size=n_customers).astype(int)
    days_since_signup = np.clip(days_since_signup, 30, 1500)
    
    # Générer l'âge du client (indépendant)
    age = np.random.normal(40, 15, n_customers).astype(int)
    age = np.clip(age, 18, 75)
    
    # Commandes totales corrélées avec l'ancienneté (clients plus anciens → plus de commandes)
    tenure_factor = days_since_signup / days_since_signup.mean()
    base_orders = np.random.poisson(lam=8, size=n_customers)
    total_orders = (base_orders * tenure_factor).astype(int)
    total_orders = np.clip(total_orders, 1, None)  # Au moins 1 commande
    
    # Fréquence (achats récents) corrélée avec les commandes totales mais avec du bruit
    order_engagement = total_orders / (days_since_signup + 1) * 365  # Commandes par an
    frequency = np.random.poisson(lam=np.clip(order_engagement * 2, 1, 20), size=n_customers)
    
    # Valeur monétaire corrélée avec la fréquence et les commandes totales
    # Clients à haute fréquence et nombreuses commandes dépensent plus
    base_monetary = np.random.gamma(shape=2, scale=400, size=n_customers)
    engagement_boost = 1 + 0.5 * (frequency / frequency.mean())
    order_boost = 1 + 0.3 * (total_orders / total_orders.mean())
    monetary_value = (base_monetary * engagement_boost * order_boost).round(2)
    monetary_value = np.clip(monetary_value, 50, None)
    
    # Valeur moyenne de commande (dérivée mais bruitée)
    avg_order_value = monetary_value / (total_orders + 1)
    # Ajouter du bruit pour le rendre réaliste
    avg_order_value = avg_order_value * np.random.normal(1, 0.1, n_customers)
    avg_order_value = np.clip(avg_order_value, 10, None).round(2)
    
    # Récence : les clients actifs (haute fréquence) ont une récence plus faible
    # Les clients inactifs ont une récence plus élevée
    base_recency = np.random.exponential(scale=30, size=n_customers)
    # Inverser la fréquence : haute fréquence → faible récence
    recency_penalty = 1 + 2 * (1 - frequency / (frequency.max() + 1))
    recency_days = (base_recency * recency_penalty).astype(int)
    recency_days = np.clip(recency_days, 0, 365)
    
    # Tickets support : quelque peu corrélés avec les commandes (plus de commandes → plus de tickets)
    # Mais aussi aléatoire (certains clients se plaignent plus)
    base_tickets = np.random.poisson(lam=1, size=n_customers)
    order_factor = 1 + 0.3 * (total_orders / total_orders.mean())
    support_tickets = (base_tickets * order_factor * np.random.uniform(0.5, 1.5, n_customers)).astype(int)
    support_tickets = np.clip(support_tickets, 0, None)

    # Créer le DataFrame
    data = {
        'customer_id': range(1, n_customers + 1),
        'recency_days': recency_days,
        'frequency': frequency,
        'monetary_value': monetary_value,
        'avg_order_value': avg_order_value,
        'days_since_signup': days_since_signup,
        'total_orders': total_orders,
        'support_tickets': support_tickets,
        'age': age,
    }

    df = pd.DataFrame(data)

    # Créer la variable cible avec un comportement de churn réaliste
    # Normaliser les caractéristiques
    recency_norm = (df['recency_days'] - df['recency_days'].mean()) / df['recency_days'].std()
    frequency_norm = (df['frequency'] - df['frequency'].mean()) / df['frequency'].std()
    monetary_norm = (df['monetary_value'] - df['monetary_value'].mean()) / df['monetary_value'].std()
    support_norm = (df['support_tickets'] - df['support_tickets'].mean()) / df['support_tickets'].std()
    
    # Construire le score de churn
    churn_score = (
        1.8 * recency_norm -        # HAUTE récence (inactif) → PLUS de churn
        1.5 * frequency_norm -       # HAUTE fréquence (actif) → MOINS de churn  
        1.0 * monetary_norm +        # HAUTE valeur → MOINS de churn
        0.8 * support_norm           # PLUS de tickets → PLUS de churn
    )

    # Convertir en probabilité avec sigmoïde
    churn_prob = 1 / (1 + np.exp(-churn_score))
    # Ajouter du bruit pour éviter une séparabilité parfaite
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.15, n_customers), 0.05, 0.95)
    
    # Générer la cible
    df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

    return df


def main():
    print("=" * 60)
    print("Génération de Données Clients Synthétiques")
    print("=" * 60)

    # Générer les données
    df = generate_customer_data()

    # Afficher le résumé
    print(f"\nDimensions du dataset : {df.shape}")
    print(f"Taux de churn : {df['churned'].mean():.2%}")
    print("\nStatistiques des caractéristiques :")
    print(df.describe().round(2))

    # S'assurer que le répertoire de sortie existe
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder en CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDonnées sauvegardées dans : {OUTPUT_PATH}")

    # Afficher des lignes d'exemple
    print("\nLignes d'exemple :")
    print(df.head(10).to_string())

    print("\n" + "=" * 60)
    print("Terminé ! Vous pouvez maintenant exécuter les notebooks et pipelines.")
    print("=" * 60)


if __name__ == "__main__":
    main()
