"""
Generate sample customer data for the MLflow + Orchestrators workshop.

This script creates synthetic customer data that can be used by:
- The Jupyter notebooks (01_messy_notebook.ipynb, 02_mlflow_organized.ipynb)
- The orchestrator pipelines (Prefect, Airflow, Dagster)

Run this script to create data/customer_data.csv:
    python generate_sample_data.py

The pipelines will automatically use this CSV if it exists,
otherwise they generate synthetic data on the fly.
"""

import pandas as pd
import numpy as np
import os

# Configuration
OUTPUT_PATH = "data/customer_data.csv"
RANDOM_SEED = 42
N_CUSTOMERS = 5000


def generate_customer_data(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic customer data for churn prediction.

    Features:
    - customer_id: Unique customer identifier
    - recency_days: Days since last purchase
    - frequency: Number of purchases in last period
    - monetary_value: Total spend
    - avg_order_value: Average order value
    - days_since_signup: Customer tenure
    - total_orders: Lifetime orders
    - support_tickets: Number of support tickets opened
    - age: Customer age
    - churned: Target variable (1 = churned, 0 = active)

    Features are generated with realistic correlations:
    - Longer tenure → more total orders
    - Higher frequency → higher monetary value
    - Active customers (low recency) → higher engagement
    - Higher value customers → potentially more support tickets
    """
    np.random.seed(seed)

    # Start with customer tenure (foundational feature)
    days_since_signup = np.random.gamma(shape=2.5, scale=200, size=n_customers).astype(int)
    days_since_signup = np.clip(days_since_signup, 30, 1500)
    
    # Generate customer age (independent)
    age = np.random.normal(40, 15, n_customers).astype(int)
    age = np.clip(age, 18, 75)
    
    # Total orders correlated with tenure (longer customers → more orders)
    tenure_factor = days_since_signup / days_since_signup.mean()
    base_orders = np.random.poisson(lam=8, size=n_customers)
    total_orders = (base_orders * tenure_factor).astype(int)
    total_orders = np.clip(total_orders, 1, None)  # At least 1 order
    
    # Frequency (recent purchases) correlated with total orders but with noise
    order_engagement = total_orders / (days_since_signup + 1) * 365  # Orders per year
    frequency = np.random.poisson(lam=np.clip(order_engagement * 2, 1, 20), size=n_customers)
    
    # Monetary value correlated with frequency and total orders
    # High-frequency, high-order customers spend more
    base_monetary = np.random.gamma(shape=2, scale=400, size=n_customers)
    engagement_boost = 1 + 0.5 * (frequency / frequency.mean())
    order_boost = 1 + 0.3 * (total_orders / total_orders.mean())
    monetary_value = (base_monetary * engagement_boost * order_boost).round(2)
    monetary_value = np.clip(monetary_value, 50, None)
    
    # Average order value (derived but noisy)
    avg_order_value = monetary_value / (total_orders + 1)
    # Add some noise to make it realistic
    avg_order_value = avg_order_value * np.random.normal(1, 0.1, n_customers)
    avg_order_value = np.clip(avg_order_value, 10, None).round(2)
    
    # Recency: active customers (high frequency) have lower recency
    # Inactive customers have higher recency
    base_recency = np.random.exponential(scale=30, size=n_customers)
    # Invert frequency: high frequency → low recency
    recency_penalty = 1 + 2 * (1 - frequency / (frequency.max() + 1))
    recency_days = (base_recency * recency_penalty).astype(int)
    recency_days = np.clip(recency_days, 0, 365)
    
    # Support tickets: somewhat correlated with orders (more orders → more tickets)
    # But also random (some customers complain more)
    base_tickets = np.random.poisson(lam=1, size=n_customers)
    order_factor = 1 + 0.3 * (total_orders / total_orders.mean())
    support_tickets = (base_tickets * order_factor * np.random.uniform(0.5, 1.5, n_customers)).astype(int)
    support_tickets = np.clip(support_tickets, 0, None)

    # Create DataFrame
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

    # Create target variable with realistic churn behavior
    # Normalize features
    recency_norm = (df['recency_days'] - df['recency_days'].mean()) / df['recency_days'].std()
    frequency_norm = (df['frequency'] - df['frequency'].mean()) / df['frequency'].std()
    monetary_norm = (df['monetary_value'] - df['monetary_value'].mean()) / df['monetary_value'].std()
    support_norm = (df['support_tickets'] - df['support_tickets'].mean()) / df['support_tickets'].std()
    
    # Build churn score
    churn_score = (
        1.8 * recency_norm -        # HIGH recency (inactive) → MORE churn
        1.5 * frequency_norm -       # HIGH frequency (active) → LESS churn  
        1.0 * monetary_norm +        # HIGH value → LESS churn
        0.8 * support_norm           # MORE tickets → MORE churn
    )

    # Convert to probability with sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_score))
    # Add noise to prevent perfect separability
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.15, n_customers), 0.05, 0.95)
    
    # Generate target
    df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

    return df


def main():
    print("=" * 60)
    print("Generating Sample Customer Data")
    print("=" * 60)

    # Generate data
    df = generate_customer_data()

    # Print summary
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print("\nFeature statistics:")
    print(df.describe().round(2))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nData saved to: {OUTPUT_PATH}")

    # Print sample rows
    print("\nSample rows:")
    print(df.head(10).to_string())

    print("\n" + "=" * 60)
    print("Done! You can now run the notebooks and pipelines.")
    print("=" * 60)


if __name__ == "__main__":
    main()
