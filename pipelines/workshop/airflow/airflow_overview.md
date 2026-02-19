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

### Pain Point 1: XCom Limitations

XCom is how tasks share data, but it has a **48KB limit** by default.

**Problem**: You can't pass DataFrames through XCom.

```python
# What you WANT to do (but can't):
def load_data(**context):
    df = pd.read_csv('data.csv')
    return df  # Too large for XCom!

# What you HAVE to do:
def load_data(**context):
    df = pd.read_csv('data.csv')

    # Save to disk
    temp_path = f"/tmp/data_{context['run_id']}.parquet"
    df.to_parquet(temp_path)

    # Push the PATH (not the data)
    context['ti'].xcom_push(key='data_path', value=temp_path)
```

### Pain Point 2: File I/O Everywhere

Every task that needs data must:
1. Pull the file path from XCom
2. Load data from disk
3. Process it
4. Save to a new file
5. Push the new path to XCom

```python
def process_data(**context):
    # Pull path from previous task
    path = context['ti'].xcom_pull(key='data_path', task_ids='load_data')

    # Load from disk
    df = pd.read_parquet(path)

    # Process
    df = df.dropna()

    # Save to new file
    new_path = f"/tmp/processed_{context['run_id']}.parquet"
    df.to_parquet(new_path)

    # Push new path
    context['ti'].xcom_push(key='data_path', value=new_path)
```

**Compare with Prefect:**
```python
@task
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()  # That's it!
```

### Pain Point 3: Manual Cleanup

All those temp files? You need to clean them up:

```python
def cleanup_temp_files(**context):
    """Delete all temp files created during the DAG run."""
    paths = []
    paths.append(context['ti'].xcom_pull(key='data_path', task_ids='load_data'))
    paths.append(context['ti'].xcom_pull(key='data_path', task_ids='process'))
    paths.append(context['ti'].xcom_pull(key='model_path', task_ids='train'))
    # ... every task that created a file

    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)

# Must run after everything, even on failure
cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_temp_files,
    trigger_rule='all_done',
    dag=dag
)
```

### Pain Point 4: Verbose Configuration

Before any business logic, you need ~30 lines of boilerplate:

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

# NOW you can define tasks...
load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)
```

### Pain Point 5: Dependencies Declared Separately

Task dependencies are defined separately from the functions:

```python
# Define tasks
load = PythonOperator(task_id='load', ...)
process = PythonOperator(task_id='process', ...)
train = PythonOperator(task_id='train', ...)
evaluate = PythonOperator(task_id='evaluate', ...)

# Dependencies declared separately (easy to get wrong)
load >> process >> train >> evaluate
```

**Compare with Prefect:**
```python
@flow
def pipeline():
    data = load()
    processed = process(data)  # Dependency is obvious!
    model = train(processed)
    evaluate(model)
```

---

## Reference Implementation

See the complete Airflow implementation:

```
pipelines/examples/Airflow_ML_Pipeline.py
```

This file implements the same ML pipeline as Prefect and Dagster, showing:
- XCom push/pull patterns
- File I/O for DataFrames
- Cleanup task implementation
- MLflow integration with Airflow
- `**context` kwargs usage

### Key Sections to Review

| Line Range | What It Shows |
|------------|---------------|
| 44-48 | Configuration and paths |
| 60-100 | `load_customer_data` with XCom push |
| 100-140 | `engineer_features` with XCom pull/push |
| 200-250 | `train_model` with pickle serialization |
| 350-380 | `cleanup_temp_files` task |
| 400-450 | DAG definition and task wiring |

---

## When Airflow Makes Sense

Despite the complexity, Airflow is the right choice when:

| Scenario | Why Airflow |
|----------|-------------|
| Existing infrastructure | Your company already uses it |
| Cross-system orchestration | Need to coordinate Spark, databases, cloud services |
| Enterprise requirements | Audit logs, RBAC, compliance |
| Dedicated platform team | Can handle the operational complexity |
| Non-Python workflows | Need BashOperator, KubernetesOperator, etc. |

---

## Summary: Airflow vs Prefect

| Aspect | Airflow | Prefect |
|--------|---------|---------|
| Data passing | XCom + temp files | Return values |
| Configuration | 30+ lines boilerplate | 2 decorators |
| Cleanup | Manual task required | Automatic |
| Dependencies | `task1 >> task2` | Function calls |
| Type hints | Not supported | Native Python |
| Learning curve | Steep | Gentle |

---

## Next Steps

1. **Review the example**: Open `pipelines/examples/Airflow_ML_Pipeline.py`
2. **Compare with Prefect**: Look at the same operations in `pipelines/examples/Prefect_ML_Pipeline.py`
3. **Understand the trade-offs**: Airflow's complexity buys you enterprise features and ecosystem
