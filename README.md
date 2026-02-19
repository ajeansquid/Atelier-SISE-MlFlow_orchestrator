# MLflow + Orchestrators Workshop

## Context

**Duration:** 3 hours
**Audience:** Master 2 SISE students (mixed profiles: data scientists, ML engineers, data engineers)
**Goal:** Learn MLOps practices - orchestrate ML pipelines with scheduling and automation, track experiments with MLflow

---

## Workshop Progression

The pedagogical journey covers the **full ML pipeline**:

```
DEVELOP -> TRACK -> REGISTER -> AUTOMATE -> MONITOR
```

### Phase 1: MLflow Fundamentals (Notebooks)

| Notebook | Purpose |
|----------|---------|
| `01_messy_notebook.ipynb` | Starting point - typical data science chaos |
| `01b_mlflow_transition.ipynb` | Interactive guided transition to MLflow |
| `02_mlflow_organized.ipynb` | Complete MLflow reference solution |

**MLflow Transition Structure:**
- Part 1: Fully guided (tracking basics)
- Part 2: Progressive reveal (models, artifacts, search)
- Part 3: Fill-in-the-blanks (exercises)
- Part 4: Serving & inference (registry, load, serve, batch inference)

### Phase 2: Orchestration with Automation (Main Focus)

**Prefect Workshop** (`pipelines/workshop/prefect/Prefect_Workshop.py`):
- Part 1: Tasks & Flows (basics)
- Part 2: Resilience (retries, exponential backoff)
- Part 3: Efficiency (caching, parallel model training)
- Part 4: Flexibility (parameters, subflows)
- Part 5: Full Pipeline with MLflow integration
- **Part 6: AUTOMATION - Deploy, schedule, watch it run!**

### Phase 3: Dagster Bonus

**Dagster Workshop** (`pipelines/workshop/dagster/Dagster_Workshop.py`):
- Transform tasks to assets
- Automatic dependency inference
- Jobs and schedules for automation
- Visual asset lineage

---

## Architecture: Docker-Based Orchestration

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

YOUR MACHINE:
  - VS Code with notebooks (learning)
  - Deploy flows via CLI
  - Watch automation in UIs
```

**Key Point:** This is REAL orchestration. Flows run automatically on schedules. Workers execute them. You watch in the UIs.

---

## Quick Start

### 1. Start the Stack

```bash
# Start MLflow + Prefect (server + worker)
docker-compose up -d

# Verify services
docker-compose ps
# Wait for "healthy" status

# Optional: Add Dagster
docker-compose --profile dagster up -d
```

### 2. Install Local Dependencies

```bash
pip install -r requirements.txt
```

### 3. Access the UIs

| Service | URL | Purpose |
|---------|-----|---------|
| **Prefect** | http://localhost:4200 | Flow deployments, runs, schedules |
| **MLflow** | http://localhost:5000 | Experiments, models, artifacts |
| **Dagster** | http://localhost:3000 | Asset graph, materializations (bonus) |

### 4. Run the Workshop

**Learn the patterns (Parts 1-5):**
```bash
python pipelines/workshop/prefect/Prefect_Workshop.py part1  # Tasks & Flows
python pipelines/workshop/prefect/Prefect_Workshop.py part2  # Retries
python pipelines/workshop/prefect/Prefect_Workshop.py part3  # Caching & Parallelism
python pipelines/workshop/prefect/Prefect_Workshop.py part4  # Parameters
python pipelines/workshop/prefect/Prefect_Workshop.py part5  # Full Pipeline + MLflow
```

**See real automation (Part 6):**
```bash
python pipelines/workshop/prefect/Prefect_Workshop.py deploy
```

This deploys a scheduled flow that runs every 2 minutes. Watch:
- **Prefect UI**: See deployments and automatic runs
- **MLflow UI**: See experiments appearing automatically

**Dagster bonus:**
```bash
docker-compose --profile dagster up -d
# Open http://localhost:3000
# Enable schedule: Overview > Schedules > churn_training_schedule
# Watch runs appear every 2 minutes!
```

---

## Project Structure

```
+-- notebooks/                        # Jupyter notebooks
|   +-- 01_messy_notebook.ipynb       # Starting point - chaos
|   +-- 01b_mlflow_transition.ipynb   # Guided MLflow transition
|   +-- 02_mlflow_organized.ipynb     # Complete MLflow solution
|
+-- pipelines/
|   +-- workshop/                     # Learning materials
|   |   +-- prefect/
|   |   |   +-- Prefect_Workshop.py   # 6-part workshop (with automation!)
|   |   +-- dagster/
|   |   |   +-- Dagster_Workshop.py   # Asset-centric bonus (with schedules!)
|   |   +-- airflow/
|   |       +-- airflow_overview.md   # Pain points guide
|   |
|   +-- examples/                     # Reference implementations
|       +-- Prefect_ML_Pipeline.py
|       +-- Airflow_ML_Pipeline.py
|       +-- Dagster_ML_Pipeline.py
|
+-- docs/
|   +-- mlflow_cheatsheet.md          # Quick reference
|   +-- README_DOCKER.md              # Docker setup instructions
|   +-- mlflow/                       # Course PDFs
|   +-- airflow/                      # Course PDFs
|
+-- data/                             # Generated data
|   +-- customer_data.csv
|
+-- docker-compose.yml                # MLflow + Prefect + Dagster
+-- Dockerfile.prefect                # Prefect worker image
+-- Dockerfile.dagster                # Dagster base image (webserver + daemon)
+-- requirements.txt
+-- generate_sample_data.py
```

---

## Docker Services

| Service | Port | Purpose | Default |
|---------|------|---------|---------|
| **mlflow** | 5000 | Experiment tracking, model registry | Yes |
| **prefect-server** | 4200 | Flow monitoring, deployments, schedules | Yes |
| **prefect-worker** | - | Executes scheduled flows | Yes |
| **dagster-webserver** | 3000 | Asset graph UI (bonus) | `--profile dagster` |
| **dagster-daemon** | - | Executes scheduled jobs (bonus) | `--profile dagster` |

### Commands

```bash
# Main workshop (MLflow + Prefect)
docker-compose up -d

# With Dagster bonus
docker-compose --profile dagster up -d

# Stop all
docker-compose down

# Clean slate (remove data)
docker-compose down -v

# View logs
docker-compose logs -f prefect-worker
docker-compose logs -f mlflow
docker-compose logs -f dagster-daemon
```

See `docs/README_DOCKER.md` for detailed Docker instructions.

---

## Orchestrators Covered

| Tool | Coverage | Role |
|------|----------|------|
| **Prefect** | Main focus | Hands-on practice, Pythonic approach, real automation |
| **Dagster** | Bonus | Modern alternative, asset-centric paradigm |
| **Airflow** | Overview | Industry standard reference, shows complexity |

---

## Key Orchestration Patterns

The workshop teaches orchestration patterns through ML use cases:

| Pattern | ML Problem | Solution |
|---------|-----------|----------|
| **Retries** | API fails randomly | `@task(retries=3, retry_delay_seconds=60)` |
| **Exponential Backoff** | Rate limits | `retry_delay_seconds=[10, 30, 60]` |
| **Caching** | Expensive feature engineering | `cache_key_fn=task_input_hash` |
| **Parallelism** | Comparing multiple models | Run training tasks in parallel |
| **Parameters** | Hyperparameter tuning | `@flow` with typed arguments |
| **Schedules** | Daily retraining | `cron="0 6 * * *"` |

---

## The Three Orchestrators Compared

### Airflow - Operations-First, Task-Centric

**Philosophy:** "I define a graph of tasks to execute"

**Pain points for ML:**
- XCom limited to 48KB - must serialize DataFrames to disk
- Requires cleanup tasks for temp files
- Heavy infrastructure

### Prefect - Code-First, Task-Centric

**Philosophy:** "I write Python functions, Prefect handles orchestration"

**Why it's better for ML:**
```python
@task(retries=3)
def my_task(input_df: pd.DataFrame) -> pd.DataFrame:
    return result_df  # Just return it! No I/O!

@flow
def my_pipeline():
    data = load_data()
    result = my_task(data)  # Normal function call
```

### Dagster - Code-First, Asset-Centric

**Philosophy:** "I define the data assets I want to exist"

**The paradigm shift:**
```python
@asset
def customer_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Dagster infers dependency from parameter name!
    return processed_df
```

---

## Division of Responsibilities

| Concern | Orchestrator | MLflow |
|---------|--------------|--------|
| Retry on failure | X | |
| Cache computations | X | |
| Run tasks in parallel | X | |
| Schedule pipelines | X | |
| Log parameters | | X |
| Log metrics | | X |
| Store model artifacts | | X |
| Version models | | X |
| Serve models | | X |

**Key insight:** Orchestrators handle HOW your pipeline runs. MLflow handles WHAT gets tracked.

---

## Workshop Flow

### Part 1: MLflow Fundamentals (~60 min)
1. Start with messy notebook - identify pain points
2. Work through MLflow transition notebook
3. Learn tracking, models, registry, serving

### Part 2: Orchestration Patterns (~60 min)
1. Work through Prefect Workshop Parts 1-5
2. Learn retries, caching, parallelism, parameters
3. Build a complete ML pipeline with MLflow

### Part 3: Real Automation (~30 min)
1. Deploy a scheduled flow (Part 6)
2. Watch it run automatically
3. See experiments appear in MLflow

### Part 4: Dagster Bonus (~30 min)
1. Explore asset-centric paradigm
2. Enable schedules in UI
3. Compare with Prefect approach

---

## Student Takeaways

### Technical Understanding:
1. **MLflow Components:** Tracking, Models, Registry, Serving
2. **Orchestration patterns:** Retries, caching, parallelism, parameters, schedules
3. **Real automation:** Deploy flows, set schedules, watch them run
4. **Design trade-offs:** Task-centric vs asset-centric

### Practical Skills:
- Build production ML pipelines with MLflow tracking
- Implement orchestration patterns in Prefect
- Deploy and schedule automated pipelines
- Use Docker for ML infrastructure

### Resume/Interview Language:
> "I've implemented ML pipelines using MLflow for experiment tracking and model registry, with Prefect for orchestration and scheduling. I can deploy automated retraining pipelines and understand the trade-offs between different orchestration approaches."

---

## Additional Resources

- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **Prefect Documentation:** https://docs.prefect.io/
- **Dagster Documentation:** https://docs.dagster.io/
- **Airflow Documentation:** https://airflow.apache.org/docs/
