# Docker Setup for MLflow + Orchestrators Workshop

## Architecture Overview

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

YOUR MACHINE:
  - VS Code with notebooks (learning)
  - Deploy flows via CLI
  - Watch automation in UIs
```

**Key Point:** Everything runs in Docker. You deploy flows and watch them run automatically. This is REAL orchestration.

---

## Quick Start

### Main Workshop: MLflow + Prefect

```bash
# Start MLflow + Prefect (server + worker)
docker-compose up -d

# Verify services are running
docker-compose ps

# Access UIs
# Prefect: http://localhost:4200
# MLflow:  http://localhost:5000
```

### Bonus: Add Dagster

```bash
# Start everything including Dagster
docker-compose --profile dagster up -d

# Access Dagster UI
# http://localhost:3000
```

---

## Service Details

| Service | Port | Purpose |
|---------|------|---------|
| **mlflow** | 5000 | Experiment tracking, model registry |
| **prefect-server** | 4200 | Flow monitoring, deployments, schedules |
| **prefect-worker** | - | Executes scheduled flows |
| **dagster-webserver** (bonus) | 3000 | Asset graph UI, materializations |
| **dagster-daemon** (bonus) | - | Executes scheduled jobs |

---

## Workshop Workflow

### 1. Start the Stack

```bash
docker-compose up -d
```

Wait for services to be healthy:
```bash
docker-compose ps
# Look for "healthy" status on mlflow and prefect-server
```

### 2. Learn the Patterns (Parts 1-5)

Run workshop parts locally to learn orchestration patterns:

```bash
# Install dependencies
pip install -r requirements.txt

# Run workshop parts
python pipelines/workshop/prefect/Prefect_Workshop.py part1  # Tasks & Flows
python pipelines/workshop/prefect/Prefect_Workshop.py part2  # Retries
python pipelines/workshop/prefect/Prefect_Workshop.py part3  # Caching & Parallelism
python pipelines/workshop/prefect/Prefect_Workshop.py part4  # Parameters
python pipelines/workshop/prefect/Prefect_Workshop.py part5  # Full Pipeline + MLflow
```

### 3. See Real Automation (Part 6)

Deploy a scheduled flow:

```bash
python pipelines/workshop/prefect/Prefect_Workshop.py deploy
```

This will:
1. Register the flow with Prefect server
2. Create a schedule (every 2 minutes)
3. Keep running until you press Ctrl+C

Now watch:
- **Prefect UI** (http://localhost:4200): See deployments and runs
- **MLflow UI** (http://localhost:5000): See experiments appearing automatically

### 4. Bonus: Dagster

```bash
# Start Dagster
docker-compose --profile dagster up -d
```

Open http://localhost:3000:
1. **Asset Graph**: See the data lineage
2. **Materialize**: Click to run assets
3. **Schedules**: Enable `churn_training_schedule`
4. **Watch**: Runs appear every 2 minutes!

---

## Common Commands

```bash
# Start main workshop (MLflow + Prefect)
docker-compose up -d

# Start with Dagster bonus
docker-compose --profile dagster up -d

# Stop all services
docker-compose down

# Stop and remove data (clean slate)
docker-compose down -v

# View logs
docker-compose logs -f mlflow
docker-compose logs -f prefect-server
docker-compose logs -f prefect-worker
docker-compose logs -f dagster-webserver
docker-compose logs -f dagster-daemon

# Rebuild images after Dockerfile changes
docker-compose build prefect-worker
docker-compose --profile dagster build

# Restart a service
docker-compose restart prefect-worker
```

---

## Troubleshooting

### Services not starting

```bash
# Check status
docker-compose ps

# Check logs for errors
docker-compose logs mlflow
docker-compose logs prefect-server
docker-compose logs prefect-worker
```

### Port already in use

Create a `.env` file:

```bash
MLFLOW_PORT=5001
PREFECT_PORT=4201
DAGSTER_PORT=3001
```

### Prefect worker not executing flows

```bash
# Check worker logs
docker-compose logs -f prefect-worker

# Restart worker
docker-compose restart prefect-worker
```

### Dagster schedules not running

Ensure the daemon is running:
```bash
docker-compose logs -f dagster-daemon
```

### MLflow not accessible from containers

The containers use Docker networking. Inside containers:
- MLflow: `http://mlflow:5000`
- Prefect: `http://prefect-server:4200/api`

From your machine:
- MLflow: `http://localhost:5000`
- Prefect: `http://localhost:4200`

---

## Data Persistence

Data is stored in Docker volumes:

| Volume | Contains |
|--------|----------|
| `workshop-mlflow-data` | Experiments, runs, models |
| `workshop-prefect-data` | Flow run history, deployments |
| `workshop-dagster-data` | Asset materializations |

To reset all data:
```bash
docker-compose down -v
```

---

## For Instructors

### Before Workshop

```bash
# Clean state
docker-compose down -v

# Pull and build images
docker-compose pull
docker-compose build
docker-compose --profile dagster build

# Start and verify
docker-compose up -d
docker-compose ps  # All should be healthy
```

### During Workshop

Students run:
```bash
docker-compose up -d
pip install -r requirements.txt
```

Then follow the workshop flow:
1. Parts 1-5: Learn patterns locally
2. Part 6: Deploy and watch automation
3. Bonus: Explore Dagster

### Key Teaching Points

1. **Why Docker?** Real orchestration needs infrastructure (servers, workers, daemons)
2. **Prefect architecture**: Server (API + UI) + Worker (executes flows)
3. **Dagster architecture**: Webserver (UI) + Daemon (executes schedules)
4. **MLflow integration**: Both orchestrators log to the same MLflow server
5. **Automation**: Flows run on schedule without manual intervention
