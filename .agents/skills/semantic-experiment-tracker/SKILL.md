---
name: semantic-experiment-tracker
description: Log and search experiments by idea/description instead of ticket number.
---

# Semantic Experiment Tracker

You are equipped to manage the project's experiments based on the *semantic ideas* behind them, preventing the loss of knowledge and avoiding the repetition of failed experiments.

## Instructions

Whenever an experiment finishes, you must log it. When the PI asks about past results, you must retrieve them based on the text descriptions.

### 1. Logging an Experiment

When an experiment finishes, you MUST log it. 

Create or append to `artifacts/experiment_log.jsonl` (JSON Lines format). Record a JSON object with this schema:

```json
{
  "timestamp": "2023-10-27T10:00:00Z",
  "category": "Edge Weighting", 
  "parent_idea": "RR-102" ,
  "prd_link": "docs/prd/edge_weights_v2.md",
  "git_hash": "a1b2c3d...",
  "idea": "Added dynamic attention weighting to the node features to see if it improves AUC.",
  "parameters": {
    "learning_rate": 0.001,
    "model_type": "DynamicGAT"
  },
  "results": {
    "roc_auc": 0.92,
    "pr_auc": 0.85
  },
  "is_champion": false
}
```

**Crucial Fields:**
-   `category`: High-level group this experiment belongs to.
-   `parent_idea`: The run ID or hash of the experiment this iterates on (for lineage). If it is completely new, leave null.
-   `prd_link`: Path to the markdown PRD describing the goal.
-   `is_champion`: Set to `true` if this run beat the previous best run in this category during the Orchestrator's Champion vs Challenger phase. If so, you must find the previous champion and update its `is_champion` status to `false`.

### 2. Retrieving Experiments

When asked to find past experiments (e.g., "Find the champion for the Edge Weighting category"):

1.  Read `artifacts/experiment_log.jsonl`.
2.  Filter by `category` and check the `is_champion` flag to find benchmarks. Alternatively, fuzzy match the `idea` fields.
3.  Format retrieved runs into a clear Markdown table, displaying `category`, `idea`, key metrics, and `is_champion` status.

**Example:**

| Idea | Hyperparams | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- |
| Baseline GCN | lr=0.01 | 0.85 | 0.70 |
| Added edge weights | lr=0.01 | 0.87 | 0.73 |
