# Feature: Basic NLP TF-IDF Baseline

## Category: Baseline
## Parent Idea: N/A (New category)
## Git Branch: `feat/baseline-drill`

## Intent
We need a simple baseline to understand the performance of standard NLP techniques before moving to graphs. The plan is to compute TF-IDF features on the text and evaluate ROC-AUC using a Logistic Regression model.

## Methodology
- Extract text features.
- Compute TF-IDF (max_features=1000).
- Train `sklearn.linear_model.LogisticRegression`.
- Evaluate ROC-AUC and PR-AUC.
