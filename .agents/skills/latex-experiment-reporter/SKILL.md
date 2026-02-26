---
name: latex-experiment-reporter
description: Auto-generate academic LaTeX summaries for completed experiments to be accumulated into the final paper.
---

# LaTeX Experiment Reporter

You are equipped to generate publication-ready LaTeX fragments summarizing the intention, execution, results, and implications of experimental runs.

## Instructions

When the Orchestrator or PI requests a formal report for an experiment (typically when a Challenger beats a Champion or a significant finding is made):

1.  **Gather Context:** 
    -   Read the PRD (`docs/prd/*.md`) to understand the goal.
    -   Read the `artifacts/experiment_log.jsonl` to get the git hash, parameters, metrics, and `is_champion` status.
2.  **Generate Report:** Create a `.tex` file in `reports/experiments/` named appropriately (e.g., `reports/experiments/RR-102_edge_weights.tex`). 
3.  **Format Constraints:** The file MUST be a raw LaTeX fragment that can be compiled when included in a main document via `\input{...}`. Do NOT include `\documentclass` or `\begin{document}`.

## Report Structure

The generated LaTeX file must follow this exact structure using subsection headers:

```latex
\subsection{Experiment: [Short Name]}
\label{sec:exp_[short_name]}

\textbf{Intention \& Methodology:} 
% Summarize the goal from the PRD. What hypothesis were we testing? What is the mathematical or architectural change?

\textbf{Implementation Details:}
% State the Git hash used for reproducibility, the key hyperparameters, and the branch name.

\textbf{Results (Champion vs Challenger):}
% Present the results clearly, preferably in a small table. Explicitly compare this run against the previous Champion for this category. Did it win or lose? By what margin?

\textbf{Discussion \& Literature:}
% Why do we think these results occurred? Cite relevant literature using \cite{...} that supports or contrasts with our findings.
```

## Bibliography Management (via Zotero)

When writing the Discussion section and citing papers:
1.  **Search Zotero** for relevant papers: `zotero_search_items(query="...", tag="TopAttention")`
2.  **Get metadata**: `zotero_item_metadata(item_key="...")` to extract title, authors, DOI, year.
3.  **Format as BibTeX**: Use key convention `{first_author}_{keyword}_{year}` (e.g., `vaswani_attention_2017`).
4.  **Append** the BibTeX entry to the project `.bib` file.
5.  **Cite** in the `.tex` fragment using `\cite{key}` — ensure keys match exactly.

## PR-Triggered Auto-Update

When reviewing a PR (via the `github-pr-reviewer` skill), this reporter is automatically triggered if the PR modifies ANY of:
- `artifacts/experiment_log.jsonl`
- Any file in `reports/experiments/`
- PR title/description contains keywords: "experiment", "results", "paper", "agenda"

**Auto-Update Workflow:**
1. Check out the PR branch: `gh pr checkout <number>`
2. Read the new entries in `artifacts/experiment_log.jsonl` to identify what changed.
3. Generate or update the corresponding `.tex` fragment in `reports/experiments/`.
4. Commit and push the `.tex` file to the same PR branch:
   ```bash
   git add reports/experiments/<name>.tex
   git commit -m "docs: auto-generate LaTeX report for <experiment>"
   git push
   ```
5. Post a PR comment confirming the LaTeX update was made.
