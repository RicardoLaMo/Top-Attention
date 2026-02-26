---
name: github-code-researcher
description: Search GitHub and PyPI for existing code implementations from papers.
---

# GitHub Code Researcher

You are equipped with the capability to search for existing code to solve technical challenges rather than re-implementing them from scratch. This is particularly useful when you need to use an algorithm from a recent paper or a complex, established piece of software.

## Instructions

When the PI assigns a task that involves a known algorithm or paper:

1.  **Search First:** Use the `search_web` tool to look for open-source repositories (e.g., query "GitHub [Algorithm Name] implementation" or "PyPI [Algorithm Name]").
2.  **Evaluate:** Review the found repositories. Look for recent commits, clear documentation, and a structure that is easy to integrate.
3.  **Clone/Install:**
    -   If it's a PyPI package, add it to `requirements.txt` and install it.
    -   If it's a GitHub repository that needs adaptation, clone the relevant files into the project's `vendor/` or `references/` directory. You can use the `run_command` tool with `git clone`.
4.  **Adapt:** Write integration code in the `src/` directory to interface the external codebase with the project's data loader and evaluation pipeline.

## Example Workflow

1.  **PI Task:** "Implement the Graph Attention Network (GAT) from Velickovic et al. 2018."
2.  **Action:** Search for PyTorch Geometric (PyG) or Deep Graph Library (DGL) examples of GAT, rather than writing the GAT layer manually.
3.  **Integration:** Create `src/models/gat.py` that wraps the PyG `GATConv` layer.
