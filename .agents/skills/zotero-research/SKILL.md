---
name: zotero-research
description: Search, retrieve, and cite papers from the Zotero library for the TopAttention project. Integrates with PI, MLE, and LaTeX reporter agents.
---

# Zotero Research Skill (TopAttention Project)

You are equipped to search, read, and cite academic papers from the Zotero library. This is the primary literature management tool for the project.

## Project Collection

All papers relevant to this project should be tagged with **`TopAttention`** in Zotero. When adding new papers, always apply this tag.

## Available Tools

You have access to these Zotero MCP tools:

| Tool | Purpose |
|---|---|
| `zotero_search_items` | Search by title, author, year, or full text |
| `zotero_item_metadata` | Get metadata (title, authors, DOI, URL, tags) |
| `zotero_item_fulltext` | Get the full text content of a paper |

## Common Workflows

### 1. Search for Papers on a Topic

```
# Search broadly
zotero_search_items(query="attention mechanism", qmode="everything", limit=10)

# Search by author
zotero_search_items(query="Vaswani 2017", qmode="titleCreatorYear", limit=10)

# Search within project papers only
zotero_search_items(query="attention", tag="TopAttention", limit=10)
```

### 2. Read a Paper's Content

After finding a paper's key from a search result:
```
# Get metadata first
zotero_item_metadata(item_key="ABC123")

# Then read the full text
zotero_item_fulltext(item_key="ABC123")
```

### 3. Extract BibTeX-Style Citation

After retrieving metadata, format a BibTeX entry for the project `.bib` file:

```bibtex
@article{lastname_keyword_year,
  title  = {Paper Title},
  author = {Last, First and Last2, First2},
  journal = {Journal Name},
  year   = {2024},
  doi    = {10.xxxx/...},
}
```

**Key naming convention:** `{first_author_lastname}_{one_keyword}_{year}`
Example: `vaswani_attention_2017`

### 4. Add a Paper by DOI (via Code Execution)

When you find a new relevant paper, add it to Zotero and tag it for the project:

```python
import sys
sys.path.append('/Volumes/MacMini/.gemini/antigravity/skills/zotero-mcp-code/scripts')
import setup_paths
from zotero_lib import SearchOrchestrator

orchestrator = SearchOrchestrator()
item_key = orchestrator.create_item_from_doi("10.xxxx/paper-doi")
print(f"Added paper with key: {item_key}")
# Note: manually add 'TopAttention' tag in Zotero after adding
```

### 5. Comprehensive Literature Search (Code Execution)

For deep, multi-strategy searches:

```python
import sys
sys.path.append('/Volumes/MacMini/.gemini/antigravity/skills/zotero-mcp-code/scripts')
import setup_paths
from zotero_lib import SearchOrchestrator, format_results

orchestrator = SearchOrchestrator()
results = orchestrator.comprehensive_search(
    "graph attention networks fraud detection",
    max_results=20
)
print(format_results(results, include_abstracts=True))
```

## Agent-Specific Instructions

### For the PI Agent
- **Before proposing a new idea:** Search Zotero for related work. Use `comprehensive_search()` with the idea keywords.
- **When writing paper sections:** Use `zotero_item_fulltext()` to read cited papers and summarize their methods/results.
- **When discovering new papers:** Add them via DOI and tag with `TopAttention`.

### For the MLE Agent
- **Before implementing an algorithm:** Search Zotero for the original paper, read its methodology section via `zotero_item_fulltext()`, then search GitHub for implementations (via `github-code-researcher`).

### For the LaTeX Reporter
- **When writing the Discussion section:** Search Zotero for papers related to the experiment's methodology, extract their BibTeX-style metadata, and add `\cite{key}` references.
- **Add new BibTeX entries** to the project `.bib` file when citing papers not yet in the bibliography.
