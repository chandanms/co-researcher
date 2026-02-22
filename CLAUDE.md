# CLAUDE.md

Project context for AI assistants working on this codebase.

## Project Overview

Co-Researcher is a parallel research assistant that generates marimo notebooks from natural language research questions. It uses Azure Anthropic Foundry for LLM calls.

## Key Architecture Decisions

- **LLM Client**: All LLM calls go through `llm_client.py` which wraps Azure Anthropic Foundry. Never use `anthropic.Anthropic` directly elsewhere.
- **Marimo Integration**: Uses marimo's internal APIs (`marimo._ast.app`, `marimo._ast.load`, `marimo._schemas.serialization`) for programmatic notebook manipulation.
- **Parallel Execution**: Research branches run in parallel via `ThreadPoolExecutor` in `research_orchestrator.py`.

## File Responsibilities

| File | Purpose |
|------|---------|
| `llm_client.py` | Azure Anthropic Foundry client, JSON response parsing |
| `orchestrator.py` | Single notebook generation, cell synthesis, error recovery |
| `research_orchestrator.py` | Multi-branch coordination, parallel execution |
| `prompts.py` | All LLM prompt templates |
| `cli.py` | Terminal UI with ANSI colors |
| `marimo_context_extractor.py` | Extracts notebook state for LLM context |

## Common Patterns

### LLM Calls
```python
from llm_client import LLMClient

llm = LLMClient()  # Loads config from .env
result = llm.call(prompt)  # Returns dict, check for "error" key
if "error" in result:
    # Handle error
```

### Adding Notebook Cells
```python
from orchestrator import add_cell, create_empty_notebook

app = create_empty_notebook()
app = add_cell(app, "import pandas as pd", name="imports")
```

## Environment Variables

Required in `.env`:
- `AZURE_ANTHROPIC_ENDPOINT` - Azure endpoint URL
- `AZURE_ANTHROPIC_API_KEY` - API key
- `AZURE_ANTHROPIC_MODEL` - Model/deployment name (default: `claude-opus-4-5`)

## Testing

MCP client tests require a running marimo server:
```bash
marimo edit --mcp some_notebook.py
pytest tests/ -v
```

## Code Style

- Python 3.13+ (uses `|` union types, not `Optional`)
- Type hints on function signatures
- Dataclasses for structured data
- f-strings for formatting

## Things to Avoid

- Don't instantiate `Anthropic` or `AnthropicFoundry` outside `llm_client.py`
- Don't use `json.loads` on LLM responses directly - use `LLMClient.call()` which handles markdown code blocks
- Don't modify marimo notebooks with string manipulation - use the `add_cell`/`create_empty_notebook` helpers
