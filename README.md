# Co-Researcher

A parallel research assistant that uses LLM-generated marimo notebooks to investigate data from multiple angles simultaneously.

## Overview

Co-Researcher takes a research question and a CSV dataset, then:

1. **Decomposes** the question into independent research branches using an LLM
2. **Generates** marimo notebook cells for each branch
3. **Executes** branches in parallel
4. **Produces** multiple notebooks with different analysis approaches

## Project Structure

```
co-researcher/
├── main.py                    # Entry point
├── cli.py                     # Interactive command-line interface
├── llm_client.py              # Azure Anthropic Foundry client
├── research_orchestrator.py   # Coordinates parallel research branches
├── orchestrator.py            # Marimo notebook generation and execution
├── prompts.py                 # LLM prompt templates
├── marimo_context_extractor.py # Extracts context from marimo notebooks
├── tests/
│   └── test_mcp_client.py     # MCP client tests
├── demo_notebook.py           # Example marimo notebook
└── sample_data.csv            # Sample dataset
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd co-researcher

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

Copy the example environment file and fill in your Azure Anthropic credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
AZURE_ANTHROPIC_ENDPOINT=https://your-resource.services.ai.azure.com/anthropic/
AZURE_ANTHROPIC_API_KEY=your-api-key-here
AZURE_ANTHROPIC_MODEL=claude-opus-4-5
```

## Usage

### Interactive CLI

```bash
python main.py
```

This launches an interactive session where you:
1. Select CSV files from the current directory
2. Enter a research question
3. Review the generated research plan
4. Execute parallel research branches
5. Get generated marimo notebooks in the `output/` directory

### Example Session

```
══════════════════════════════════════════════════════════════════════
Research Plan - 3 Branches
══════════════════════════════════════════════════════════════════════

[1] correlation_analysis
    Description: Analyze correlations between numeric columns
    Cells: 4 steps
      1. Load data and compute correlation matrix
      2. Visualize correlations with heatmap
      3. Identify strong correlations
      ... and 1 more steps

[2] distribution_analysis
    Description: Examine statistical distributions
    Cells: 3 steps
      ...

Ready to execute 3 research branches.
Press Enter to continue or 'q' to quit:
```

### Opening Generated Notebooks

```bash
marimo edit output/correlation_analysis_20240222_143200.py
```

## Architecture

### LLM Client (`llm_client.py`)

Handles communication with Azure Anthropic Foundry. Provides JSON parsing with automatic markdown code block handling.

### Research Orchestrator (`research_orchestrator.py`)

- Analyzes CSV metadata (columns, types, sample rows)
- Calls LLM to decompose research questions into branches
- Executes branches in parallel using ThreadPoolExecutor
- Tracks progress and handles errors

### Marimo Orchestrator (`orchestrator.py`)

- Creates and modifies marimo notebooks programmatically
- Handles cell synthesis, error recovery, and verification
- Supports backtracking when cells fail

### Prompts (`prompts.py`)

Contains all LLM prompt templates for:
- Research question decomposition
- Task decomposition into cells
- Cell code synthesis
- Error recovery
- Verification assertions
- Backtrack decisions

## Development

### Running Tests

```bash
pytest tests/ -v
```

Note: MCP tests require a running marimo server with `--mcp` flag.

### Code Style

The project uses Python 3.13+ features including:
- Type hints with `|` union syntax
- Dataclasses
- f-strings

## License

MIT
