"""
Prompt generators for the Marimo LLM co-researcher system.

Each function returns a formatted prompt string for a specific LLM task.
All LLM responses are expected to be JSON only.
"""

import json
from typing import Any

from marimo_context_extractor import MarimoContextExtractor


def research_decomposer_prompt(
    question: str,
    csv_info: dict[str, Any],
) -> str:
    """
    Generate a prompt for decomposing a research question into multiple
    independent notebook branches that can run in parallel.

    Args:
        question: The research question from the user
        csv_info: Dictionary containing CSV metadata:
            - file_path: Path to the CSV file
            - columns: List of column names
            - dtypes: Dict mapping column names to their dtypes
            - row_count: Number of rows
            - sample_rows: First few rows as a string representation

    Returns:
        Formatted prompt string
    """
    file_path = csv_info.get("file_path", "data.csv")

    prompt = f"""You are a research assistant helping to analyze data. Your task is to decompose
a research question into multiple INDEPENDENT investigation branches that can run in parallel.

RESEARCH QUESTION:
{question}

AVAILABLE DATA:
File: {file_path}
Columns: {json.dumps(csv_info.get("columns", []))}
Data Types: {json.dumps(csv_info.get("dtypes", {}))}
Row Count: {csv_info.get("row_count", "unknown")}

Sample Data:
{csv_info.get("sample_rows", "No sample available")}

INSTRUCTIONS:
1. Break down the research question into 2-5 INDEPENDENT investigation branches
2. Each branch should be a self-contained analysis that can run without depending on other branches
3. Each branch will become a separate Marimo notebook
4. For each branch, provide the complete Python code cells needed

CRITICAL RULES FOR MARIMO CELLS:
1. Marimo cells are NOT functions - they are module-level code blocks
2. NEVER use 'return' statements - cells communicate via variable assignments
3. Each cell defines variables that downstream cells can reference
4. The FIRST cell of each branch MUST import required libraries (pandas, numpy, etc.)
5. The SECOND cell MUST load the data using: df = pd.read_csv('{file_path}')

Respond with ONLY valid JSON matching this schema:
{{
  "reasoning": "explanation of how you decomposed the question",
  "branches": [
    {{
      "name": "snake_case_branch_name",
      "description": "what this branch investigates",
      "cells": [
        {{
          "code": "import pandas as pd\\nimport numpy as np",
          "purpose": "Import libraries"
        }},
        {{
          "code": "df = pd.read_csv('{file_path}')",
          "purpose": "Load the data"
        }},
        {{
          "code": "# analysis code here",
          "purpose": "what this cell does"
        }}
      ]
    }}
  ]
}}

Do NOT include markdown formatting. Output ONLY the JSON object."""

    return prompt


def task_decomposer_prompt(
    task_description: str,
    dataset_info: str,
    extractor: MarimoContextExtractor,
) -> str:
    """
    Generate a prompt for decomposing a task into cells.

    The LLM should suggest new cells as code snippets. Marimo infers
    the dependency graph automatically from variable names.

    Args:
        task_description: What the user wants to accomplish
        dataset_info: Description of available data
        extractor: MarimoContextExtractor for current notebook state

    Returns:
        Formatted prompt string
    """
    context = extractor.get_full_context()

    # Format existing cells for context
    existing_cells = []
    for cell_id, cell_info in context["cells"].items():
        existing_cells.append(
            {
                "name": cell_info["name"],
                "code": cell_info["code"],
                "defines": cell_info["defines"],
                "references": cell_info["references"],
            }
        )

    prompt = f"""You are a data science expert helping design a Marimo notebook.

CRITICAL RULES FOR MARIMO CELLS:
1. Marimo cells are NOT functions - they are module-level code blocks
2. NEVER use 'return' statements - cells communicate via variable assignments
3. Each cell defines variables that downstream cells can reference
4. Marimo automatically infers dependencies from variable names
5. Do NOT redefine variables that are already defined by upstream cells

CURRENT NOTEBOOK STATE:
{json.dumps(existing_cells, indent=2)}

EXECUTION ORDER: {json.dumps(context["execution_order"])}

AVAILABLE DEFINITIONS: {json.dumps(context["definitions"])}

TASK DESCRIPTION:
{task_description}

DATASET INFORMATION:
{dataset_info}

Based on the task, suggest new cells to add to this notebook. For each cell:
- Write the actual Python code (no function wrappers)
- Ensure variable names don't conflict with existing definitions
- Reference existing variables where appropriate

Respond with ONLY valid JSON matching this schema:
{{
  "reasoning": "explanation of your approach",
  "cells": [
    {{
      "suggested_code": "the Python code for this cell",
      "purpose": "what this cell accomplishes",
      "expected_defines": ["list", "of", "variables", "this", "cell", "defines"],
      "expected_references": ["list", "of", "variables", "this", "cell", "uses"]
    }}
  ]
}}

Do NOT include markdown formatting. Output ONLY the JSON object."""

    return prompt


def cell_synthesizer_prompt(
    purpose: str,
    cell_name: str,
    extractor: MarimoContextExtractor,
    runtime_defs: dict[str, Any],
) -> str:
    """
    Generate a prompt for synthesizing a single cell's code.

    Uses runtime variable information to provide concrete types and values.

    Args:
        purpose: What this cell should accomplish
        cell_name: Suggested name for the cell
        extractor: MarimoContextExtractor for current notebook state
        runtime_defs: Runtime definitions from app.run()

    Returns:
        Formatted prompt string
    """
    # Get available variables with their types and values
    available_vars = extractor.get_available_variables(cell_name, runtime_defs)

    # Get full notebook context
    context = extractor.get_full_context()

    # Format variable info for the prompt
    var_descriptions = []
    for var_name, var_info in available_vars.items():
        desc = f"  - {var_name}: {var_info['type']}"
        if "shape" in var_info:
            desc += f", shape={var_info['shape']}"
        if "columns" in var_info:
            desc += f", columns={var_info['columns']}"
        if "value" in var_info:
            desc += f", value={var_info['value']}"
        if "sample" in var_info:
            desc += f"\n    sample: {var_info['sample']}"
        var_descriptions.append(desc)

    prompt = f"""You are a data science expert writing code for a Marimo notebook cell.

CRITICAL RULES FOR MARIMO CELLS:
1. Marimo cells are NOT functions - write module-level code only
2. NEVER use 'return' statements
3. Do NOT redefine variables from upstream cells
4. Variables you define become available to downstream cells
5. Use simple assignments, not function definitions (unless the function is what you're defining)

CELL PURPOSE:
{purpose}

CELL NAME: {cell_name}

AVAILABLE VARIABLES FROM UPSTREAM CELLS:
{chr(10).join(var_descriptions) if var_descriptions else "  (no upstream variables available)"}

EXISTING DEFINITIONS IN NOTEBOOK: {json.dumps(context["definitions"])}

Write Python code for this cell that accomplishes the stated purpose.

Respond with ONLY valid JSON matching this schema:
{{
  "code": "the Python code for this cell (multiline string)",
  "explanation": "brief explanation of what the code does",
  "defines": ["list", "of", "variables", "defined"],
  "references": ["list", "of", "variables", "referenced"]
}}

Do NOT include markdown formatting. Output ONLY the JSON object."""

    return prompt


def error_recovery_prompt(
    failed_cell_name: str,
    extractor: MarimoContextExtractor,
    runtime_defs: dict[str, Any],
    error: Exception,
) -> str:
    """
    Generate a prompt for recovering from a cell execution error.

    Shows concrete type information to help identify mismatches.

    Args:
        failed_cell_name: Name of the cell that failed
        extractor: MarimoContextExtractor for current notebook state
        runtime_defs: Runtime definitions from app.run() before failure
        error: The exception that was raised

    Returns:
        Formatted prompt string
    """
    # Get the failed cell's code
    cell_code = extractor.get_cell_code(failed_cell_name)

    # Get cell context including ancestors
    cell_context = extractor.get_cell_context(failed_cell_name)

    # Get available variables with types
    available_vars = extractor.get_available_variables(failed_cell_name, runtime_defs)

    # Format variable info
    var_descriptions = []
    for var_name, var_info in available_vars.items():
        desc = f"  - {var_name}: {var_info['type']}"
        if "shape" in var_info:
            desc += f", shape={var_info['shape']}"
        if "dtype" in var_info:
            desc += f", dtype={var_info['dtype']}"
        if "columns" in var_info:
            desc += f", columns={var_info['columns']}"
        var_descriptions.append(desc)

    # Format ancestor chain
    ancestor_info = []
    for anc in cell_context.get("ancestors", []):
        ancestor_info.append(f"  {anc['name']}: defines {anc['defines']}")

    prompt = f"""You are debugging a Marimo notebook cell that raised an error.

FAILED CELL: {failed_cell_name}

CELL CODE:
```python
{cell_code}
```

ERROR TYPE: {type(error).__name__}
ERROR MESSAGE: {str(error)}

AVAILABLE VARIABLES (from upstream cells):
{chr(10).join(var_descriptions) if var_descriptions else "  (no variables available)"}

ANCESTOR CELLS:
{chr(10).join(ancestor_info) if ancestor_info else "  (no ancestors)"}

CRITICAL RULES FOR THE FIX:
1. Marimo cells are NOT functions - no 'return' statements
2. Do NOT redefine variables from upstream cells
3. The fix should produce the same output variables as the original cell intended

Analyze the error and provide fixed code.

Respond with ONLY valid JSON matching this schema:
{{
  "fixed_code": "the corrected Python code",
  "bug_type": "TypeError|ValueError|AttributeError|etc",
  "fix_explanation": "what was wrong and how it was fixed"
}}

Do NOT include markdown formatting. Output ONLY the JSON object."""

    return prompt


def verification_spec_prompt(
    cell_name: str,
    extractor: MarimoContextExtractor,
    runtime_defs: dict[str, Any],
) -> str:
    """
    Generate a prompt for creating verification assertions for a cell.

    Looks at downstream usage to determine appropriate assertions.

    Args:
        cell_name: Name of the cell to verify
        extractor: MarimoContextExtractor for current notebook state
        runtime_defs: Runtime definitions from app.run()

    Returns:
        Formatted prompt string
    """
    # Get what this cell defines
    cell_defines = extractor.get_cell_defines(cell_name)

    # Get downstream cells to see how outputs are used
    downstream = extractor.get_downstream_cells(cell_name)

    # Get downstream cell code to analyze usage patterns
    downstream_usage = []
    for ds_name in downstream:
        ds_code = extractor.get_cell_code(ds_name)
        if ds_code:
            downstream_usage.append(
                {
                    "cell": ds_name,
                    "code": ds_code,
                    "references": extractor.get_cell_references(ds_name),
                }
            )

    # Get actual types of defined variables
    var_types = {}
    for var_name in cell_defines:
        if var_name in runtime_defs:
            var_info = extractor._describe_variable(var_name, runtime_defs[var_name])
            var_types[var_name] = var_info

    prompt = f"""You are creating verification assertions for a Marimo notebook cell.

CELL NAME: {cell_name}

VARIABLES DEFINED BY THIS CELL: {cell_defines}

ACTUAL TYPES AND VALUES OF OUTPUTS:
{json.dumps(var_types, indent=2)}

DOWNSTREAM CELLS THAT USE THESE OUTPUTS:
{json.dumps(downstream_usage, indent=2)}

Based on how downstream cells use the outputs from this cell, generate assertions
that verify the outputs meet the expected contract.

For example:
- If downstream calls .shape, assert the output has the expected shape
- If downstream indexes with ['column'], assert the column exists
- If downstream uses arithmetic, assert the output is numeric

Respond with ONLY valid JSON matching this schema:
{{
  "reasoning": "explanation of why these assertions are appropriate",
  "assertions": [
    "assert isinstance(var, expected_type)",
    "assert var.shape[0] == expected_value"
  ]
}}

The assertions should be valid Python code that can be exec()'d.
Do NOT include markdown formatting. Output ONLY the JSON object."""

    return prompt


def backtrack_decision_prompt(
    failed_cell_name: str,
    attempt_count: int,
    extractor: MarimoContextExtractor,
    runtime_defs: dict[str, Any],
) -> str:
    """
    Generate a prompt for deciding whether to backtrack to an upstream cell.

    Used after multiple failed attempts to fix a cell.

    Args:
        failed_cell_name: Name of the cell that keeps failing
        attempt_count: How many fix attempts have been made
        extractor: MarimoContextExtractor for current notebook state
        runtime_defs: Runtime definitions from app.run()

    Returns:
        Formatted prompt string
    """
    # Get cell context with ancestor chain
    cell_context = extractor.get_cell_context(failed_cell_name)

    # Get impact analysis
    impact = extractor.get_impact_of_change(failed_cell_name)

    # Get available variables with types
    available_vars = extractor.get_available_variables(failed_cell_name, runtime_defs)

    # Format ancestor info with their code
    ancestor_details = []
    for anc in cell_context.get("ancestors", []):
        ancestor_details.append(
            {
                "name": anc["name"],
                "code": anc["code"],
                "defines": anc["defines"],
            }
        )

    # Format variable types
    var_types = {}
    for var_name, var_info in available_vars.items():
        var_types[var_name] = {
            "type": var_info["type"],
            "shape": var_info.get("shape"),
            "dtype": var_info.get("dtype"),
        }

    prompt = f"""You are deciding whether to backtrack and regenerate an upstream cell.

FAILED CELL: {failed_cell_name}
FAILED CODE:
```python
{cell_context.get("code", "N/A")}
```

FIX ATTEMPTS SO FAR: {attempt_count}

CELL REFERENCES: {cell_context.get("references", [])}

AVAILABLE VARIABLES AND THEIR TYPES:
{json.dumps(var_types, indent=2)}

ANCESTOR CELLS (potential backtrack targets):
{json.dumps(ancestor_details, indent=2)}

IF WE CHANGE {failed_cell_name}, IMPACT:
- Directly affected: {impact.get("directly_affected", [])}
- Transitively affected: {impact.get("transitively_affected", [])}

Consider:
1. Is the error likely caused by incorrect types/shapes from an upstream cell?
2. Would regenerating an upstream cell fix the root cause?
3. What is the blast radius of changing each potential target?

Respond with ONLY valid JSON matching this schema:
{{
  "reasoning": "explanation of your decision",
  "cell_to_regenerate": "cell_name_to_regenerate_or_null_to_retry_current",
  "confidence": 0.0 to 1.0
}}

Set cell_to_regenerate to null if you think retrying the current cell with different code is better.
Do NOT include markdown formatting. Output ONLY the JSON object."""

    return prompt
