"""
Tests for direct marimo notebook API without MCP.
Tests creating notebooks, executing cells, reactivity, and dependency graphs.

Run with: pytest tests/test_notebook_api.py -v
"""

from pathlib import Path
from typing import Any

import marimo
import pandas as pd
import pytest
from marimo._ast.app import App, InternalApp
from marimo._ast.cell import CellConfig
from marimo._ast.load import load_app, load_notebook_ir
from marimo._schemas.serialization import (
    AppInstantiation,
    CellDef,
    NotebookSerializationV1,
)

# =============================================================================
# Helper Functions for Notebook Management
# =============================================================================


def create_notebook(
    cells: list[dict[str, Any]],
    app_config: dict[str, Any] | None = None,
) -> App:
    """
    Create a marimo App from a list of cell definitions using native marimo IR.

    Args:
        cells: List of dicts with 'code', 'name' (optional), 'options' (optional)
        app_config: Optional app configuration dict

    Returns:
        A marimo App object
    """
    cell_defs = []
    for i, cell in enumerate(cells):
        cell_defs.append(
            CellDef(
                code=cell.get("code", ""),
                name=cell.get("name", f"cell_{i}"),
                options=cell.get("options", {}),
            )
        )

    notebook_ir = NotebookSerializationV1(
        app=AppInstantiation(options=app_config or {}),
        cells=cell_defs,
    )

    return load_notebook_ir(notebook_ir)


def save_notebook(app: App, filepath: str | Path) -> Path:
    """
    Save a marimo App to a file.

    Args:
        app: The marimo App object
        filepath: Path to write the notebook file

    Returns:
        Path to the saved notebook file
    """
    filepath = Path(filepath)
    internal = InternalApp(app)
    content = internal.to_py()
    filepath.write_text(content)
    return filepath


def load_notebook(filepath: str | Path) -> App:
    """
    Load a marimo notebook from a file path using native marimo loader.

    Args:
        filepath: Path to the notebook file

    Returns:
        The marimo App object

    Raises:
        MarimoFileError: If the file is not a valid marimo notebook
        FileNotFoundError: If the file doesn't exist
    """
    app = load_app(filepath)
    if app is None:
        raise ValueError(f"Could not load notebook from {filepath}")
    return app


def add_cell(
    app: App, code: str, name: str | None = None, options: dict | None = None
) -> App:
    """
    Add a cell to an existing notebook and return a new App.

    Since marimo Apps are somewhat immutable after initialization,
    this creates a new notebook IR with the additional cell.

    Args:
        app: The existing marimo App
        code: The cell code
        name: Optional cell name
        options: Optional cell options (disabled, hide_code, etc.)

    Returns:
        A new App with the cell added
    """
    internal = InternalApp(app)

    # Get existing cells from IR
    existing_ir = internal.to_ir()

    # Add new cell
    new_cell = CellDef(
        code=code,
        name=name or f"cell_{len(existing_ir.cells)}",
        options=options or {},
    )

    new_cells = list(existing_ir.cells) + [new_cell]

    # Create new notebook IR
    new_ir = NotebookSerializationV1(
        app=existing_ir.app,
        header=existing_ir.header,
        cells=new_cells,
    )

    return load_notebook_ir(new_ir)


def get_dependency_graph(app: App) -> dict[str, Any]:
    """
    Extract the dependency graph from a marimo app.

    Args:
        app: The marimo App object

    Returns:
        Dict containing graph information:
        - cells: Dict mapping cell_id to cell info (code, defs, refs, parents, children)
        - definitions: Dict mapping variable names to defining cell_ids
        - execution_order: List of cell_ids in topological order
        - has_cycles: Boolean indicating if cycles exist
    """
    internal = InternalApp(app)
    graph = internal.graph

    info = {
        "cells": {},
        "definitions": {
            name: [str(cid) for cid in cells]
            for name, cells in graph.definitions.items()
        },
        "execution_order": [str(cid) for cid in internal.execution_order],
        "has_cycles": bool(graph.cycles),
        "cycles": list(graph.cycles) if graph.cycles else [],
    }

    for cell_id, cell in graph.cells.items():
        cell_data = internal.cell_manager.get_cell_data(cell_id)
        info["cells"][str(cell_id)] = {
            "name": cell_data.name if cell_data else None,
            "code": cell.code,
            "defs": list(cell.defs),
            "refs": list(cell.refs),
            "parents": [str(p) for p in graph.parents[cell_id]],
            "children": [str(c) for c in graph.children[cell_id]],
            "stale": cell.stale,
            "disabled": cell.config.disabled,
        }

    return info


def run_notebook(
    app: App,
    overrides: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Run a marimo notebook and return outputs and definitions.

    Args:
        app: The marimo App object
        overrides: Optional dict of variable overrides

    Returns:
        Tuple of (outputs, definitions)
    """
    outputs, defs = app.run(defs=overrides or {})
    return list(outputs), dict(defs)


def get_cell_by_name(app: App, name: str) -> dict[str, Any] | None:
    """
    Get cell information by its function name.

    Args:
        app: The marimo App object
        name: The cell's function name

    Returns:
        Cell info dict or None if not found
    """
    internal = InternalApp(app)
    cell_data = internal.cell_manager.get_cell_data_by_name(name)

    if cell_data is None:
        return None

    graph = internal.graph
    cell_id = cell_data.cell_id
    cell = graph.cells.get(cell_id)

    if cell is None:
        return None

    return {
        "cell_id": str(cell_id),
        "code": cell.code,
        "defs": list(cell.defs),
        "refs": list(cell.refs),
        "name": name,
    }


def get_downstream_cells(app: App, cell_name: str) -> list[str]:
    """
    Get all cells that depend on a given cell (descendants).

    Args:
        app: The marimo App object
        cell_name: Name of the cell

    Returns:
        List of cell names that are downstream
    """
    internal = InternalApp(app)
    graph = internal.graph

    cell_data = internal.cell_manager.get_cell_data_by_name(cell_name)
    if cell_data is None:
        return []

    target_cell_id = cell_data.cell_id
    descendants = graph.descendants(target_cell_id)

    result = []
    for cid in descendants:
        cd = internal.cell_manager.get_cell_data(cid)
        result.append(cd.name if cd else str(cid))

    return result


def get_upstream_cells(app: App, cell_name: str) -> list[str]:
    """
    Get all cells that a given cell depends on (ancestors).

    Args:
        app: The marimo App object
        cell_name: Name of the cell

    Returns:
        List of cell names that are upstream
    """
    internal = InternalApp(app)
    graph = internal.graph

    cell_data = internal.cell_manager.get_cell_data_by_name(cell_name)
    if cell_data is None:
        return []

    target_cell_id = cell_data.cell_id
    ancestors = graph.ancestors(target_cell_id)

    result = []
    for cid in ancestors:
        cd = internal.cell_manager.get_cell_data(cid)
        result.append(cd.name if cd else str(cid))

    return result


# =============================================================================
# Tests
# =============================================================================


class TestNotebookCreation:
    """Test creating marimo notebooks programmatically."""

    def test_create_simple_notebook(self):
        """Test creating a simple notebook with one cell."""
        cells = [
            {"code": "x = 1", "name": "define_x"},
        ]

        app = create_notebook(cells)

        # Verify app was created
        assert app is not None

        # Verify we can get the graph
        graph = get_dependency_graph(app)
        assert "x" in graph["definitions"]

    def test_create_notebook_with_multiple_cells(self):
        """Test creating a notebook with multiple dependent cells."""
        cells = [
            {"code": "x = 10", "name": "define_x"},
            {"code": "y = x * 2", "name": "define_y"},
            {"code": "result = x + y", "name": "compute_result"},
        ]

        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        assert defs["x"] == 10
        assert defs["y"] == 20
        assert defs["result"] == 30

    def test_create_notebook_with_pandas(self):
        """Test creating a notebook that uses pandas."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {
                "code": "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
                "name": "create_df",
            },
            {"code": "total = df['a'].sum()", "name": "compute_total"},
        ]

        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        assert isinstance(defs["df"], pd.DataFrame)
        assert len(defs["df"]) == 3
        assert defs["total"] == 6

    def test_add_cell_to_notebook(self):
        """Test adding a cell to an existing notebook."""
        # Create initial notebook
        cells = [
            {"code": "x = 10", "name": "define_x"},
        ]
        app = create_notebook(cells)

        # Add a new cell
        app = add_cell(app, "y = x * 2", name="define_y")

        # Run and verify
        outputs, defs = run_notebook(app)
        assert defs["x"] == 10
        assert defs["y"] == 20

    def test_save_and_load_notebook(self, tmp_path):
        """Test saving a notebook to file and loading it back."""
        cells = [
            {"code": "x = 42", "name": "define_x"},
            {"code": "y = x + 1", "name": "define_y"},
        ]

        app = create_notebook(cells)
        filepath = save_notebook(app, tmp_path / "test.py")

        # Verify file was created
        assert filepath.exists()
        content = filepath.read_text()
        assert "import marimo" in content
        assert "x = 42" in content

        # Load it back
        loaded_app = load_notebook(filepath)
        outputs, defs = run_notebook(loaded_app)

        assert defs["x"] == 42
        assert defs["y"] == 43


class TestDependencyGraph:
    """Test dependency graph extraction and analysis."""

    def test_simple_dependency_chain(self):
        """Test a simple A -> B -> C dependency chain."""
        cells = [
            {"code": "a = 1", "name": "cell_a"},
            {"code": "b = a + 1", "name": "cell_b"},
            {"code": "c = b + 1", "name": "cell_c"},
        ]

        app = create_notebook(cells)
        graph_info = get_dependency_graph(app)

        # Check definitions
        assert "a" in graph_info["definitions"]
        assert "b" in graph_info["definitions"]
        assert "c" in graph_info["definitions"]

        # Check that execution order respects dependencies
        exec_order = graph_info["execution_order"]
        cell_a_info = get_cell_by_name(app, "cell_a")
        cell_b_info = get_cell_by_name(app, "cell_b")
        cell_c_info = get_cell_by_name(app, "cell_c")

        assert cell_a_info is not None
        assert cell_b_info is not None
        assert cell_c_info is not None

        a_idx = exec_order.index(cell_a_info["cell_id"])
        b_idx = exec_order.index(cell_b_info["cell_id"])
        c_idx = exec_order.index(cell_c_info["cell_id"])

        assert a_idx < b_idx < c_idx

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency: A -> B, A -> C, B -> D, C -> D."""
        cells = [
            {"code": "a = 1", "name": "cell_a"},
            {"code": "b = a * 2", "name": "cell_b"},
            {"code": "c = a * 3", "name": "cell_c"},
            {"code": "d = b + c", "name": "cell_d"},
        ]

        app = create_notebook(cells)
        graph_info = get_dependency_graph(app)

        # Check no cycles
        assert not graph_info["has_cycles"]

        # Run and verify
        outputs, defs = run_notebook(app)
        assert defs["a"] == 1
        assert defs["b"] == 2
        assert defs["c"] == 3
        assert defs["d"] == 5

    def test_get_downstream_cells(self):
        """Test finding cells that depend on a given cell."""
        cells = [
            {"code": "x = 1", "name": "root"},
            {"code": "y = x + 1", "name": "child1"},
            {"code": "z = x + 2", "name": "child2"},
            {"code": "w = y + z", "name": "grandchild"},
        ]

        app = create_notebook(cells)
        downstream = get_downstream_cells(app, "root")

        # root should have child1, child2, and grandchild as descendants
        assert "child1" in downstream
        assert "child2" in downstream
        assert "grandchild" in downstream
        assert len(downstream) == 3

    def test_get_upstream_cells(self):
        """Test finding cells that a given cell depends on."""
        cells = [
            {"code": "x = 1", "name": "root"},
            {"code": "y = x + 1", "name": "child1"},
            {"code": "z = y + 1", "name": "grandchild"},
        ]

        app = create_notebook(cells)
        upstream = get_upstream_cells(app, "grandchild")

        assert "root" in upstream
        assert "child1" in upstream
        assert len(upstream) == 2


class TestReactivity:
    """Test that marimo's reactivity works correctly."""

    def test_override_propagates(self):
        """Test that overriding a variable propagates to dependent cells."""
        cells = [
            {"code": "x = 10", "name": "define_x"},
            {"code": "y = x * 2", "name": "define_y"},
            {"code": "z = y + 5", "name": "define_z"},
        ]

        app = create_notebook(cells)

        # Run with default values
        outputs, defs = run_notebook(app)
        assert defs["x"] == 10
        assert defs["y"] == 20
        assert defs["z"] == 25

        # Run with override - this should propagate through y to z
        outputs, defs = run_notebook(app, overrides={"x": 100, "y": 200})
        assert defs["x"] == 100
        assert defs["y"] == 200
        assert defs["z"] == 205

    def test_partial_execution_with_overrides(self):
        """Test that cells can be skipped when their outputs are overridden."""
        cells = [
            {"code": "expensive = sum(range(1000000))", "name": "expensive_cell"},
            {"code": "result = expensive + 1", "name": "use_expensive"},
        ]

        app = create_notebook(cells)

        # Run with override to skip expensive computation
        outputs, defs = run_notebook(app, overrides={"expensive": 42})
        assert defs["expensive"] == 42
        assert defs["result"] == 43


class TestCellResults:
    """Test retrieving cell results, especially tables."""

    def test_dataframe_result(self):
        """Test that DataFrame results are properly returned."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {
                "code": """data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})""",
                "name": "create_data",
            },
            {"code": "filtered = data[data['age'] > 25]", "name": "filter_data"},
        ]

        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        # Check original data
        assert isinstance(defs["data"], pd.DataFrame)
        assert len(defs["data"]) == 3
        assert list(defs["data"].columns) == ["name", "age", "city"]

        # Check filtered data
        assert isinstance(defs["filtered"], pd.DataFrame)
        assert len(defs["filtered"]) == 2
        assert all(defs["filtered"]["age"] > 25)

    def test_table_operations(self):
        """Test various table operations."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {
                "code": """df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
df2 = pd.DataFrame({'id': [2, 3, 4], 'score': [100, 200, 300]})""",
                "name": "create_tables",
            },
            {
                "code": "merged = pd.merge(df1, df2, on='id', how='inner')",
                "name": "merge_tables",
            },
            {
                "code": "summary = merged.describe()",
                "name": "summarize",
            },
        ]

        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        # Check merged result
        assert isinstance(defs["merged"], pd.DataFrame)
        assert len(defs["merged"]) == 2  # Only ids 2 and 3 match
        assert "value" in defs["merged"].columns
        assert "score" in defs["merged"].columns

        # Check summary
        assert isinstance(defs["summary"], pd.DataFrame)


class TestErrorHandling:
    """Test error handling in notebooks."""

    def test_syntax_error_detection(self):
        """Test that syntax errors are caught."""
        cells = [
            {"code": "x = ", "name": "bad_cell"},  # Syntax error
        ]

        # Should raise an error when trying to run
        with pytest.raises(Exception):
            app = create_notebook(cells)
            run_notebook(app)

    def test_runtime_error_in_cell(self):
        """Test that runtime errors are properly reported."""
        cells = [
            {"code": "x = 1 / 0", "name": "divide_by_zero"},
        ]

        app = create_notebook(cells)

        with pytest.raises(ZeroDivisionError):
            run_notebook(app)


class TestComplexWorkflows:
    """Test more complex notebook workflows."""

    def test_data_pipeline(self):
        """Test a realistic data processing pipeline."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {
                "code": """raw_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'sales': [100, 150, 120, 180, 200, 170, 190, 210, 230, 250],
    'region': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
})""",
                "name": "load_data",
            },
            {
                "code": """cleaned = raw_data.copy()
cleaned['sales'] = cleaned['sales'].astype(float)""",
                "name": "clean_data",
            },
            {
                "code": "by_region = cleaned.groupby('region')['sales'].sum().reset_index()",
                "name": "aggregate",
            },
            {
                "code": "total_sales = cleaned['sales'].sum()",
                "name": "total",
            },
        ]

        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        # Verify pipeline results
        assert len(defs["raw_data"]) == 10
        assert len(defs["by_region"]) == 2
        assert defs["total_sales"] == 1800

        # Check graph structure
        graph = get_dependency_graph(app)
        assert not graph["has_cycles"]

        # Verify dependencies
        downstream_of_load = get_downstream_cells(app, "load_data")
        assert "clean_data" in downstream_of_load
        assert "aggregate" in downstream_of_load
        assert "total" in downstream_of_load

    def test_iterative_notebook_building(self):
        """Test building a notebook cell by cell."""
        # Start with imports
        app = create_notebook([{"code": "import pandas as pd", "name": "imports"}])

        # Add data creation
        app = add_cell(app, "df = pd.DataFrame({'x': [1, 2, 3]})", name="create_data")

        # Add computation
        app = add_cell(app, "total = df['x'].sum()", name="compute")

        # Run and verify
        outputs, defs = run_notebook(app)
        assert defs["total"] == 6

        # Verify graph
        graph = get_dependency_graph(app)
        assert len(graph["cells"]) == 3
        assert "total" in graph["definitions"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
