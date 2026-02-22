"""
Marimo Context Extractor - Extracts dependency graph and runtime context from marimo notebooks.

This module provides rich introspection of marimo notebooks for LLM-assisted development.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from marimo._ast.app import App, InternalApp


class MarimoContextExtractor:
    """
    Extracts comprehensive context from a marimo App for LLM prompts.

    Provides static graph analysis (dependencies, execution order) and
    runtime variable inspection (types, shapes, sample values).
    """

    def __init__(self, app: App):
        """
        Initialize the extractor with a marimo App.

        Args:
            app: A marimo App object
        """
        self.app = app
        self._internal = InternalApp(app)
        self._graph = self._internal.graph

    def _cell_id_to_name(self, cell_id) -> str:
        """Convert a cell_id to its name."""
        cell_data = self._internal.cell_manager.get_cell_data(cell_id)
        return cell_data.name if cell_data else str(cell_id)

    def _name_to_cell_id(self, name: str):
        """Convert a cell name to its cell_id."""
        cell_data = self._internal.cell_manager.get_cell_data_by_name(name)
        return cell_data.cell_id if cell_data else None

    def get_full_context(self) -> dict:
        """
        Get complete notebook context including all cells, definitions, and graph structure.

        Returns:
            Dict containing:
            - cells: Dict mapping cell_id to cell info
            - definitions: Dict mapping variable names to defining cell_ids
            - execution_order: List of cell_ids in topological order
            - has_cycles: Boolean indicating if cycles exist
            - parallel_groups: List of cell groups that can run in parallel
        """
        cells_info = {}

        for cell_id, cell in self._graph.cells.items():
            cell_id_str = str(cell_id)
            cell_data = self._internal.cell_manager.get_cell_data(cell_id)

            cells_info[cell_id_str] = {
                "name": cell_data.name if cell_data else None,
                "code": cell.code,
                "defines": list(cell.defs),
                "references": list(cell.refs),
                "parents": [
                    self._cell_id_to_name(p) for p in self._graph.parents[cell_id]
                ],
                "children": [
                    self._cell_id_to_name(c) for c in self._graph.children[cell_id]
                ],
                "stale": cell.stale,
                "disabled": cell.config.disabled,
            }

        definitions = {
            name: [self._cell_id_to_name(cid) for cid in cell_ids]
            for name, cell_ids in self._graph.definitions.items()
        }

        execution_order = [
            self._cell_id_to_name(cid) for cid in self._internal.execution_order
        ]

        return {
            "cells": cells_info,
            "definitions": definitions,
            "execution_order": execution_order,
            "has_cycles": bool(self._graph.cycles),
            "parallel_groups": self.get_parallel_groups(),
        }

    def get_cell_context(self, cell_name: str) -> dict:
        """
        Get full context for one cell including its ancestor chain with their definitions.

        Args:
            cell_name: The name of the cell

        Returns:
            Dict containing cell info and ancestor chain
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return {"error": f"Cell '{cell_name}' not found"}

        cell = self._graph.cells.get(cell_id)
        if cell is None:
            return {"error": f"Cell '{cell_name}' not in graph"}

        # Get ancestors
        ancestor_ids = self._graph.ancestors(cell_id)
        ancestors = []
        for anc_id in ancestor_ids:
            anc_cell = self._graph.cells.get(anc_id)
            if anc_cell:
                ancestors.append(
                    {
                        "name": self._cell_id_to_name(anc_id),
                        "defines": list(anc_cell.defs),
                        "code": anc_cell.code,
                    }
                )

        # Sort ancestors by execution order
        exec_order = list(self._internal.execution_order)
        ancestors.sort(
            key=lambda a: (
                exec_order.index(self._name_to_cell_id(a["name"]))
                if self._name_to_cell_id(a["name"]) in exec_order
                else float("inf")
            )
        )

        return {
            "name": cell_name,
            "code": cell.code,
            "defines": list(cell.defs),
            "references": list(cell.refs),
            "parents": [self._cell_id_to_name(p) for p in self._graph.parents[cell_id]],
            "children": [
                self._cell_id_to_name(c) for c in self._graph.children[cell_id]
            ],
            "stale": cell.stale,
            "disabled": cell.config.disabled,
            "ancestors": ancestors,
        }

    def get_available_variables(self, cell_name: str, defs: dict) -> dict:
        """
        Get variables available to a cell from its ancestors, with rich type information.

        Args:
            cell_name: The name of the cell
            defs: The runtime definitions dict from app.run()

        Returns:
            Dict mapping variable names to type/value info
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return {}

        # Get all variables defined by ancestors
        ancestor_ids = self._graph.ancestors(cell_id)
        available_vars = set()

        for anc_id in ancestor_ids:
            anc_cell = self._graph.cells.get(anc_id)
            if anc_cell:
                available_vars.update(anc_cell.defs)

        # Also include variables from direct parents
        for parent_id in self._graph.parents[cell_id]:
            parent_cell = self._graph.cells.get(parent_id)
            if parent_cell:
                available_vars.update(parent_cell.defs)

        result = {}
        for var_name in available_vars:
            if var_name in defs:
                result[var_name] = self._describe_variable(var_name, defs[var_name])

        return result

    def _describe_variable(self, name: str, value: Any) -> dict:
        """
        Create a rich description of a variable's type and value.

        Args:
            name: Variable name
            value: Variable value

        Returns:
            Dict with type info, shape (if applicable), and sample/value
        """
        type_name = type(value).__name__
        module = type(value).__module__
        if module and module != "builtins":
            full_type = f"{module}.{type_name}"
        else:
            full_type = type_name

        info = {"type": full_type}

        # Handle pandas DataFrame
        if isinstance(value, pd.DataFrame):
            info["shape"] = f"({value.shape[0]}, {value.shape[1]})"
            info["columns"] = list(value.columns)
            info["dtypes"] = {col: str(dtype) for col, dtype in value.dtypes.items()}
            # Sample first few rows
            sample_df = value.head(3)
            info["sample"] = sample_df.to_string(max_cols=5)
            return info

        # Handle pandas Series
        if isinstance(value, pd.Series):
            info["shape"] = f"({len(value)},)"
            info["dtype"] = str(value.dtype)
            info["sample"] = str(value.head(5).tolist())
            return info

        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            info["shape"] = str(value.shape)
            info["dtype"] = str(value.dtype)
            if value.size <= 10:
                info["value"] = str(value.tolist())
            else:
                flat = value.flatten()
                info["sample"] = f"[{flat[0]}, {flat[1]}, ..., {flat[-1]}]"
            return info

        # Handle lists
        if isinstance(value, list):
            info["length"] = len(value)
            if len(value) <= 5:
                info["value"] = self._truncate_repr(value)
            else:
                info["sample"] = (
                    f"[{repr(value[0])}, {repr(value[1])}, ..., {repr(value[-1])}] (len={len(value)})"
                )
            return info

        # Handle dicts
        if isinstance(value, dict):
            info["length"] = len(value)
            keys = list(value.keys())[:5]
            info["keys_sample"] = keys
            if len(value) <= 3:
                info["value"] = self._truncate_repr(value)
            return info

        # Handle sets
        if isinstance(value, set):
            info["length"] = len(value)
            sample = list(value)[:3]
            info["sample"] = str(sample)
            return info

        # Handle callables (functions, classes, etc.)
        if callable(value):
            info["callable"] = True
            if hasattr(value, "__doc__") and value.__doc__:
                info["docstring"] = value.__doc__[:100]
            return info

        # Handle primitive types
        if isinstance(value, (int, float, bool, str, type(None))):
            if isinstance(value, str) and len(value) > 100:
                info["value"] = value[:100] + "..."
            else:
                info["value"] = repr(value)
            return info

        # Fallback for other types
        info["repr"] = self._truncate_repr(value)
        return info

    def _truncate_repr(self, value: Any, max_len: int = 200) -> str:
        """Truncate repr output to max_len characters."""
        r = repr(value)
        if len(r) > max_len:
            return r[:max_len] + "..."
        return r

    def get_parallel_groups(self) -> list[list[str]]:
        """
        Find groups of cells that can run in parallel.

        Cells can run in parallel if they have the same ancestors but no
        dependency between each other.

        Returns:
            List of groups, each group is a list of cell names
        """
        # Build depth map based on longest path from roots
        depths = {}
        exec_order = list(self._internal.execution_order)

        for cell_id in exec_order:
            ancestors = self._graph.ancestors(cell_id)
            if not ancestors:
                depths[cell_id] = 0
            else:
                max_ancestor_depth = max(depths.get(a, 0) for a in ancestors)
                depths[cell_id] = max_ancestor_depth + 1

        # Group cells by depth
        depth_groups = defaultdict(list)
        for cell_id, depth in depths.items():
            depth_groups[depth].append(cell_id)

        # Within each depth, find cells with no dependency between them
        parallel_groups = []
        for depth, cell_ids in sorted(depth_groups.items()):
            if len(cell_ids) <= 1:
                continue

            # Check which cells in this depth have no dependency between them
            independent = []
            for cid in cell_ids:
                # A cell is independent of others at same depth if none of them
                # are in its ancestors or descendants
                others_in_depth = set(cell_ids) - {cid}
                ancestors = self._graph.ancestors(cid)
                descendants = self._graph.descendants(cid)

                # If this cell has no overlap with others at same depth
                if not (others_in_depth & (ancestors | descendants)):
                    independent.append(cid)

            if len(independent) > 1:
                parallel_groups.append(
                    [self._cell_id_to_name(cid) for cid in independent]
                )

        return parallel_groups

    def get_impact_of_change(self, cell_name: str) -> dict:
        """
        Analyze the blast radius if a cell's output changes.

        Args:
            cell_name: The name of the cell

        Returns:
            Dict containing directly affected, transitively affected cells,
            and which parallel branches are affected
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return {"error": f"Cell '{cell_name}' not found"}

        # Direct children
        direct_children = self._graph.children[cell_id]
        directly_affected = [self._cell_id_to_name(c) for c in direct_children]

        # All descendants
        all_descendants = self._graph.descendants(cell_id)
        transitively_affected = [
            self._cell_id_to_name(d)
            for d in all_descendants
            if d not in direct_children
        ]

        # Find parallel branches that would be affected
        parallel_groups = self.get_parallel_groups()
        parallel_branches_affected = []

        all_affected_names = set(directly_affected + transitively_affected)
        for group in parallel_groups:
            affected_in_group = [name for name in group if name in all_affected_names]
            if len(affected_in_group) > 1:
                parallel_branches_affected.append(affected_in_group)

        return {
            "directly_affected": directly_affected,
            "transitively_affected": transitively_affected,
            "parallel_branches_affected": parallel_branches_affected,
        }

    def get_cell_code(self, cell_name: str) -> str | None:
        """
        Get the code for a specific cell.

        Args:
            cell_name: The name of the cell

        Returns:
            The cell's code or None if not found
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return None

        cell = self._graph.cells.get(cell_id)
        return cell.code if cell else None

    def get_cell_defines(self, cell_name: str) -> list[str]:
        """
        Get the variables defined by a cell.

        Args:
            cell_name: The name of the cell

        Returns:
            List of variable names defined by the cell
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return []

        cell = self._graph.cells.get(cell_id)
        return list(cell.defs) if cell else []

    def get_cell_references(self, cell_name: str) -> list[str]:
        """
        Get the variables referenced by a cell.

        Args:
            cell_name: The name of the cell

        Returns:
            List of variable names referenced by the cell
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return []

        cell = self._graph.cells.get(cell_id)
        return list(cell.refs) if cell else []

    def get_downstream_cells(self, cell_name: str) -> list[str]:
        """
        Get all cells that depend on this cell.

        Args:
            cell_name: The name of the cell

        Returns:
            List of cell names that are downstream
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return []

        descendants = self._graph.descendants(cell_id)
        return [self._cell_id_to_name(d) for d in descendants]

    def get_upstream_cells(self, cell_name: str) -> list[str]:
        """
        Get all cells that this cell depends on.

        Args:
            cell_name: The name of the cell

        Returns:
            List of cell names that are upstream
        """
        cell_id = self._name_to_cell_id(cell_name)
        if cell_id is None:
            return []

        ancestors = self._graph.ancestors(cell_id)
        return [self._cell_id_to_name(a) for a in ancestors]
