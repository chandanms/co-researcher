"""
Research Orchestrator - Coordinates multiple parallel research branches.

This module decomposes research questions into independent investigation
branches and executes them as parallel marimo notebooks.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

from llm_client import LLMClient
from orchestrator import MarimoOrchestrator, add_cell, create_empty_notebook
from prompts import research_decomposer_prompt

load_dotenv()


@dataclass
class ResearchBranch:
    """Represents a single research investigation branch."""

    name: str
    description: str
    cells: list[dict[str, str]]
    notebook_path: str = ""
    status: str = "pending"  # pending, running, completed, failed
    error: str | None = None
    progress: int = 0  # 0-100


@dataclass
class NotebookResult:
    """Result of executing a research branch notebook."""

    branch_name: str
    notebook_path: str
    success: bool
    definitions: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ResearchOrchestrator:
    """
    Orchestrates multiple parallel research branches.

    Decomposes a research question into independent branches,
    creates marimo notebooks for each, and executes them in parallel.
    """

    def __init__(
        self,
        csv_files: list[str],
        output_dir: str = "output",
        max_workers: int = 4,
    ):
        """
        Initialize the research orchestrator.

        Args:
            csv_files: List of paths to CSV files for analysis
            output_dir: Directory to save generated notebooks
            max_workers: Maximum parallel notebook executions
        """
        self.csv_files = csv_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm = LLMClient()
        self.max_workers = max_workers
        self.branches: list[ResearchBranch] = []

    def analyze_csv(self, csv_path: str) -> dict[str, Any]:
        """
        Analyze a CSV file and extract metadata for the LLM.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Dictionary with CSV metadata
        """
        df = pd.read_csv(csv_path)

        return {
            "file_path": csv_path,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "row_count": len(df),
            "sample_rows": df.head(5).to_string(),
        }

    def decompose_research_question(
        self,
        question: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> list[ResearchBranch]:
        """
        Decompose a research question into independent branches.

        Args:
            question: The research question from the user
            status_callback: Optional callback for status updates

        Returns:
            List of ResearchBranch objects
        """
        if status_callback:
            status_callback("Analyzing CSV files...")

        # Analyze the first CSV file (for now, we support single file)
        csv_info = self.analyze_csv(self.csv_files[0])

        if status_callback:
            status_callback("Decomposing research question...")

        # Generate the decomposition prompt
        prompt = research_decomposer_prompt(question, csv_info)

        # Call the LLM
        result = self.llm.call(prompt)

        if "error" in result:
            raise RuntimeError(f"Failed to parse LLM response: {result}")

        # Create ResearchBranch objects
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branches = []

        for branch_data in result.get("branches", []):
            branch = ResearchBranch(
                name=branch_data["name"],
                description=branch_data["description"],
                cells=branch_data["cells"],
                notebook_path=str(
                    self.output_dir / f"{branch_data['name']}_{timestamp}.py"
                ),
            )
            branches.append(branch)

        self.branches = branches

        if status_callback:
            status_callback(f"Created {len(branches)} research branches")

        return branches

    def _create_branch_notebook(self, branch: ResearchBranch) -> None:
        """
        Create a marimo notebook for a research branch.

        Args:
            branch: The research branch to create a notebook for
        """
        app = create_empty_notebook()

        for i, cell in enumerate(branch.cells):
            cell_name = f"cell_{i}"
            app = add_cell(app, cell["code"], name=cell_name)

        # Save the notebook
        orchestrator = MarimoOrchestrator(app)
        orchestrator.save_notebook(branch.notebook_path)

    def _execute_branch(
        self,
        branch: ResearchBranch,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> NotebookResult:
        """
        Execute a single research branch.

        Args:
            branch: The research branch to execute
            progress_callback: Callback for progress updates (branch_name, progress)

        Returns:
            NotebookResult with execution results
        """
        try:
            branch.status = "running"
            if progress_callback:
                progress_callback(branch.name, 10)

            # Create the notebook
            self._create_branch_notebook(branch)
            if progress_callback:
                progress_callback(branch.name, 30)

            # Create orchestrator and run the notebook
            app = create_empty_notebook()

            total_cells = len(branch.cells)
            for i, cell in enumerate(branch.cells):
                cell_name = f"cell_{i}"
                app = add_cell(app, cell["code"], name=cell_name)

                # Update progress
                cell_progress = 30 + int((i + 1) / total_cells * 60)
                if progress_callback:
                    progress_callback(branch.name, cell_progress)

            # Run the notebook
            try:
                outputs, defs = app.run()
                branch.status = "completed"
                branch.progress = 100

                if progress_callback:
                    progress_callback(branch.name, 100)

                # Filter out non-serializable definitions for the result
                safe_defs = {}
                for key, value in defs.items():
                    try:
                        # Try to represent the value as a string
                        if isinstance(value, pd.DataFrame):
                            safe_defs[key] = f"DataFrame({value.shape[0]} rows, {value.shape[1]} cols)"
                        elif isinstance(value, pd.Series):
                            safe_defs[key] = f"Series({len(value)} items)"
                        elif hasattr(value, "__module__") and value.__module__ != "builtins":
                            safe_defs[key] = f"{type(value).__name__}"
                        else:
                            safe_defs[key] = repr(value)[:100]
                    except Exception:
                        safe_defs[key] = f"<{type(value).__name__}>"

                return NotebookResult(
                    branch_name=branch.name,
                    notebook_path=branch.notebook_path,
                    success=True,
                    definitions=safe_defs,
                )

            except Exception as e:
                branch.status = "failed"
                branch.error = str(e)

                return NotebookResult(
                    branch_name=branch.name,
                    notebook_path=branch.notebook_path,
                    success=False,
                    error=str(e),
                )

        except Exception as e:
            branch.status = "failed"
            branch.error = str(e)

            return NotebookResult(
                branch_name=branch.name,
                notebook_path=branch.notebook_path,
                success=False,
                error=str(e),
            )

    def execute_branches_parallel(
        self,
        branches: list[ResearchBranch] | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> list[NotebookResult]:
        """
        Execute research branches in parallel.

        Args:
            branches: List of branches to execute (uses self.branches if None)
            progress_callback: Callback for progress updates (branch_name, progress)

        Returns:
            List of NotebookResult objects
        """
        if branches is None:
            branches = self.branches

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._execute_branch, branch, progress_callback): branch
                for branch in branches
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def run_research(
        self,
        question: str,
        status_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> list[NotebookResult]:
        """
        Run the complete research workflow.

        Args:
            question: The research question
            status_callback: Callback for status messages
            progress_callback: Callback for progress updates

        Returns:
            List of NotebookResult objects
        """
        # Decompose the question
        branches = self.decompose_research_question(question, status_callback)

        if status_callback:
            status_callback(f"Starting execution of {len(branches)} branches...")

        # Execute branches in parallel
        results = self.execute_branches_parallel(branches, progress_callback)

        return results
