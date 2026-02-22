"""
Marimo Orchestrator - Coordinates LLM-assisted notebook development.

This module provides the main orchestration loop for generating and executing
marimo notebook cells using LLM assistance.
"""

from typing import Any

from dotenv import load_dotenv
from marimo._ast.app import App, InternalApp
from marimo._ast.load import load_notebook_ir
from marimo._schemas.serialization import (
    AppInstantiation,
    CellDef,
    NotebookSerializationV1,
)

from llm_client import LLMClient
from marimo_context_extractor import MarimoContextExtractor
from prompts import (
    backtrack_decision_prompt,
    cell_synthesizer_prompt,
    error_recovery_prompt,
    task_decomposer_prompt,
    verification_spec_prompt,
)

load_dotenv()


def add_cell(
    app: App, code: str, name: str | None = None, options: dict | None = None
) -> App:
    """
    Add a cell to an existing notebook and return a new App.

    Args:
        app: The existing marimo App
        code: The cell code
        name: Optional cell name
        options: Optional cell options

    Returns:
        A new App with the cell added
    """
    internal = InternalApp(app)
    existing_ir = internal.to_ir()

    new_cell = CellDef(
        code=code,
        name=name or f"cell_{len(existing_ir.cells)}",
        options=options or {},
    )

    new_cells = list(existing_ir.cells) + [new_cell]

    new_ir = NotebookSerializationV1(
        app=existing_ir.app,
        header=existing_ir.header,
        cells=new_cells,
    )

    return load_notebook_ir(new_ir)


def create_empty_notebook() -> App:
    """Create an empty marimo notebook."""
    notebook_ir = NotebookSerializationV1(
        app=AppInstantiation(options={}),
        cells=[],
    )
    return load_notebook_ir(notebook_ir)


class MarimoOrchestrator:
    """
    Orchestrates LLM-assisted marimo notebook development.

    Handles the full loop of task decomposition, cell synthesis,
    error recovery, and verification.
    """

    def __init__(self, app: App | None = None):
        """
        Initialize the orchestrator.

        Args:
            app: Initial marimo App (creates empty notebook if None)
        """
        self.app = app or create_empty_notebook()
        self.llm = LLMClient()
        self.max_retries = 3

    def run_task(self, task_description: str, dataset_info: str) -> App:
        """
        Execute a complete task, generating and running notebook cells.

        Args:
            task_description: What to accomplish
            dataset_info: Description of available data

        Returns:
            The final App with all cells added
        """
        # Step 1: Decompose the task into cells
        extractor = MarimoContextExtractor(self.app)
        decompose_prompt = task_decomposer_prompt(
            task_description, dataset_info, extractor
        )
        decomposition = self.llm.call(decompose_prompt)

        if "error" in decomposition:
            raise RuntimeError(f"Task decomposition failed: {decomposition}")

        cells_to_create = decomposition.get("cells", [])

        # Step 2: For each suggested cell, synthesize and add it
        for i, cell_spec in enumerate(cells_to_create):
            cell_name = f"gen_cell_{i}"
            purpose = cell_spec.get("purpose", "")
            suggested_code = cell_spec.get("suggested_code", "")

            # Try to run the notebook to get current defs
            try:
                _, runtime_defs = self.app.run()
            except Exception:
                runtime_defs = {}

            # Synthesize the cell (or use suggested code directly)
            if suggested_code:
                code = suggested_code
            else:
                extractor = MarimoContextExtractor(self.app)
                synth_prompt = cell_synthesizer_prompt(
                    purpose, cell_name, extractor, runtime_defs
                )
                synth_result = self.llm.call(synth_prompt)

                if "error" in synth_result:
                    print(
                        f"Warning: Cell synthesis failed for {cell_name}: {synth_result}"
                    )
                    continue

                code = synth_result.get("code", "")

            if not code:
                continue

            # Add the cell
            self.app = add_cell(self.app, code, name=cell_name)

            # Try to run and handle errors
            success = False
            attempts = 0

            while not success and attempts < self.max_retries:
                try:
                    _, runtime_defs = self.app.run()
                    success = True
                except Exception as e:
                    attempts += 1

                    if attempts >= self.max_retries:
                        # Try backtracking
                        extractor = MarimoContextExtractor(self.app)
                        backtrack_prompt = backtrack_decision_prompt(
                            cell_name, attempts, extractor, runtime_defs
                        )
                        backtrack_result = self.llm.call(backtrack_prompt)

                        target = backtrack_result.get("cell_to_regenerate")
                        if target and target != cell_name:
                            # Would need to regenerate upstream cell
                            # For now, just continue with the error
                            print(f"Backtrack suggested to: {target}")
                        break

                    # Try error recovery
                    extractor = MarimoContextExtractor(self.app)
                    recovery_prompt = error_recovery_prompt(
                        cell_name, extractor, runtime_defs, e
                    )
                    recovery_result = self.llm.call(recovery_prompt)

                    if "error" not in recovery_result:
                        fixed_code = recovery_result.get("fixed_code", "")
                        if fixed_code:
                            # Replace the cell with fixed code
                            self.app = self._replace_cell(
                                self.app, cell_name, fixed_code
                            )

            # Run verification if successful
            if success:
                try:
                    _, runtime_defs = self.app.run()
                    extractor = MarimoContextExtractor(self.app)
                    verify_prompt = verification_spec_prompt(
                        cell_name, extractor, runtime_defs
                    )
                    verify_result = self.llm.call(verify_prompt)

                    if "error" not in verify_result:
                        assertions = verify_result.get("assertions", [])
                        failed = self._run_assertions(assertions, runtime_defs)
                        if failed:
                            print(f"Verification warnings for {cell_name}: {failed}")
                except Exception as e:
                    print(f"Verification failed for {cell_name}: {e}")

        return self.app

    def _replace_cell(self, app: App, cell_name: str, new_code: str) -> App:
        """
        Replace a cell's code in the notebook.

        Args:
            app: The current App
            cell_name: Name of the cell to replace
            new_code: New code for the cell

        Returns:
            New App with replaced cell
        """
        internal = InternalApp(app)
        existing_ir = internal.to_ir()

        new_cells = []
        for cell in existing_ir.cells:
            if cell.name == cell_name:
                new_cells.append(
                    CellDef(
                        code=new_code,
                        name=cell_name,
                        options=cell.options,
                    )
                )
            else:
                new_cells.append(cell)

        new_ir = NotebookSerializationV1(
            app=existing_ir.app,
            header=existing_ir.header,
            cells=new_cells,
        )

        return load_notebook_ir(new_ir)

    def _run_assertions(self, assertions: list[str], defs: dict) -> list[str]:
        """
        Execute assertions and return list of failed ones.

        Args:
            assertions: List of assertion code strings
            defs: Namespace dict for execution

        Returns:
            List of failed assertion strings
        """
        failed = []

        # Create execution namespace with common imports
        namespace = dict(defs)
        namespace["np"] = __import__("numpy")
        namespace["pd"] = __import__("pandas")

        for assertion in assertions:
            try:
                exec(assertion, namespace)
            except AssertionError:
                failed.append(assertion)
            except Exception as e:
                failed.append(f"{assertion} (error: {e})")

        return failed

    def synthesize_cell(
        self,
        purpose: str,
        cell_name: str,
        runtime_defs: dict[str, Any] | None = None,
    ) -> dict:
        """
        Synthesize a single cell for a given purpose.

        Args:
            purpose: What the cell should accomplish
            cell_name: Name for the cell
            runtime_defs: Current runtime definitions

        Returns:
            Dict with 'code', 'explanation', 'defines', 'references'
        """
        if runtime_defs is None:
            try:
                _, runtime_defs = self.app.run()
            except Exception:
                runtime_defs = {}

        extractor = MarimoContextExtractor(self.app)
        prompt = cell_synthesizer_prompt(purpose, cell_name, extractor, runtime_defs)
        return self.llm.call(prompt)

    def recover_from_error(
        self,
        cell_name: str,
        error: Exception,
        runtime_defs: dict[str, Any] | None = None,
    ) -> dict:
        """
        Generate recovery code for a failed cell.

        Args:
            cell_name: Name of the failed cell
            error: The exception that occurred
            runtime_defs: Runtime definitions before failure

        Returns:
            Dict with 'fixed_code', 'bug_type', 'fix_explanation'
        """
        if runtime_defs is None:
            runtime_defs = {}

        extractor = MarimoContextExtractor(self.app)
        prompt = error_recovery_prompt(cell_name, extractor, runtime_defs, error)
        return self.llm.call(prompt)

    def get_verification_spec(
        self,
        cell_name: str,
        runtime_defs: dict[str, Any] | None = None,
    ) -> dict:
        """
        Generate verification assertions for a cell.

        Args:
            cell_name: Name of the cell to verify
            runtime_defs: Current runtime definitions

        Returns:
            Dict with 'reasoning' and 'assertions'
        """
        if runtime_defs is None:
            try:
                _, runtime_defs = self.app.run()
            except Exception:
                runtime_defs = {}

        extractor = MarimoContextExtractor(self.app)
        prompt = verification_spec_prompt(cell_name, extractor, runtime_defs)
        return self.llm.call(prompt)

    def decide_backtrack(
        self,
        cell_name: str,
        attempt_count: int,
        runtime_defs: dict[str, Any] | None = None,
    ) -> dict:
        """
        Decide whether to backtrack to an upstream cell.

        Args:
            cell_name: Name of the failing cell
            attempt_count: Number of fix attempts so far
            runtime_defs: Current runtime definitions

        Returns:
            Dict with 'reasoning', 'cell_to_regenerate', 'confidence'
        """
        if runtime_defs is None:
            try:
                _, runtime_defs = self.app.run()
            except Exception:
                runtime_defs = {}

        extractor = MarimoContextExtractor(self.app)
        prompt = backtrack_decision_prompt(
            cell_name, attempt_count, extractor, runtime_defs
        )
        return self.llm.call(prompt)

    def save_notebook(self, filepath: str) -> None:
        """
        Save the current notebook to a file.

        Args:
            filepath: Path to save the notebook
        """
        internal = InternalApp(self.app)
        content = internal.to_py()
        with open(filepath, "w") as f:
            f.write(content)
