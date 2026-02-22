"""
Tests for the Marimo Context Extractor and Orchestrator.

Reuses helper functions from test_notebook_api.py.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from marimo_context_extractor import MarimoContextExtractor
from orchestrator import MarimoOrchestrator, add_cell, create_empty_notebook
from prompts import (
    backtrack_decision_prompt,
    cell_synthesizer_prompt,
    error_recovery_prompt,
    task_decomposer_prompt,
    verification_spec_prompt,
)

# Import helper functions from existing test file
from tests.test_notebook_api import (
    create_notebook,
    get_dependency_graph,
    get_downstream_cells,
    get_upstream_cells,
    run_notebook,
)

# =============================================================================
# Tests for MarimoContextExtractor
# =============================================================================


class TestMarimoContextExtractor:
    """Test the MarimoContextExtractor class."""

    def test_get_full_context_simple(self):
        """Test getting full context from a simple notebook."""
        cells = [
            {"code": "x = 1", "name": "cell_a"},
            {"code": "y = x + 1", "name": "cell_b"},
            {"code": "z = y + 1", "name": "cell_c"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        context = extractor.get_full_context()

        # Check structure
        assert "cells" in context
        assert "definitions" in context
        assert "execution_order" in context
        assert "has_cycles" in context
        assert "parallel_groups" in context

        # Check definitions
        assert "x" in context["definitions"]
        assert "y" in context["definitions"]
        assert "z" in context["definitions"]

        # Check execution order
        assert len(context["execution_order"]) == 3

        # No cycles in this graph
        assert context["has_cycles"] is False

    def test_get_full_context_with_parallel(self):
        """Test getting context from a notebook with parallel cells."""
        cells = [
            {"code": "x = 1", "name": "root"},
            {"code": "y = x + 1", "name": "branch_a"},
            {"code": "z = x + 2", "name": "branch_b"},
            {"code": "result = y + z", "name": "merge"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        context = extractor.get_full_context()

        # branch_a and branch_b should be in parallel groups
        parallel_groups = context["parallel_groups"]

        # Check that we found at least one parallel group
        found_parallel = False
        for group in parallel_groups:
            if "branch_a" in group and "branch_b" in group:
                found_parallel = True
                break

        assert found_parallel, "branch_a and branch_b should be parallel"

    def test_get_cell_context(self):
        """Test getting context for a single cell."""
        cells = [
            {"code": "a = 1", "name": "cell_a"},
            {"code": "b = a + 1", "name": "cell_b"},
            {"code": "c = b + 1", "name": "cell_c"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        context = extractor.get_cell_context("cell_c")

        assert context["name"] == "cell_c"
        assert context["code"] == "c = b + 1"
        assert "c" in context["defines"]
        assert "b" in context["references"]

        # Check ancestors
        ancestor_names = [a["name"] for a in context["ancestors"]]
        assert "cell_a" in ancestor_names
        assert "cell_b" in ancestor_names

    def test_get_cell_context_not_found(self):
        """Test getting context for a non-existent cell."""
        cells = [{"code": "x = 1", "name": "cell_a"}]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        context = extractor.get_cell_context("nonexistent")

        assert "error" in context

    def test_get_available_variables(self):
        """Test getting available variables with runtime info."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {
                "code": "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
                "name": "create_df",
            },
            {"code": "total = df['a'].sum()", "name": "compute"},
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        extractor = MarimoContextExtractor(app)
        available = extractor.get_available_variables("compute", defs)

        # df should be available to compute cell
        assert "df" in available
        assert "DataFrame" in available["df"]["type"]
        assert "shape" in available["df"]
        assert "columns" in available["df"]

    def test_get_available_variables_numpy(self):
        """Test variable descriptions for numpy arrays."""
        cells = [
            {"code": "import numpy as np", "name": "imports"},
            {"code": "arr = np.array([1, 2, 3, 4, 5])", "name": "create_arr"},
            {"code": "mean = arr.mean()", "name": "compute"},
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        extractor = MarimoContextExtractor(app)
        available = extractor.get_available_variables("compute", defs)

        assert "arr" in available
        assert "ndarray" in available["arr"]["type"]
        assert "shape" in available["arr"]
        assert "dtype" in available["arr"]

    def test_get_parallel_groups(self):
        """Test finding parallel cell groups."""
        cells = [
            {"code": "data = [1, 2, 3]", "name": "root"},
            {"code": "doubled = [x * 2 for x in data]", "name": "branch1"},
            {"code": "tripled = [x * 3 for x in data]", "name": "branch2"},
            {"code": "combined = doubled + tripled", "name": "merge"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        parallel_groups = extractor.get_parallel_groups()

        # branch1 and branch2 should be parallel
        found = False
        for group in parallel_groups:
            if set(group) == {"branch1", "branch2"}:
                found = True
                break

        assert found, "branch1 and branch2 should be in same parallel group"

    def test_get_impact_of_change(self):
        """Test analyzing impact of changing a cell."""
        cells = [
            {"code": "a = 1", "name": "root"},
            {"code": "b = a + 1", "name": "child1"},
            {"code": "c = a + 2", "name": "child2"},
            {"code": "d = b + c", "name": "grandchild"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        impact = extractor.get_impact_of_change("root")

        # root's children should be directly affected
        assert "child1" in impact["directly_affected"]
        assert "child2" in impact["directly_affected"]

        # grandchild should be transitively affected
        assert "grandchild" in impact["transitively_affected"]

    def test_get_downstream_cells(self):
        """Test getting downstream cells."""
        cells = [
            {"code": "x = 1", "name": "root"},
            {"code": "y = x + 1", "name": "child"},
            {"code": "z = y + 1", "name": "grandchild"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        downstream = extractor.get_downstream_cells("root")

        assert "child" in downstream
        assert "grandchild" in downstream

    def test_get_upstream_cells(self):
        """Test getting upstream cells."""
        cells = [
            {"code": "x = 1", "name": "root"},
            {"code": "y = x + 1", "name": "child"},
            {"code": "z = y + 1", "name": "grandchild"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        upstream = extractor.get_upstream_cells("grandchild")

        assert "root" in upstream
        assert "child" in upstream


# =============================================================================
# Tests for Prompts
# =============================================================================


class TestPrompts:
    """Test the prompt generation functions."""

    def test_task_decomposer_prompt(self):
        """Test task decomposition prompt generation."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
        ]
        app = create_notebook(cells)
        extractor = MarimoContextExtractor(app)

        prompt = task_decomposer_prompt(
            task_description="Load data and compute statistics",
            dataset_info="CSV file with columns: id, value, category",
            extractor=extractor,
        )

        # Check prompt contains key elements
        assert "CRITICAL RULES FOR MARIMO CELLS" in prompt
        assert (
            "no 'return' statements" in prompt.lower() or "NEVER use 'return'" in prompt
        )
        assert "Load data and compute statistics" in prompt
        assert "JSON" in prompt

    def test_cell_synthesizer_prompt(self):
        """Test cell synthesis prompt generation."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {"code": "df = pd.DataFrame({'a': [1, 2, 3]})", "name": "create_df"},
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)
        extractor = MarimoContextExtractor(app)

        prompt = cell_synthesizer_prompt(
            purpose="Calculate the sum of column 'a'",
            cell_name="compute_sum",
            extractor=extractor,
            runtime_defs=defs,
        )

        # Check prompt contains key elements
        assert "Calculate the sum of column 'a'" in prompt
        assert "compute_sum" in prompt
        assert "df" in prompt  # Should mention available variable
        assert "JSON" in prompt

    def test_error_recovery_prompt(self):
        """Test error recovery prompt generation."""
        cells = [
            {"code": "x = 1", "name": "define_x"},
            {"code": "y = x + 'string'", "name": "bad_cell"},
        ]
        app = create_notebook(cells)

        # Try to run and catch the error
        try:
            run_notebook(app)
            error = None
        except TypeError as e:
            error = e

        if error is None:
            # Create a mock error for testing
            error = TypeError("unsupported operand type(s) for +: 'int' and 'str'")

        extractor = MarimoContextExtractor(app)
        prompt = error_recovery_prompt(
            failed_cell_name="bad_cell",
            extractor=extractor,
            runtime_defs={"x": 1},
            error=error,
        )

        # Check prompt contains key elements
        assert "bad_cell" in prompt
        assert "TypeError" in prompt
        assert "JSON" in prompt

    def test_verification_spec_prompt(self):
        """Test verification spec prompt generation."""
        cells = [
            {"code": "x = [1, 2, 3]", "name": "create_list"},
            {"code": "total = sum(x)", "name": "compute_sum"},
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)
        extractor = MarimoContextExtractor(app)

        prompt = verification_spec_prompt(
            cell_name="create_list",
            extractor=extractor,
            runtime_defs=defs,
        )

        # Check prompt contains key elements
        assert "create_list" in prompt
        assert "JSON" in prompt
        assert "assertions" in prompt.lower()

    def test_backtrack_decision_prompt(self):
        """Test backtrack decision prompt generation."""
        cells = [
            {"code": "x = 1", "name": "root"},
            {"code": "y = x + 1", "name": "child"},
            {"code": "z = y + 1", "name": "failing"},
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)
        extractor = MarimoContextExtractor(app)

        prompt = backtrack_decision_prompt(
            failed_cell_name="failing",
            attempt_count=3,
            extractor=extractor,
            runtime_defs=defs,
        )

        # Check prompt contains key elements
        assert "failing" in prompt
        assert "3" in prompt  # attempt count
        assert "JSON" in prompt
        assert "backtrack" in prompt.lower() or "regenerate" in prompt.lower()


# =============================================================================
# Tests for Orchestrator
# =============================================================================


class TestOrchestrator:
    """Test the MarimoOrchestrator class."""

    def test_create_empty_notebook(self):
        """Test creating an empty notebook."""
        app = create_empty_notebook()
        graph = get_dependency_graph(app)

        assert len(graph["cells"]) == 0

    def test_add_cell(self):
        """Test adding a cell to a notebook."""
        app = create_empty_notebook()
        app = add_cell(app, "x = 1", name="define_x")

        graph = get_dependency_graph(app)
        assert len(graph["cells"]) == 1
        assert "x" in graph["definitions"]

    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        with patch("llm_client.LLMClient"):
            app = create_empty_notebook()
            orchestrator = MarimoOrchestrator(app)

            assert orchestrator.app is app
            assert orchestrator.max_retries == 3

    def test_orchestrator_init_without_env(self):
        """Test orchestrator fails without environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_ANTHROPIC"):
                MarimoOrchestrator()

    @patch("orchestrator.LLMClient")
    def test_synthesize_cell(self, mock_llm_class):
        """Test cell synthesis with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = {
            "code": "result = x + y",
            "explanation": "Add x and y",
            "defines": ["result"],
            "references": ["x", "y"],
        }
        mock_llm_class.return_value = mock_llm

        cells = [
            {"code": "x = 1", "name": "define_x"},
            {"code": "y = 2", "name": "define_y"},
        ]
        app = create_notebook(cells)
        orchestrator = MarimoOrchestrator(app)

        result = orchestrator.synthesize_cell(
            purpose="Add x and y together",
            cell_name="compute_result",
        )

        assert result["code"] == "result = x + y"
        assert "result" in result["defines"]
        mock_llm.call.assert_called_once()

    @patch("orchestrator.LLMClient")
    def test_recover_from_error(self, mock_llm_class):
        """Test error recovery with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = {
            "fixed_code": "y = str(x) + 'suffix'",
            "bug_type": "TypeError",
            "fix_explanation": "Convert int to string before concatenation",
        }
        mock_llm_class.return_value = mock_llm

        cells = [
            {"code": "x = 1", "name": "define_x"},
            {"code": "y = x + 'suffix'", "name": "bad_cell"},
        ]
        app = create_notebook(cells)
        orchestrator = MarimoOrchestrator(app)

        error = TypeError("unsupported operand type(s)")
        result = orchestrator.recover_from_error(
            cell_name="bad_cell",
            error=error,
            runtime_defs={"x": 1},
        )

        assert "fixed_code" in result
        assert result["bug_type"] == "TypeError"

    @patch("orchestrator.LLMClient")
    def test_get_verification_spec(self, mock_llm_class):
        """Test verification spec generation with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = {
            "reasoning": "downstream uses sum, so must be numeric list",
            "assertions": [
                "assert isinstance(x, list)",
                "assert all(isinstance(i, (int, float)) for i in x)",
            ],
        }
        mock_llm_class.return_value = mock_llm

        cells = [
            {"code": "x = [1, 2, 3]", "name": "create_list"},
            {"code": "total = sum(x)", "name": "compute_sum"},
        ]
        app = create_notebook(cells)
        orchestrator = MarimoOrchestrator(app)

        result = orchestrator.get_verification_spec(cell_name="create_list")

        assert "assertions" in result
        assert len(result["assertions"]) == 2

    @patch("orchestrator.LLMClient")
    def test_decide_backtrack(self, mock_llm_class):
        """Test backtrack decision with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = {
            "reasoning": "The upstream cell produces wrong type",
            "cell_to_regenerate": "define_x",
            "confidence": 0.85,
        }
        mock_llm_class.return_value = mock_llm

        cells = [
            {"code": "x = 1", "name": "define_x"},
            {"code": "y = x + 1", "name": "failing_cell"},
        ]
        app = create_notebook(cells)
        orchestrator = MarimoOrchestrator(app)

        result = orchestrator.decide_backtrack(
            cell_name="failing_cell",
            attempt_count=3,
        )

        assert result["cell_to_regenerate"] == "define_x"
        assert result["confidence"] == 0.85

    @patch("orchestrator.LLMClient")
    def test_run_assertions_pass(self, mock_llm_class):
        """Test running assertions that pass."""
        orchestrator = MarimoOrchestrator(create_empty_notebook())

        assertions = [
            "assert x == 1",
            "assert isinstance(y, list)",
        ]
        defs = {"x": 1, "y": [1, 2, 3]}

        failed = orchestrator._run_assertions(assertions, defs)

        assert len(failed) == 0

    @patch("orchestrator.LLMClient")
    def test_run_assertions_fail(self, mock_llm_class):
        """Test running assertions that fail."""
        orchestrator = MarimoOrchestrator(create_empty_notebook())

        assertions = [
            "assert x == 2",  # Will fail
            "assert isinstance(y, list)",  # Will pass
        ]
        defs = {"x": 1, "y": [1, 2, 3]}

        failed = orchestrator._run_assertions(assertions, defs)

        assert len(failed) == 1
        assert "x == 2" in failed[0]

    @patch("orchestrator.LLMClient")
    def test_replace_cell(self, mock_llm_class):
        """Test replacing a cell's code."""
        cells = [
            {"code": "x = 1", "name": "cell_a"},
            {"code": "y = x + 1", "name": "cell_b"},
        ]
        app = create_notebook(cells)
        orchestrator = MarimoOrchestrator(app)

        new_app = orchestrator._replace_cell(app, "cell_a", "x = 100")
        outputs, defs = run_notebook(new_app)

        assert defs["x"] == 100
        assert defs["y"] == 101

    @patch("orchestrator.LLMClient")
    def test_run_task_simple(self, mock_llm_class):
        """Test running a simple task with mocked LLM."""
        mock_llm = MagicMock()
        # Mock responses for task decomposition and verification
        mock_llm.call.side_effect = [
            # Task decomposition response
            {
                "reasoning": "Simple math task",
                "cells": [
                    {
                        "suggested_code": "result = 1 + 2",
                        "purpose": "Add numbers",
                        "expected_defines": ["result"],
                        "expected_references": [],
                    }
                ],
            },
            # Verification response
            {
                "reasoning": "Result should be integer",
                "assertions": ["assert isinstance(result, int)"],
            },
        ]
        mock_llm_class.return_value = mock_llm

        app = create_empty_notebook()
        orchestrator = MarimoOrchestrator(app)

        result_app = orchestrator.run_task(
            task_description="Add 1 and 2",
            dataset_info="No data needed",
        )

        # Verify the task ran
        outputs, defs = run_notebook(result_app)
        assert defs["result"] == 3

    def test_llm_client_json_parsing(self):
        """Test LLM client JSON parsing via LLMClient.call()."""
        from llm_client import LLMClient

        with patch.object(LLMClient, "__init__", lambda self: None):
            client = LLMClient()
            client.client = MagicMock()
            client.model = "test-model"
            client.max_tokens = 4096

            # Mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"key": "value"}')]
            client.client.messages.create.return_value = mock_response

            result = client.call("test prompt")

        assert result == {"key": "value"}

    def test_llm_client_markdown_stripping(self):
        """Test LLM client markdown code block stripping."""
        from llm_client import LLMClient

        with patch.object(LLMClient, "__init__", lambda self: None):
            client = LLMClient()
            client.client = MagicMock()
            client.model = "test-model"
            client.max_tokens = 4096

            # Mock response with markdown code block
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='```json\n{"key": "value"}\n```')]
            client.client.messages.create.return_value = mock_response

            result = client.call("test prompt")

        assert result == {"key": "value"}

    def test_llm_client_parse_error(self):
        """Test LLM client parse error handling."""
        from llm_client import LLMClient

        with patch.object(LLMClient, "__init__", lambda self: None):
            client = LLMClient()
            client.client = MagicMock()
            client.model = "test-model"
            client.max_tokens = 4096

            # Mock response with invalid JSON
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="not valid json")]
            client.client.messages.create.return_value = mock_response

            result = client.call("test prompt")

        assert "error" in result
        assert result["error"] == "parse_failed"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining extractor, prompts, and orchestrator."""

    def test_extractor_with_pandas_workflow(self):
        """Test extractor with a realistic pandas workflow."""
        cells = [
            {"code": "import pandas as pd\nimport numpy as np", "name": "imports"},
            {
                "code": "df = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})",
                "name": "create_data",
            },
            {"code": "df_filtered = df[df['a'] > 0]", "name": "filter_data"},
            {"code": "stats = df_filtered.describe()", "name": "compute_stats"},
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        extractor = MarimoContextExtractor(app)

        # Test full context
        context = extractor.get_full_context()
        assert len(context["cells"]) == 4

        # Test available variables for stats cell
        available = extractor.get_available_variables("compute_stats", defs)
        assert "df_filtered" in available
        assert "DataFrame" in available["df_filtered"]["type"]

        # Test impact analysis
        impact = extractor.get_impact_of_change("create_data")
        assert "filter_data" in impact["directly_affected"]
        assert "compute_stats" in impact["transitively_affected"]

    def test_prompt_chain_for_new_cell(self):
        """Test generating prompts for adding a new cell."""
        cells = [
            {"code": "import pandas as pd", "name": "imports"},
            {
                "code": "df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})",
                "name": "data",
            },
        ]
        app = create_notebook(cells)
        outputs, defs = run_notebook(app)

        extractor = MarimoContextExtractor(app)

        # Generate decomposition prompt
        decomp_prompt = task_decomposer_prompt(
            task_description="Calculate correlation between x and y",
            dataset_info="DataFrame with columns x and y",
            extractor=extractor,
        )
        assert "correlation" in decomp_prompt.lower()
        assert "DataFrame" in decomp_prompt or "definitions" in decomp_prompt.lower()

        # Generate synthesis prompt
        synth_prompt = cell_synthesizer_prompt(
            purpose="Calculate correlation",
            cell_name="correlation_cell",
            extractor=extractor,
            runtime_defs=defs,
        )
        assert "correlation_cell" in synth_prompt
        assert "df" in synth_prompt

    @patch("orchestrator.LLMClient")
    def test_full_workflow_mocked(self, mock_llm_class):
        """Test complete workflow with mocked LLM calls."""
        mock_llm = MagicMock()
        # Set up mock responses
        mock_llm.call.side_effect = [
            # Task decomposition
            {
                "reasoning": "Need to create data and compute mean",
                "cells": [
                    {
                        "suggested_code": "data = [1, 2, 3, 4, 5]",
                        "purpose": "Create sample data",
                        "expected_defines": ["data"],
                        "expected_references": [],
                    },
                    {
                        "suggested_code": "mean_value = sum(data) / len(data)",
                        "purpose": "Compute mean",
                        "expected_defines": ["mean_value"],
                        "expected_references": ["data"],
                    },
                ],
            },
            # First verification
            {
                "reasoning": "data should be a list",
                "assertions": ["assert isinstance(data, list)"],
            },
            # Second verification
            {
                "reasoning": "mean should be numeric",
                "assertions": ["assert isinstance(mean_value, (int, float))"],
            },
        ]
        mock_llm_class.return_value = mock_llm

        app = create_empty_notebook()
        orchestrator = MarimoOrchestrator(app)

        result_app = orchestrator.run_task(
            task_description="Create data and compute its mean",
            dataset_info="No external data",
        )

        outputs, defs = run_notebook(result_app)

        assert defs["data"] == [1, 2, 3, 4, 5]
        assert defs["mean_value"] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
