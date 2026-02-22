"""
Tests for the CLI and ResearchOrchestrator.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cli import CLI, TerminalUI
from prompts import research_decomposer_prompt
from research_orchestrator import (
    NotebookResult,
    ResearchBranch,
    ResearchOrchestrator,
)


class TestTerminalUI:
    """Test the TerminalUI helper class."""

    def test_progress_bar_empty(self):
        """Test progress bar at 0%."""
        bar = TerminalUI.progress_bar(0, width=10)
        assert bar == "[░░░░░░░░░░]"

    def test_progress_bar_half(self):
        """Test progress bar at 50%."""
        bar = TerminalUI.progress_bar(50, width=10)
        assert bar == "[█████░░░░░]"

    def test_progress_bar_full(self):
        """Test progress bar at 100%."""
        bar = TerminalUI.progress_bar(100, width=10)
        assert bar == "[██████████]"

    def test_status_icon_pending(self):
        """Test pending status icon."""
        icon = TerminalUI.status_icon("pending")
        assert "○" in icon

    def test_status_icon_completed(self):
        """Test completed status icon."""
        icon = TerminalUI.status_icon("completed")
        assert "✓" in icon

    def test_status_icon_failed(self):
        """Test failed status icon."""
        icon = TerminalUI.status_icon("failed")
        assert "✗" in icon


class TestCLI:
    """Test the CLI class."""

    def test_find_csv_files(self, tmp_path):
        """Test finding CSV files in a directory."""
        # Create some CSV files
        (tmp_path / "data1.csv").write_text("a,b\n1,2")
        (tmp_path / "data2.csv").write_text("x,y\n3,4")
        (tmp_path / "other.txt").write_text("not a csv")

        cli = CLI()
        files = cli.find_csv_files(str(tmp_path))

        assert len(files) == 2
        assert any("data1.csv" in f for f in files)
        assert any("data2.csv" in f for f in files)

    def test_find_csv_files_empty(self, tmp_path):
        """Test finding CSV files when none exist."""
        cli = CLI()
        files = cli.find_csv_files(str(tmp_path))
        assert files == []


class TestResearchDecomposerPrompt:
    """Test the research decomposer prompt generation."""

    def test_prompt_contains_question(self):
        """Test that the prompt contains the research question."""
        csv_info = {
            "file_path": "test.csv",
            "columns": ["a", "b", "c"],
            "dtypes": {"a": "int64", "b": "float64", "c": "object"},
            "row_count": 100,
            "sample_rows": "Sample data here",
        }

        prompt = research_decomposer_prompt(
            question="Analyze trends in the data",
            csv_info=csv_info,
        )

        assert "Analyze trends in the data" in prompt
        assert "test.csv" in prompt
        assert "columns" in prompt.lower()
        assert "JSON" in prompt

    def test_prompt_contains_csv_info(self):
        """Test that the prompt contains CSV metadata."""
        csv_info = {
            "file_path": "sales.csv",
            "columns": ["date", "amount", "category"],
            "dtypes": {"date": "object", "amount": "float64", "category": "object"},
            "row_count": 500,
            "sample_rows": "2024-01-01,100.00,A",
        }

        prompt = research_decomposer_prompt(
            question="Find correlations",
            csv_info=csv_info,
        )

        assert "sales.csv" in prompt
        assert "date" in prompt
        assert "amount" in prompt
        assert "500" in prompt

    def test_prompt_mentions_parallel_branches(self):
        """Test that the prompt mentions independent/parallel branches."""
        csv_info = {
            "file_path": "data.csv",
            "columns": ["x"],
            "dtypes": {"x": "int64"},
            "row_count": 10,
            "sample_rows": "",
        }

        prompt = research_decomposer_prompt(
            question="Analyze data",
            csv_info=csv_info,
        )

        assert "independent" in prompt.lower() or "parallel" in prompt.lower()
        assert "branches" in prompt.lower()


class TestResearchOrchestrator:
    """Test the ResearchOrchestrator class."""

    def test_init_with_api_key(self, tmp_path):
        """Test initialization with API key."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2")

        orchestrator = ResearchOrchestrator(
            csv_files=[str(csv_file)],
            output_dir=str(tmp_path / "output"),
            anthropic_api_key="test-key",
        )

        assert orchestrator.api_key == "test-key"
        assert len(orchestrator.csv_files) == 1

    def test_init_without_api_key(self, tmp_path):
        """Test initialization fails without API key."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ResearchOrchestrator(
                    csv_files=[str(csv_file)],
                    anthropic_api_key=None,
                )

    def test_analyze_csv(self, tmp_path):
        """Test CSV analysis."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 92.0, 78.5],
        })
        df.to_csv(csv_file, index=False)

        orchestrator = ResearchOrchestrator(
            csv_files=[str(csv_file)],
            anthropic_api_key="test-key",
        )

        info = orchestrator.analyze_csv(str(csv_file))

        assert info["file_path"] == str(csv_file)
        assert "name" in info["columns"]
        assert "age" in info["columns"]
        assert "score" in info["columns"]
        assert info["row_count"] == 3
        assert "int" in info["dtypes"]["age"].lower()

    @patch.object(ResearchOrchestrator, "_execute_branch")
    def test_execute_branches_parallel(self, mock_execute, tmp_path):
        """Test parallel execution of branches."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2")

        orchestrator = ResearchOrchestrator(
            csv_files=[str(csv_file)],
            anthropic_api_key="test-key",
        )

        # Create mock branches
        branches = [
            ResearchBranch(
                name="branch1",
                description="Test branch 1",
                cells=[{"code": "x = 1", "purpose": "Test"}],
                notebook_path=str(tmp_path / "branch1.py"),
            ),
            ResearchBranch(
                name="branch2",
                description="Test branch 2",
                cells=[{"code": "y = 2", "purpose": "Test"}],
                notebook_path=str(tmp_path / "branch2.py"),
            ),
        ]

        mock_execute.return_value = NotebookResult(
            branch_name="test",
            notebook_path="/tmp/test.py",
            success=True,
        )

        results = orchestrator.execute_branches_parallel(branches)

        assert len(results) == 2
        assert mock_execute.call_count == 2


class TestResearchBranch:
    """Test the ResearchBranch dataclass."""

    def test_branch_creation(self):
        """Test creating a research branch."""
        branch = ResearchBranch(
            name="test_branch",
            description="A test branch",
            cells=[
                {"code": "import pandas as pd", "purpose": "imports"},
                {"code": "df = pd.read_csv('data.csv')", "purpose": "load data"},
            ],
        )

        assert branch.name == "test_branch"
        assert branch.status == "pending"
        assert len(branch.cells) == 2

    def test_branch_default_values(self):
        """Test default values in ResearchBranch."""
        branch = ResearchBranch(
            name="test",
            description="desc",
            cells=[],
        )

        assert branch.notebook_path == ""
        assert branch.status == "pending"
        assert branch.error is None
        assert branch.progress == 0


class TestNotebookResult:
    """Test the NotebookResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful notebook result."""
        result = NotebookResult(
            branch_name="test",
            notebook_path="/path/to/notebook.py",
            success=True,
            definitions={"x": "1", "y": "2"},
        )

        assert result.success is True
        assert result.error is None
        assert len(result.definitions) == 2

    def test_failed_result(self):
        """Test creating a failed notebook result."""
        result = NotebookResult(
            branch_name="test",
            notebook_path="/path/to/notebook.py",
            success=False,
            error="Execution failed",
        )

        assert result.success is False
        assert result.error == "Execution failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
