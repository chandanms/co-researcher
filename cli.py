"""
Co-Researcher CLI - Interactive command-line interface for parallel research.

Provides a terminal-based interface for asking research questions,
selecting data files, and monitoring parallel notebook execution.
"""

import os
import sys
import threading
import time
from pathlib import Path

from research_orchestrator import NotebookResult, ResearchBranch, ResearchOrchestrator


class TerminalUI:
    """Handles terminal display with ANSI escape codes."""

    # ANSI escape codes
    CLEAR_LINE = "\033[2K"
    MOVE_UP = "\033[A"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    DIM = "\033[2m"

    @staticmethod
    def clear_screen():
        """Clear the terminal screen."""
        os.system("clear" if os.name == "posix" else "cls")

    @staticmethod
    def print_banner():
        """Print the welcome banner."""
        banner = f"""
{TerminalUI.CYAN}{TerminalUI.BOLD}
 ██████╗ ██████╗       ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗███████╗██████╗
██╔════╝██╔═══██╗      ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗
██║     ██║   ██║█████╗██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║█████╗  ██████╔╝
██║     ██║   ██║╚════╝██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║██╔══╝  ██╔══██╗
╚██████╗╚██████╔╝      ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║███████╗██║  ██║
 ╚═════╝ ╚═════╝       ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
{TerminalUI.RESET}
{TerminalUI.DIM}Parallel Research Assistant powered by Marimo + Claude{TerminalUI.RESET}
"""
        print(banner)

    @staticmethod
    def print_box(title: str, content: list[str], width: int = 70):
        """Print a bordered box with content."""
        print(f"╔{'═' * width}╗")
        print(f"║  {TerminalUI.BOLD}{title}{TerminalUI.RESET}{' ' * (width - len(title) - 2)}║")
        print(f"╠{'═' * width}╣")
        for line in content:
            # Truncate if too long
            display_line = line[: width - 4] if len(line) > width - 4 else line
            padding = width - len(display_line) - 2
            print(f"║  {display_line}{' ' * padding}║")
        print(f"╚{'═' * width}╝")

    @staticmethod
    def progress_bar(progress: int, width: int = 20) -> str:
        """Create a progress bar string."""
        filled = int(width * progress / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    @staticmethod
    def status_icon(status: str) -> str:
        """Get status icon for a branch."""
        icons = {
            "pending": f"{TerminalUI.DIM}○{TerminalUI.RESET}",
            "running": f"{TerminalUI.YELLOW}●{TerminalUI.RESET}",
            "completed": f"{TerminalUI.GREEN}✓{TerminalUI.RESET}",
            "failed": f"{TerminalUI.RED}✗{TerminalUI.RESET}",
        }
        return icons.get(status, "?")


class CLI:
    """Main CLI interface for co-researcher."""

    def __init__(self):
        """Initialize the CLI."""
        self.ui = TerminalUI()
        self.orchestrator: ResearchOrchestrator | None = None
        self.branch_progress: dict[str, int] = {}
        self.status_message: str = ""
        self.display_lock = threading.Lock()

    def find_csv_files(self, directory: str = ".") -> list[str]:
        """
        Find all CSV files in the given directory.

        Args:
            directory: Directory to search

        Returns:
            List of CSV file paths
        """
        path = Path(directory)
        return sorted([str(f) for f in path.glob("**/*.csv") if ".venv" not in str(f)])

    def select_files(self, csv_files: list[str]) -> list[str]:
        """
        Interactive file selection.

        Args:
            csv_files: List of available CSV files

        Returns:
            List of selected file paths
        """
        if not csv_files:
            print(f"\n{TerminalUI.RED}No CSV files found in current directory.{TerminalUI.RESET}")
            print("Please place a CSV file in the current directory or specify a path.")
            path = input(f"\n{TerminalUI.BOLD}Enter CSV file path (or 'q' to quit): {TerminalUI.RESET}").strip()
            if path.lower() == "q":
                return []
            if Path(path).exists():
                return [path]
            print(f"{TerminalUI.RED}File not found: {path}{TerminalUI.RESET}")
            return []

        print(f"\n{TerminalUI.BOLD}Available CSV files:{TerminalUI.RESET}")
        print(f"{TerminalUI.DIM}─" * 50 + f"{TerminalUI.RESET}")

        for i, file in enumerate(csv_files, 1):
            # Get file size
            size = Path(file).stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
            print(f"  {TerminalUI.CYAN}{i}{TerminalUI.RESET}. {file} {TerminalUI.DIM}({size_str}){TerminalUI.RESET}")

        print(f"{TerminalUI.DIM}─" * 50 + f"{TerminalUI.RESET}")
        print(f"\n{TerminalUI.DIM}Enter file numbers separated by commas (e.g., 1,2,3){TerminalUI.RESET}")
        print(f"{TerminalUI.DIM}Or press Enter to select all files{TerminalUI.RESET}")

        selection = input(f"\n{TerminalUI.BOLD}Select files: {TerminalUI.RESET}").strip()

        if not selection:
            # Select all files if Enter pressed
            return csv_files

        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected = [csv_files[i] for i in indices if 0 <= i < len(csv_files)]
            return selected
        except (ValueError, IndexError):
            print(f"{TerminalUI.RED}Invalid selection. Using first file.{TerminalUI.RESET}")
            return [csv_files[0]]

    def get_research_question(self) -> str:
        """
        Get the research question from the user.

        Returns:
            The research question string
        """
        print(f"\n{TerminalUI.BOLD}What would you like to research?{TerminalUI.RESET}")
        print(f"{TerminalUI.DIM}Examples:{TerminalUI.RESET}")
        print(f"{TerminalUI.DIM}  - Analyze trends and correlations in the data{TerminalUI.RESET}")
        print(f"{TerminalUI.DIM}  - Compare different clustering methods{TerminalUI.RESET}")
        print(f"{TerminalUI.DIM}  - Find anomalies and outliers{TerminalUI.RESET}")

        question = input(f"\n{TerminalUI.BOLD}Research question: {TerminalUI.RESET}").strip()
        return question

    def _status_callback(self, message: str):
        """Callback for status updates."""
        with self.display_lock:
            self.status_message = message
            print(f"\n{TerminalUI.CYAN}→{TerminalUI.RESET} {message}")

    def _progress_callback(self, branch_name: str, progress: int):
        """Callback for progress updates."""
        with self.display_lock:
            self.branch_progress[branch_name] = progress

    def display_branches(self, branches: list[ResearchBranch]):
        """
        Display detailed information about research branches after decomposition.

        Args:
            branches: List of research branches
        """
        print(f"\n{TerminalUI.BOLD}{'═' * 70}{TerminalUI.RESET}")
        print(f"{TerminalUI.BOLD}Research Plan - {len(branches)} Branches{TerminalUI.RESET}")
        print(f"{TerminalUI.BOLD}{'═' * 70}{TerminalUI.RESET}")

        for i, branch in enumerate(branches, 1):
            print(f"\n{TerminalUI.CYAN}[{i}] {branch.name}{TerminalUI.RESET}")
            print(f"    {TerminalUI.DIM}Description:{TerminalUI.RESET} {branch.description}")
            print(f"    {TerminalUI.DIM}Cells:{TerminalUI.RESET} {len(branch.cells)} steps")

            # Show brief cell descriptions
            for j, cell in enumerate(branch.cells[:3], 1):
                purpose = cell.get("purpose", "")[:50]
                if len(cell.get("purpose", "")) > 50:
                    purpose += "..."
                print(f"      {TerminalUI.DIM}{j}. {purpose}{TerminalUI.RESET}")

            if len(branch.cells) > 3:
                print(f"      {TerminalUI.DIM}... and {len(branch.cells) - 3} more steps{TerminalUI.RESET}")

        print(f"\n{TerminalUI.BOLD}{'─' * 70}{TerminalUI.RESET}")

    def display_progress(self, branches: list[ResearchBranch], running: bool = True):
        """
        Display the current progress of all branches.

        Args:
            branches: List of research branches
            running: Whether execution is still running
        """
        print(f"\n{TerminalUI.BOLD}Research Branches:{TerminalUI.RESET}")
        print(f"{'─' * 70}")

        for branch in branches:
            progress = self.branch_progress.get(branch.name, 0)
            icon = TerminalUI.status_icon(branch.status)
            bar = TerminalUI.progress_bar(progress)

            # Color based on status
            if branch.status == "completed":
                color = TerminalUI.GREEN
            elif branch.status == "failed":
                color = TerminalUI.RED
            elif branch.status == "running":
                color = TerminalUI.YELLOW
            else:
                color = TerminalUI.DIM

            name_display = branch.name[:25].ljust(25)
            print(f"  {icon} {color}{name_display}{TerminalUI.RESET} {bar} {progress:3d}%")

            if branch.status == "failed" and branch.error:
                error_short = branch.error[:50] + "..." if len(branch.error) > 50 else branch.error
                print(f"      {TerminalUI.RED}Error: {error_short}{TerminalUI.RESET}")

        print(f"{'─' * 70}")

    def display_results(self, results: list[NotebookResult]):
        """
        Display the final results.

        Args:
            results: List of notebook results
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n{TerminalUI.BOLD}{'═' * 70}{TerminalUI.RESET}")
        print(f"{TerminalUI.BOLD}RESEARCH COMPLETE{TerminalUI.RESET}")
        print(f"{TerminalUI.BOLD}{'═' * 70}{TerminalUI.RESET}")

        print(f"\n{TerminalUI.GREEN}✓ Successful: {len(successful)}{TerminalUI.RESET}")
        print(f"{TerminalUI.RED}✗ Failed: {len(failed)}{TerminalUI.RESET}")

        if successful:
            print(f"\n{TerminalUI.BOLD}Generated Notebooks:{TerminalUI.RESET}")
            print(f"{'─' * 70}")
            for result in successful:
                print(f"  {TerminalUI.GREEN}✓{TerminalUI.RESET} {result.branch_name}")
                print(f"    {TerminalUI.DIM}→ {result.notebook_path}{TerminalUI.RESET}")

        if failed:
            print(f"\n{TerminalUI.BOLD}Failed Branches:{TerminalUI.RESET}")
            print(f"{'─' * 70}")
            for result in failed:
                print(f"  {TerminalUI.RED}✗{TerminalUI.RESET} {result.branch_name}")
                if result.error:
                    print(f"    {TerminalUI.RED}Error: {result.error[:100]}{TerminalUI.RESET}")

        print(f"\n{TerminalUI.BOLD}{'─' * 70}{TerminalUI.RESET}")
        print(f"\n{TerminalUI.DIM}Open notebooks with: marimo edit <notebook_path>{TerminalUI.RESET}")

    def run(self):
        """Main entry point for the CLI."""
        try:
            # Print banner
            TerminalUI.clear_screen()
            TerminalUI.print_banner()

            # Find and select CSV files
            csv_files = self.find_csv_files()
            selected_files = self.select_files(csv_files)

            if not selected_files:
                print(f"\n{TerminalUI.RED}No files selected. Exiting.{TerminalUI.RESET}")
                return

            print(f"\n{TerminalUI.GREEN}Selected files:{TerminalUI.RESET}")
            for f in selected_files:
                print(f"  • {f}")

            # Get research question
            question = self.get_research_question()
            if not question:
                print(f"\n{TerminalUI.RED}No question provided. Exiting.{TerminalUI.RESET}")
                return

            # Initialize orchestrator
            print(f"\n{TerminalUI.CYAN}Initializing research orchestrator...{TerminalUI.RESET}")
            try:
                self.orchestrator = ResearchOrchestrator(
                    csv_files=selected_files,
                    output_dir="output",
                )
            except ValueError as e:
                print(f"\n{TerminalUI.RED}Error: {e}{TerminalUI.RESET}")
                print(f"{TerminalUI.DIM}Please set ANTHROPIC_API_KEY environment variable.{TerminalUI.RESET}")
                return

            # Decompose the research question
            print(f"\n{TerminalUI.BOLD}Analyzing research question...{TerminalUI.RESET}")
            print(f"{TerminalUI.DIM}Question: {question}{TerminalUI.RESET}")

            branches = self.orchestrator.decompose_research_question(
                question=question,
                status_callback=self._status_callback,
            )

            # Display the research plan
            self.display_branches(branches)

            # Confirm execution
            print(f"\n{TerminalUI.BOLD}Ready to execute {len(branches)} research branches.{TerminalUI.RESET}")
            confirm = input(f"{TerminalUI.DIM}Press Enter to continue or 'q' to quit: {TerminalUI.RESET}").strip()
            if confirm.lower() == 'q':
                print(f"\n{TerminalUI.YELLOW}Cancelled by user.{TerminalUI.RESET}")
                return

            # Execute branches
            print(f"\n{TerminalUI.BOLD}Executing research branches...{TerminalUI.RESET}")

            results = self.orchestrator.execute_branches_parallel(
                branches=branches,
                progress_callback=self._progress_callback,
            )

            # Display final progress
            self.display_progress(branches, running=False)

            # Display results
            self.display_results(results)

        except KeyboardInterrupt:
            print(f"\n\n{TerminalUI.YELLOW}Interrupted by user.{TerminalUI.RESET}")
            sys.exit(0)
        except Exception as e:
            print(f"\n{TerminalUI.RED}Error: {e}{TerminalUI.RESET}")
            raise


def main():
    """Entry point for the CLI."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
