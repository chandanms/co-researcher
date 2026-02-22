"""
Co-Researcher - Parallel Research Assistant

A CLI tool that uses marimo notebooks and LLM-generated code
to parallelize research tasks.

Usage:
    python main.py
"""

from cli import CLI


def main():
    """Launch the co-researcher CLI."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
