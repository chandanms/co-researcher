"""
Simple MCP client tests for marimo MCP server.
Run with: pytest tests/test_mcp_client.py -v
Requires: marimo notebook running with --mcp flag
"""

import asyncio
import json
from typing import List, Optional

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

MCP_SERVER_URL = "http://localhost:2718/mcp/server"


# Pydantic models matching marimo's actual structure
class SummaryInfo(BaseModel):
    total_notebooks: int
    active_connections: int


class MarimoNotebookInfo(BaseModel):
    name: str
    path: str
    session_id: str


class GetActiveNotebooksData(BaseModel):
    summary: SummaryInfo
    notebooks: List[MarimoNotebookInfo]


class GetActiveNotebooksOutput(BaseModel):
    status: str
    data: GetActiveNotebooksData
    next_steps: Optional[List[str]] = None


async def call_tool(session: ClientSession, tool_name: str, arguments: dict = None):
    """Call an MCP tool and return the parsed result."""
    if arguments is None:
        arguments = {}

    result = await session.call_tool(tool_name, arguments)

    text_content = None
    if hasattr(result, "content"):
        for content in result.content:
            if hasattr(content, "text"):
                text_content = content.text
                if text_content.startswith("Error executing tool"):
                    raise RuntimeError(text_content)

    if hasattr(result, "isError") and result.isError:
        raise RuntimeError("Tool returned error")

    if not text_content:
        raise RuntimeError("No content returned")

    return text_content


async def get_session_and_cell():
    """Helper to get session_id and cell_id from active notebook."""
    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await call_tool(session, "get_active_notebooks", {"args": {}})
            response = GetActiveNotebooksOutput.model_validate_json(result)

            if not response.data.notebooks:
                return None, None

            session_id = response.data.notebooks[0].session_id

            result = await call_tool(
                session,
                "get_lightweight_cell_map",
                {"args": {"session_id": session_id, "preview_lines": 1}},
            )
            data = json.loads(result)
            cells = data.get("cells", [])
            cell_id = cells[0].get("cell_id") if cells else None

            return session_id, cell_id


def test_list_tools():
    """Test that all expected tools are available."""

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = [tool.name for tool in tools.tools]

                expected_tools = [
                    "get_active_notebooks",
                    "get_lightweight_cell_map",
                    "get_tables_and_variables",
                    "get_notebook_errors",
                    "lint_notebook",
                    "get_marimo_rules",
                    "get_database_tables",
                    "get_cell_runtime_data",
                    "get_cell_outputs",
                ]
                for tool in expected_tools:
                    assert tool in tool_names, f"Expected tool '{tool}' not found"

    asyncio.run(run())


def test_get_active_notebooks():
    """Test get_active_notebooks returns valid response."""

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(session, "get_active_notebooks", {"args": {}})
                response = GetActiveNotebooksOutput.model_validate_json(result)
                assert response.status == "success"
                assert response.data.summary.total_notebooks >= 0

    asyncio.run(run())


def test_get_marimo_rules():
    """Test get_marimo_rules returns valid response."""

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(session, "get_marimo_rules", {"args": {}})
                data = json.loads(result)
                assert data.get("status") == "success"
                assert "rules_content" in data

    asyncio.run(run())


def test_get_lightweight_cell_map():
    """Test get_lightweight_cell_map returns valid response."""
    session_id, _ = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(
                    session,
                    "get_lightweight_cell_map",
                    {"args": {"session_id": session_id, "preview_lines": 3}},
                )
                data = json.loads(result)
                assert data.get("status") == "success"
                assert "cells" in data

    asyncio.run(run())


def test_get_tables_and_variables():
    """Test get_tables_and_variables returns valid response."""
    session_id, _ = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(
                    session,
                    "get_tables_and_variables",
                    {"args": {"session_id": session_id, "variable_names": []}},
                )
                data = json.loads(result)
                assert data.get("status") == "success"

    asyncio.run(run())


def test_get_notebook_errors():
    """Test get_notebook_errors returns valid response."""
    session_id, _ = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(
                    session,
                    "get_notebook_errors",
                    {"args": {"session_id": session_id}},
                )
                data = json.loads(result)
                assert data.get("status") == "success"

    asyncio.run(run())


def test_lint_notebook():
    """Test lint_notebook returns valid response."""
    session_id, _ = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(
                    session,
                    "lint_notebook",
                    {"args": {"session_id": session_id}},
                )
                data = json.loads(result)
                assert data.get("status") == "success"

    asyncio.run(run())


def test_get_database_tables():
    """Test get_database_tables returns a response (may error if no DB configured)."""
    session_id, _ = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_database_tables",
                    {"args": {"session_id": session_id}},
                )
                # Tool should return something (success or expected error)
                assert result.content is not None

    asyncio.run(run())


def test_get_cell_runtime_data():
    """Test get_cell_runtime_data returns valid response."""
    session_id, cell_id = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")
    if not cell_id:
        pytest.skip("No cells found")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(
                    session,
                    "get_cell_runtime_data",
                    {"args": {"session_id": session_id, "cell_ids": [cell_id]}},
                )
                data = json.loads(result)
                assert data.get("status") == "success"

    asyncio.run(run())


def test_get_cell_outputs():
    """Test get_cell_outputs returns valid response."""
    session_id, cell_id = asyncio.run(get_session_and_cell())
    if not session_id:
        pytest.skip("No active notebooks")
    if not cell_id:
        pytest.skip("No cells found")

    async def run():
        async with streamable_http_client(MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await call_tool(
                    session,
                    "get_cell_outputs",
                    {"args": {"session_id": session_id, "cell_ids": [cell_id]}},
                )
                data = json.loads(result)
                assert data.get("status") == "success"

    asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
