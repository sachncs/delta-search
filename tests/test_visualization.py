"""Tests for delta_search.visualization module."""

from __future__ import annotations

import json
import pathlib

from delta_search.graph import Graph
from delta_search.visualization import export_solution_graph, solution_summary


class TestExportSolutionGraph:
    def test_creates_file(self, tmp_path: object) -> None:
        path = pathlib.Path(str(tmp_path)) / "solution.json"
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        sub = Graph[int].from_edges([(1, 2)])
        export_solution_graph(g, sub, path)
        assert path.exists()

    def test_file_content_structure(self, tmp_path: object) -> None:
        path = pathlib.Path(str(tmp_path)) / "solution.json"
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        sub = Graph[int].from_edges([(1, 2)])
        export_solution_graph(g, sub, path)
        with open(path) as f:
            data = json.load(f)
        assert "input_graph" in data
        assert "solution" in data
        assert data["input_graph"]["num_nodes"] == 3
        assert len(data["solution"]["edges"]) == 1

    def test_with_metadata(self, tmp_path: object) -> None:
        path = pathlib.Path(str(tmp_path)) / "solution.json"
        g = Graph[int].from_edges([(1, 2)])
        sub = Graph[int](nodes=[1])
        export_solution_graph(g, sub, path, metadata={"objective": 5.0})
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert data["metadata"]["objective"] == 5.0


class TestSolutionSummary:
    def test_summary_keys(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        summary = solution_summary(g)
        assert "num_nodes" in summary
        assert "num_edges" in summary
        assert "nodes" in summary
        assert "edges" in summary

    def test_summary_values(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        summary = solution_summary(g)
        assert summary["num_nodes"] == 3
        assert summary["num_edges"] == 2
        assert len(summary["nodes"]) == 3
