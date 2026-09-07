"""Tests for delta_search.cli module."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import pytest

from delta_search.cli import build_parser, cmd_validate, import_problem, main
from delta_search.graph import Graph
from delta_search.io import save_graph

if TYPE_CHECKING:
    from pathlib import Path


class TestImportProblem:
    def test_valid_short_name(self) -> None:
        cls = import_problem("mps")
        assert cls.__name__ == "MaximumPlanarSubgraphProblem"

    def test_all_short_names(self) -> None:
        for name in ["mps", "mcds", "mwis", "pcvc", "uflp", "mwst"]:
            cls = import_problem(name)
            assert isinstance(cls, type)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown problem"):
            import_problem("nonexistent")

    def test_not_a_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown problem"):
            import_problem("not_a_real_problem_name")


class TestBuildParser:
    def test_returns_parser(self) -> None:
        parser = build_parser()
        assert parser.prog == "delta-search"

    def test_solve_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--problem",
                "mps",
                "--graph",
                "input.json",
            ]
        )
        assert args.command == "solve"
        assert args.problem == "mps"
        assert args.graph == "input.json"

    def test_validate_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validate", "--graph", "input.json"])
        assert args.command == "validate"
        assert args.graph == "input.json"

    def test_solve_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "solve",
                "--problem",
                "mps",
                "--graph",
                "input.json",
            ]
        )
        assert args.max_iterations == 1000
        assert args.max_evaluations is None
        assert args.stall is None
        assert args.output is None
        assert args.verbose is False


class TestMain:
    def test_no_command_returns_1(self) -> None:
        assert main([]) == 1

    def test_unknown_command_returns_1(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["unknown"])
        assert exc_info.value.code == 2

    def test_validate_valid_graph(self, tmp_path: Path) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        graph_file = tmp_path / "graph.json"
        save_graph(g, graph_file)
        assert cmd_validate(argparse.Namespace(graph=str(graph_file))) == 0

    def test_validate_invalid_file(self, tmp_path: Path) -> None:
        graph_file = tmp_path / "bad.json"
        graph_file.write_text("not json")
        assert cmd_validate(argparse.Namespace(graph=str(graph_file))) == 1

    def test_validate_nonexistent_file(self) -> None:
        assert cmd_validate(argparse.Namespace(graph="/nonexistent/file.json")) == 1

    def test_solve_end_to_end(self, tmp_path: Path) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        graph_file = tmp_path / "graph.json"
        save_graph(g, graph_file)
        result = main(
            [
                "solve",
                "--problem",
                "mps",
                "--graph",
                str(graph_file),
                "--max-iterations",
                "10",
            ]
        )
        assert result == 0

    def test_solve_with_output_file(self, tmp_path: Path) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        graph_file = tmp_path / "graph.json"
        output_file = tmp_path / "result.json"
        save_graph(g, graph_file)
        result = main(
            [
                "solve",
                "--problem",
                "mps",
                "--graph",
                str(graph_file),
                "--output",
                str(output_file),
                "--max-iterations",
                "5",
            ]
        )
        assert result == 0
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "objective" in data
        assert "iterations" in data

    def test_solve_verbose(self, tmp_path: Path) -> None:
        g = Graph[int].from_edges([(1, 2)])
        graph_file = tmp_path / "graph.json"
        save_graph(g, graph_file)
        result = main(
            [
                "solve",
                "--problem",
                "mps",
                "--graph",
                str(graph_file),
                "--verbose",
                "--max-iterations",
                "3",
            ]
        )
        assert result == 0

    def test_solve_output_visible(self, tmp_path: Path, caplog) -> None:
        caplog.set_level("INFO")
        g = Graph[int].from_edges([(1, 2)])
        graph_file = tmp_path / "graph.json"
        save_graph(g, graph_file)
        main(
            [
                "solve",
                "--problem",
                "mps",
                "--graph",
                str(graph_file),
                "--max-iterations",
                "3",
            ]
        )
        assert "objective" in caplog.text

    def test_solve_bad_graph_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            main(
                [
                    "solve",
                    "--problem",
                    "mps",
                    "--graph",
                    str(bad_file),
                ]
            )

    def test_solve_bad_problem_name(self, tmp_path: Path) -> None:
        g = Graph[int].from_edges([(1, 2)])
        graph_file = tmp_path / "graph.json"
        save_graph(g, graph_file)
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "solve",
                    "--problem",
                    "nonexistent",
                    "--graph",
                    str(graph_file),
                ]
            )
        assert exc_info.value.code == 2
