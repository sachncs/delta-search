"""Tests for delta_search.io module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from delta_search.graph import Graph
from delta_search.io import load_graph, save_graph


class TestSaveLoadGraph:
    def test_roundtrip(self) -> None:
        g = Graph[int]()
        g.add_node(1, label="a")
        g.add_node(2)
        g.add_edge(1, 2, weight=3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_graph(g, path)
            g2 = load_graph(path)

        assert g2.num_nodes == 2
        assert g2.num_edges == 1
        assert g2.node_data(1) == {"label": "a"}
        assert g2.edge_data(1, 2) == {"weight": 3.0}

    def test_empty_graph(self) -> None:
        g = Graph[int]()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            save_graph(g, path)
            g2 = load_graph(path)

        assert g2.num_nodes == 0
        assert g2.num_edges == 0

    def test_json_format(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "format.json"
            save_graph(g, path)
            with open(path) as f:
                data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_graph("/nonexistent/path.json")

    def test_malformed_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")
        with pytest.raises(json.JSONDecodeError):
            load_graph(bad_file)

    def test_missing_node_id(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "no_id.json"
        bad_file.write_text('{"nodes": [{"label": "a"}], "edges": []}')
        with pytest.raises(KeyError):
            load_graph(bad_file)

    def test_missing_edge_source(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "no_source.json"
        bad_file.write_text('{"nodes": [{"id": 1}], "edges": [{"target": 2}]}')
        with pytest.raises(KeyError):
            load_graph(bad_file)

    def test_missing_edge_target(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "no_target.json"
        bad_file.write_text('{"nodes": [{"id": 1}], "edges": [{"source": 1}]}')
        with pytest.raises(KeyError):
            load_graph(bad_file)

    def test_empty_arrays(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text('{"nodes": [], "edges": []}')
        g = load_graph(empty_file)
        assert g.num_nodes == 0
        assert g.num_edges == 0

    def test_roundtrip_no_attributes(self, tmp_path: Path) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        path = tmp_path / "no_attrs.json"
        save_graph(g, path)
        g2 = load_graph(path)
        assert g2.num_nodes == 3
        assert g2.num_edges == 2
        assert g2.node_data(1) == {}
        assert g2.edge_data(1, 2) == {}

    def test_original_not_mutated(self, tmp_path: Path) -> None:
        original_data = {
            "nodes": [{"id": 1, "x": 10}, {"id": 2}],
            "edges": [{"source": 1, "target": 2, "w": 5}],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(original_data))
        load_graph(path)
        with open(path) as f:
            reloaded = json.load(f)
        assert reloaded == original_data

    def test_not_a_dict_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "list.json"
        bad_file.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="Expected a JSON object"):
            load_graph(bad_file)

    def test_missing_nodes_and_edges_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "empty_obj.json"
        bad_file.write_text('{"foo": "bar"}')
        with pytest.raises(ValueError, match="must contain"):
            load_graph(bad_file)

    def test_nodes_only(self, tmp_path: Path) -> None:
        f = tmp_path / "nodes_only.json"
        f.write_text('{"nodes": [{"id": 1}, {"id": 2}]}')
        g = load_graph(f)
        assert g.num_nodes == 2
        assert g.num_edges == 0

    def test_edges_only(self, tmp_path: Path) -> None:
        f = tmp_path / "edges_only.json"
        f.write_text('{"edges": [{"source": 1, "target": 2}]}')
        g = load_graph(f)
        assert g.num_nodes == 2
        assert g.num_edges == 1

    def test_non_dict_node_entry_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad_node.json"
        bad_file.write_text('{"nodes": ["not a dict"]}')
        with pytest.raises(ValueError, match="must be a dict"):
            load_graph(bad_file)

    def test_non_dict_edge_entry_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad_edge.json"
        bad_file.write_text('{"edges": ["not a dict"]}')
        with pytest.raises(ValueError, match="must be a dict"):
            load_graph(bad_file)
