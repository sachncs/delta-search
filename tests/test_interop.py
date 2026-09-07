"""Tests for delta_search.interop module."""

from __future__ import annotations

import importlib.util

import pytest

from delta_search.graph import Graph

has_networkx = importlib.util.find_spec("networkx") is not None
skip_no_networkx = pytest.mark.skipif(not has_networkx, reason="NetworkX not installed")


@skip_no_networkx
class TestNetworkXInterop:
    def test_from_networkx(self) -> None:
        import networkx

        from delta_search.interop import from_networkx

        nxg = networkx.Graph()
        nxg.add_edge(1, 2, weight=3.0)
        nxg.add_edge(2, 3)
        nxg.add_node(4, label="d")

        g = from_networkx(nxg)
        assert g.num_nodes == 4
        assert g.num_edges == 2
        assert g.has_edge(1, 2)
        assert g.edge_data(1, 2) == {"weight": 3.0}
        assert g.node_data(4) == {"label": "d"}

    def test_to_networkx(self) -> None:
        from delta_search.interop import to_networkx

        g = Graph[int]()
        g.add_edge(1, 2, weight=3.0)
        g.add_edge(2, 3)
        g.add_node(4, label="d")

        nxg = to_networkx(g)
        assert nxg.number_of_nodes() == 4
        assert nxg.number_of_edges() == 2
        assert nxg[1][2]["weight"] == 3.0
        assert nxg.nodes[4]["label"] == "d"

    def test_roundtrip(self) -> None:
        import networkx

        from delta_search.interop import from_networkx, to_networkx

        nxg = networkx.complete_graph(5)
        g = from_networkx(nxg)
        nxg2 = to_networkx(g)

        assert set(nxg.nodes()) == set(nxg2.nodes())
        assert set(nxg.edges()) == set(nxg2.edges())

    def test_type_check(self) -> None:
        from delta_search.interop import from_networkx

        with pytest.raises(TypeError, match="Expected a networkx.Graph"):
            from_networkx("not a graph")
