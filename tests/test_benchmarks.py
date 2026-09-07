"""Tests for delta_search.benchmarks module."""

from __future__ import annotations

from delta_search.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    generate_complete_graph,
    generate_grid_graph,
    generate_random_graph,
)


class TestGraphGenerators:
    def test_complete_graph_nodes(self) -> None:
        g = generate_complete_graph(5)
        assert g.num_nodes == 5

    def test_complete_graph_edges(self) -> None:
        g = generate_complete_graph(5)
        assert g.num_edges == 10

    def test_complete_graph_single_node(self) -> None:
        g = generate_complete_graph(1)
        assert g.num_nodes == 1
        assert g.num_edges == 0

    def test_grid_graph_structure(self) -> None:
        g = generate_grid_graph(2, 3)
        assert g.num_nodes == 6
        assert g.num_edges == 7  # 2*3-1 horizontal + 3-1 vertical

    def test_grid_graph_1x1(self) -> None:
        g = generate_grid_graph(1, 1)
        assert g.num_nodes == 1
        assert g.num_edges == 0

    def test_random_graph_nodes(self) -> None:
        g = generate_random_graph(10, seed=42)
        assert g.num_nodes == 10

    def test_random_graph_reproducible(self) -> None:
        g1 = generate_random_graph(10, seed=42)
        g2 = generate_random_graph(10, seed=42)
        assert g1.num_edges == g2.num_edges

    def test_random_graph_zero_prob(self) -> None:
        g = generate_random_graph(10, edge_prob=0.0, seed=42)
        assert g.num_edges == 0

    def test_random_graph_one_prob(self) -> None:
        g = generate_random_graph(10, edge_prob=1.0, seed=42)
        assert g.num_edges == 45


class TestBenchmarkSuite:
    def test_generate_cases(self) -> None:
        suite = BenchmarkSuite(sizes=[5, 10])
        cases = suite.generate_cases()
        assert len(cases) == 12  # 6 problems * 2 sizes

    def test_case_names(self) -> None:
        suite = BenchmarkSuite(sizes=[5])
        cases = suite.generate_cases()
        names = [c.name for c in cases]
        assert "mps_complete_n5" in names
        assert "mcds_grid_2x2" in names
        assert "mwis_random_n5" in names
        assert "pcvc_random_n5" in names
        assert "uflp_random_n5" in names
        assert "mwst_random_n5" in names

    def test_run_single_case(self) -> None:
        suite = BenchmarkSuite(max_iterations=5)
        case = suite.generate_cases()[0]
        result = suite.run_case(case)
        assert isinstance(result, BenchmarkResult)
        assert result.case_name == case.name
        assert result.graph_nodes > 0

    def test_run_all_cases(self) -> None:
        suite = BenchmarkSuite(sizes=[5], max_iterations=5)
        results = suite.run()
        assert len(results) == 6  # 6 problems

    def test_print_table(self) -> None:
        results = [
            BenchmarkResult(
                case_name="test_case",
                problem_class="TestProblem",
                graph_nodes=10,
                graph_edges=15,
                objective=5.0,
                iterations=3,
                evaluations=20,
                elapsed_ms=1.23,
            ),
        ]
        table = BenchmarkSuite.print_table(results)
        assert "test_case" in table
        assert "10" in table

    def test_objectives_are_numeric(self) -> None:
        suite = BenchmarkSuite(sizes=[5], max_iterations=5)
        results = suite.run()
        for r in results:
            assert isinstance(r.objective, float)
            assert isinstance(r.elapsed_ms, float)
