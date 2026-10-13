#!/usr/bin/env python3
"""Generate the on-brand visual assets for the delta-search landing site.

The graphs and convergence curves are produced by running the package's own
solvers on real instances, then rendered in the site's visual language.

Outputs (into site/assets/):
    screenshot-graph.png       -- solved Maximum Planar Subgraph instance
    screenshot-convergence.png -- objective-as-a-function-of-iteration
    og.png                     -- social share card (1200x630)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import networkx as nx
except ImportError:  # pragma: no cover - asset generation has an optional path
    nx = None

from delta_search import (
    Action,
    DefaultState,
    DeltaResult,
    EarlyTerminationCondition,
    Graph,
    GreedySolver,
    MaximumPlanarSubgraphProblem,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- brand palette ---------------------------------------------------------
BG = "#0B0B0F"
BG_SOFT = "#14141C"
INK = "#F5F5F7"
MUTED = "#9A9BA6"
VIOLET = "#5367E8"
CYAN = "#B8F36B"
EDGE_INPUT = "#2A2A34"
NODE_INPUT = "#202028"
EDGE_SOLUTION = "#88A0FF"
NODE_SOLUTION = "#C5F36C"

plt.rcParams.update(
    {
        "font.family": "Helvetica Neue",
        "font.size": 13,
        "text.color": INK,
        "axes.edgecolor": EDGE_INPUT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


# --- 1. build and solve a real instance ------------------------------------
class PlanarDemoProblem(MaximumPlanarSubgraphProblem):
    """Maximum Planar Subgraph with the vertex set fixed up front.

    The stock implementation seeds an empty subgraph for this monotone
    (edge-add-only) problem; for the demo we seed all vertices and let
    the solver select a maximum-edge subgraph screened by the repository's
    Euler-bound feasibility heuristic.
    """

    def evaluate_initial_state(self, graph: Graph[int]) -> DefaultState[int]:
        seeded = Graph[int]()
        for node in graph.nodes:
            seeded.add_node(node)
        return DefaultState(graph=seeded)


def build_input_graph(
    seed: int = 1, nodes: int = 62, density: float = 0.12
) -> Graph[int]:
    rng = random.Random(seed)
    g = Graph[int]()
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if rng.random() < density:
                g.add_edge(i, j)
    return g


class ObjectiveObserver:
    def __init__(self) -> None:
        self.objectives: list[float] = []
        self.iterations: list[int] = []

    def on_action_evaluated(
        self, action: Action, delta: DeltaResult, elapsed_ms: float
    ) -> None:
        pass

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        pass

    def on_iteration_complete(
        self, iteration: int, best_action: Action | None, objective: float
    ) -> None:
        self.iterations.append(iteration)
        self.objectives.append(objective)


def solve_mps(graph: Graph[int]) -> tuple[Graph[int], list[float], list[int]]:
    observer = ObjectiveObserver()
    solver = GreedySolver(
        PlanarDemoProblem(graph),
        early_stop=EarlyTerminationCondition(max_iterations=2200, stall_iterations=220),
    )
    result = solver.solve(observer=observer)
    solution = Graph[int]()
    for n in result.best_state.graph.nodes:
        solution.add_node(n)
    for u, v in result.best_state.graph.sorted_edges():
        solution.add_edge(u, v)
    return solution, observer.objectives, observer.iterations


# --- 2. simple force-directed layout ---------------------------------------
def force_layout(
    graph: Graph[int], seed: int = 3, iters: int = 900
) -> dict[int, tuple[float, float]]:
    rng = random.Random(seed)
    pos = {n: (rng.random(), rng.random()) for n in graph.nodes}
    k = 1.2 / math.sqrt(max(1, graph.num_nodes))
    for _ in range(iters):
        disp = {n: [0.0, 0.0] for n in graph.nodes}
        nodes = list(graph.nodes)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                d = max(math.hypot(dx, dy), 1e-4)
                rep = k * k / d
                disp[a][0] += dx / d * rep
                disp[a][1] += dy / d * rep
                disp[b][0] -= dx / d * rep
                disp[b][1] -= dy / d * rep
        for a in nodes:
            for b in graph.neighbors(a):
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                d = max(math.hypot(dx, dy), 1e-4)
                attr = d * d / k
                disp[a][0] -= dx / d * attr / 2
                disp[a][1] -= dy / d * attr / 2
                disp[b][0] += dx / d * attr / 2
                disp[b][1] += dy / d * attr / 2
        for n in nodes:
            d = max(math.hypot(*disp[n]), 1e-4)
            step = min(d, 0.06)
            pos[n] = (
                pos[n][0] + disp[n][0] / d * step,
                pos[n][1] + disp[n][1] / d * step,
            )
    xs = [pos[n][0] for n in graph.nodes]
    ys = [pos[n][1] for n in graph.nodes]
    ymin, ymax = min(ys), max(ys)
    return {
        n: (
            (pos[n][0] - min(xs)) / (max(xs) - min(xs)),
            1 - (pos[n][1] - ymin) / (ymax - ymin),
        )
        for n in graph.nodes
    }


def representative_edges(
    graph: Graph[int], extra_edges: int = 42
) -> set[frozenset[int]]:
    """Return a readable, connected sample of a solved graph's edges.

    A 62-node planar solution can contain 180 edges. Drawing all of them makes
    the marketing image a hairball, so the image shows a spanning forest plus a
    deterministic sample of additional retained edges. The solve itself is not
    truncated; this is only a presentation view.
    """
    selected: set[frozenset[int]] = set()
    visited: set[int] = set()

    for root in sorted(graph.nodes):
        if root in visited:
            continue
        visited.add(root)
        queue = [root]
        while queue:
            node = queue.pop(0)
            for neighbor in sorted(graph.neighbors(node)):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                selected.add(frozenset((node, neighbor)))

    remaining = [
        frozenset((u, v))
        for u, v in graph.sorted_edges()
        if frozenset((u, v)) not in selected
    ]
    selected.update(remaining[:extra_edges])
    return selected


def solution_layout(
    solution: Graph[int], seed: int = 3
) -> dict[int, tuple[float, float]]:
    """Prefer a planar embedding, with a deterministic force fallback."""
    if nx is not None:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(solution.nodes)
        nx_graph.add_edges_from(solution.sorted_edges())
        if nx.check_planarity(nx_graph)[0]:
            raw = nx.planar_layout(nx_graph, scale=1.0)
            xs = [float(raw[node][0]) for node in solution.nodes]
            ys = [float(raw[node][1]) for node in solution.nodes]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            return {
                node: (
                    (float(raw[node][0]) - x_min) / max(x_max - x_min, 1e-9),
                    1 - (float(raw[node][1]) - y_min) / max(y_max - y_min, 1e-9),
                )
                for node in solution.nodes
            }
    return force_layout(solution, seed=seed, iters=1200)


# --- 3. render the solved graph --------------------------------------------
def render_graph(
    graph: Graph[int], solution: Graph[int], pos: dict[int, tuple[float, float]]
) -> None:
    w, h = 1040, 660
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sol_nodes = set(solution.nodes)
    sol_edges = {frozenset((u, v)) for u, v in solution.sorted_edges()}
    display_edges = representative_edges(solution)
    candidate_edges = [
        frozenset((u, v))
        for u, v in graph.sorted_edges()
        if frozenset((u, v)) not in sol_edges
    ]
    display_edges.update(candidate_edges[:24])

    for u in graph.nodes:
        if u not in pos:
            continue
        x, y = pos[u]
        glow = Circle((x, y), 0.018, facecolor=NODE_SOLUTION, alpha=0.08, zorder=1)
        ax.add_patch(glow)

    for u in graph.nodes:
        if u not in pos:
            continue
        for v in graph.neighbors(u):
            if v < u:
                continue
            if u not in pos:
                continue
            edge_key = frozenset((u, v))
            if edge_key not in display_edges:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            in_sol = edge_key in sol_edges
            color = EDGE_SOLUTION if in_sol else EDGE_INPUT
            alpha = 0.78 if in_sol else 0.22
            lw = 1.5 if in_sol else 0.7
            zorder = 4 if in_sol else 0
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                alpha=alpha,
                lw=lw,
                zorder=zorder,
                solid_capstyle="round",
            )

    for n in graph.nodes:
        if n not in pos:
            continue
        x, y = pos[n]
        in_sol = n in sol_nodes
        fc = NODE_SOLUTION if in_sol else NODE_INPUT
        zorder = 5 if in_sol else 1
        ax.scatter(
            [x], [y], s=60 if in_sol else 26, c=fc, edgecolors="none", zorder=zorder
        )
        if in_sol:
            ax.scatter(
                [x],
                [y],
                s=150,
                facecolors="none",
                edgecolors=VIOLET,
                linewidths=1.1,
                zorder=6,
            )

    plt.savefig(
        ASSETS / "screenshot-graph.png",
        dpi=150,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


# --- 4. render the convergence chart ----------------------------------------
def render_convergence(iterations: list[int], objectives: list[float]) -> None:
    w, h = 1040, 400
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.12)

    xs = iterations
    ys = objectives
    ax.plot(xs, ys, color=VIOLET, lw=3, zorder=3, solid_capstyle="round")
    ax.plot(xs, ys, color=VIOLET, lw=10, alpha=0.12, zorder=2, solid_capstyle="round")
    ax.fill_between(xs, ys, min(ys), color=VIOLET, alpha=0.06, zorder=1)

    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        s=90,
        color=CYAN,
        zorder=4,
        edgecolors="#0B0B0F",
        linewidths=2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Iteration", labelpad=12)
    ax.set_ylabel("Objective", labelpad=12)
    ax.grid(axis="y", color="#1C1C26", linewidth=1, alpha=0.8)

    plt.savefig(
        ASSETS / "screenshot-convergence.png",
        dpi=150,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


# --- 5. compose the social card ---------------------------------------------
def smooth_gradient(
    size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> Image.Image:
    w, h = size
    base = Image.new("RGB", size)
    draw = ImageDraw.Draw(base)
    for y in range(h):
        t = y / (h - 1)
        draw.line(
            [(0, y), (w, y)],
            fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
        )
    return base


def radial_glow(
    size: tuple[int, int],
    center: tuple[int, int],
    radius: float,
    color: tuple[int, int, int],
    strength: float = 90,
) -> Image.Image:
    x, y = center
    w, h = size
    layer = Image.new("L", size, 0)
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = x - radius, y - radius, x + radius, y + radius
    draw.ellipse([x0, y0, x1, y1], fill=255)
    layer = layer.filter(ImageFilter.GaussianBlur(radius * 0.18))
    glow = Image.new("RGBA", size, (*color, 0))
    glow.putalpha(layer.point(lambda p: int(p * strength / 255)))
    return glow


def make_og() -> None:
    width, height = 1200, 630
    bg = Image.new("RGB", (width, height), (11, 11, 15))
    blue = radial_glow((width, height), (width - 300, 140), 560, (83, 103, 232), 95)
    lime = radial_glow((width, height), (160, height - 120), 460, (197, 243, 108), 34)
    bg.paste(blue, (0, 0), blue)
    bg.paste(lime, (0, 0), lime)

    draw = ImageDraw.Draw(bg, "RGBA")

    font_ultra = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc", 60, index=5
    )
    font_bold = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc", 54, index=1
    )
    font_med = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc", 30, index=1
    )
    font_light = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc", 24, index=0
    )

    def draw_delta(
        draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float
    ) -> None:
        stripes = 48
        for i in range(stripes):
            t = i / (stripes - 1)
            inset = t * size / 3
            x0, y0 = cx - size + inset, cy + size - inset
            x1, y1 = cx + size - inset, cy + size - inset
            x2 = cx
            y2 = cy - size + inset
            top = (
                int(136 + (197 - 136) * t),
                int(168 + (243 - 168) * t),
                int(255 + (108 - 255) * t),
            )
            draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=(*top, 255))

    draw_delta(draw, 250, 330, 138)
    draw.line([(120, 300), (120, 368)], fill=(136, 168, 255, 255), width=6)

    draw.text((420, 190), "delta-search", font=font_bold, fill=(245, 245, 247, 255))
    draw.text(
        (420, 268),
        "Heuristic graph search",
        font=font_ultra,
        fill=(245, 245, 247, 255),
    )
    draw.text(
        (420, 360),
        "Alpha \u00b7 local install \u00b7 6 problem types",
        font=font_light,
        fill=(154, 155, 166, 255),
    )

    pill = (83, 103, 232, 255)
    px1, py1, px2, py2 = 420, 452, 620, 508
    draw.rounded_rectangle([px1, py1, px2, py2], radius=28, fill=pill)
    draw.text((454, 466), "Read the docs", font=font_med, fill=(245, 245, 247, 255))

    bg.save(ASSETS / "og.png")
    bg.save(Path(__file__).resolve().parents[2] / ".github/assets/social-preview.png")


def main() -> None:
    graph = build_input_graph()
    solution, objectives, iterations = solve_mps(graph)
    # Layout the retained solution first. The input graph can be dense;
    # using it as the layout makes the result read as a hairball instead of a
    # solution. Candidate edges still render faintly on the same coordinates.
    pos = solution_layout(solution)
    render_graph(graph, solution, pos)
    render_convergence(iterations, objectives)
    make_og()
    print(
        f"solved: {graph.num_nodes} nodes / {graph.num_edges} edges -> "
        f"{solution.num_edges} retained edges (Euler cap {solve_cap(solution.num_nodes)})"
    )
    print(
        f"convergence: {len(iterations)} iterations tracked, final objective {objectives[-1]:.1f}"
    )
    print("wrote:", ", ".join(p.name for p in sorted(ASSETS.iterdir())))


def solve_cap(num_nodes: int) -> int:
    return max(0, 3 * num_nodes - 6)


if __name__ == "__main__":
    main()
