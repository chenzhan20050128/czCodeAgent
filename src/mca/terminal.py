"""Small, dependency-free terminal presentation and multiline editing helpers."""

from __future__ import annotations

import codecs
import os
import re
import sys
import termios
import tty
import unicodedata
from dataclasses import dataclass

from .approval import _escape_terminal_text
from .code_graph import CodeGraphNodeView, CodeGraphView


_RESET = "\x1b[0m"
# Deliberately muted 256-color palette. These are semantic roles, not raw
# red/yellow/green status lights, so the terminal stays readable on dark themes.
_PALETTE = {
    "info": "38;5;110",       # steel blue
    "workspace": "38;5;67",   # deep slate blue
    "model": "38;5;141",      # muted indigo
    "tool": "38;5;130",       # deep burnt amber
    "approval": "38;5;137",   # muted amber-brown
    "success": "38;5;72",     # deep teal-green
    "failure": "38;5;167",    # dusty brick
    "muted": "38;5;245",      # stone gray
    "prompt": "38;5;117",     # pale blue
}


class TerminalTheme:
    """Render a restrained ANSI theme, with a safe plain-text fallback."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)

    @classmethod
    def auto(cls, *, isatty: bool) -> TerminalTheme:
        disabled = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
        return cls(enabled=isatty and not disabled)

    def style(self, text: str, role: str) -> str:
        if not self.enabled:
            return text
        color = _PALETTE.get(role, _PALETTE["muted"])
        return f"\x1b[{color}m{text}{_RESET}"

    def label(self, text: str) -> str:
        return self.style(text, "info")


_GRAPH_STATUS = {
    "planned": ("○", "PLANNED"),
    "started": ("▶", "RUNNING"),
    "succeeded": ("✓", "SUCCEEDED"),
    "user_confirmed_success": ("✓", "CONFIRMED"),
    "failed": ("✗", "FAILED"),
    "denied": ("⊘", "DENIED"),
    "invalid_arguments": ("✗", "INVALID"),
    "unknown_tool": ("✗", "UNKNOWN_TOOL"),
    "conflict": ("✗", "CONFLICT"),
    "timed_out": ("◷", "TIMED_OUT"),
    "interrupted": ("!", "INTERRUPTED"),
    "cancelled": ("⊘", "CANCELLED"),
    "not_executed": ("⊘", "NOT_EXECUTED"),
    "upstream_failed": ("⊘", "UPSTREAM_FAILED"),
    "outcome_unknown": ("?", "OUTCOME_UNKNOWN"),
    "abandoned": ("⊘", "ABANDONED"),
    "batch_limit_exceeded": ("⊘", "BATCH_LIMIT"),
    "user_confirmed_failure": ("✗", "CONFIRMED_FAILED"),
}
_GRAPH_FAILURE_STATUSES = frozenset(
    {
        "failed",
        "denied",
        "invalid_arguments",
        "unknown_tool",
        "conflict",
        "timed_out",
        "interrupted",
        "cancelled",
        "upstream_failed",
        "outcome_unknown",
        "abandoned",
        "batch_limit_exceeded",
        "user_confirmed_failure",
    }
)
_MAX_LAYOUT_VERTICES = 512
_MAX_NODE_LABEL_WIDTH = 80


@dataclass(frozen=True)
class _LayoutVertex:
    key: str
    rank: int
    ordinal: int | None
    sort_key: tuple[int, ...]


@dataclass(frozen=True)
class _LayoutEdge:
    source: str
    target: str


@dataclass(frozen=True)
class _HorizontalLayout:
    lines: tuple[str, ...]
    width: int


def render_code_graph_plain(
    graph: CodeGraphView, *, width: int = 100, expanded: bool = True
) -> str:
    """Render a stable complete DAG without terminal control sequences."""

    if not isinstance(graph, CodeGraphView):
        raise TypeError("graph must be a CodeGraphView")
    if type(width) is not int or width < 20:
        raise ValueError("width must be at least 20")
    content_width = max(1, width - 3)
    full = _render_horizontal_layout(graph, compact=False)
    compact = _render_horizontal_layout(graph, compact=True)
    if full is not None and full.width <= content_width:
        mode = "left to right"
        diagram = full.lines
    elif compact is not None and compact.width <= content_width:
        mode = "compact"
        diagram = compact.lines
    else:
        mode = "vertical"
        diagram = _render_vertical_layout(graph, content_width)

    lines = [
        f"╭─ run_code: {_graph_text(graph.description)} {graph.status.upper()}",
        f"│  DAG · {mode}",
        "│",
    ]
    lines.extend("│  " + line for line in diagram)
    if expanded and graph.nodes:
        lines.extend(("│", "│  Details"))
        for node in graph.nodes:
            lines.append(_render_graph_detail(node))
            if node.dependency_ordinals:
                lines.extend(
                    _render_dependency_details(node.dependency_ordinals, width)
                )
            if node.result and node.status in _GRAPH_FAILURE_STATUSES:
                lines.append(f"│     {_one_line(node.result)}")
            if node.blocked_by_ordinals:
                blockers = ", ".join(f"#{value}" for value in node.blocked_by_ordinals)
                lines.append(f"│     blocked by {blockers}")
    if graph.shell_mutation_warning:
        lines.append(
            "│  warning: parallel bash + file mutation may contend for workspace resources"
        )
    lines.append(f"╰─ {_render_graph_summary(graph)}")
    return "\n".join(_fit_graph_line(line, width) for line in lines)


def render_code_graph_ansi(
    graph: CodeGraphView, *, width: int = 100, expanded: bool = True
) -> str:
    """Render the same graph content with restrained semantic coloring."""

    theme = TerminalTheme(enabled=True)
    plain_lines = render_code_graph_plain(
        graph, width=width, expanded=expanded
    ).splitlines()
    nodes = {node.ordinal: node for node in graph.nodes}
    return "\n".join(
        _style_graph_line(line, nodes, theme) for line in plain_lines
    )


def _render_graph_detail(node: CodeGraphNodeView) -> str:
    _, label = _GRAPH_STATUS.get(node.status, ("?", node.status.upper()))
    target = f"  {_graph_text(node.target)}" if node.target else ""
    elapsed = f"  {_format_elapsed(node.elapsed_ms)}" if node.elapsed_ms is not None else ""
    current = "  CURRENT" if node.is_current else ""
    approval = (
        f"  {node.approval.upper()}" if node.approval is not None else ""
    )
    return (
        f"│  #{node.ordinal} {label}{approval}{current}"
        f"  {_graph_text(node.name)}{target}{elapsed}"
    )


def _render_dependency_details(
    ordinals: tuple[int, ...], width: int
) -> list[str]:
    first_prefix = "│     after "
    continuation_prefix = "│           "
    rendered: list[str] = []
    current = first_prefix
    prefix = first_prefix
    for ordinal in ordinals:
        token = f"#{ordinal}"
        separator = "" if current == prefix else ","
        candidate = current + separator + token
        if _display_width(candidate) <= width:
            current = candidate
            continue
        if current != prefix:
            rendered.append(current)
            prefix = continuation_prefix
            current = prefix + token
            continue
        rendered.append(_fit_graph_line(candidate, width))
        prefix = continuation_prefix
        current = prefix
    if current != prefix:
        rendered.append(current)
    return rendered


def _node_label(node: CodeGraphNodeView, *, compact: bool) -> str:
    symbol, _ = _GRAPH_STATUS.get(node.status, ("?", node.status.upper()))
    token = f"{symbol} #{node.ordinal}"
    if compact:
        return f"[{token}]"
    target = f"({_graph_text(node.target)})" if node.target else ""
    inner = f"{token} {_graph_text(node.name)}{target}"
    return f"[{_fit_graph_line(inner, _MAX_NODE_LABEL_WIDTH - 2)}]"


def _node_role(node: CodeGraphNodeView) -> str:
    if node.is_current or node.status == "started":
        return "prompt"
    if node.status in {"succeeded", "user_confirmed_success"}:
        return "success"
    if node.status in {"upstream_failed", "not_executed", "denied", "cancelled"}:
        return "approval"
    if node.status in _GRAPH_FAILURE_STATUSES:
        return "failure"
    return "muted"


def _style_graph_line(
    line: str, nodes: dict[int, CodeGraphNodeView], theme: TerminalTheme
) -> str:
    if line.startswith(("╭", "╰")):
        default_role = "tool"
    elif "warning:" in line:
        default_role = "approval"
    elif line.startswith("│  Details"):
        default_role = "info"
    else:
        default_role = "muted"
    matches: list[tuple[int, int, CodeGraphNodeView]] = []
    for node in nodes.values():
        for compact in (False, True):
            label = _node_label(node, compact=compact)
            start = line.find(label)
            if start >= 0:
                matches.append((start, start + len(label), node))
                break
    matches.sort(key=lambda item: item[0])
    if not matches:
        detail = re.match(r"│  #(\d+) ", line)
        if detail is not None and int(detail.group(1)) in nodes:
            default_role = _node_role(nodes[int(detail.group(1))])
        return theme.style(line, default_role)
    parts: list[str] = []
    cursor = 0
    for start, end, node in matches:
        if start > cursor:
            parts.append(theme.style(line[cursor:start], default_role))
        parts.append(theme.style(line[start:end], _node_role(node)))
        cursor = end
    if cursor < len(line):
        parts.append(theme.style(line[cursor:], default_role))
    return "".join(parts)


def _render_horizontal_layout(
    graph: CodeGraphView, *, compact: bool
) -> _HorizontalLayout | None:
    layered = _layered_vertices(graph)
    if layered is None:
        return None
    ranks, vertices, edges = layered
    if not vertices:
        return _HorizontalLayout(("(no nodes)",), len("(no nodes)"))

    nodes = {node.ordinal: node for node in graph.nodes}
    labels = {
        vertex.key: _node_label(nodes[vertex.ordinal], compact=compact)
        for vertex in vertices.values()
        if vertex.ordinal is not None
    }
    rank_widths = {
        rank: max(
            [1]
            + [_display_width(labels[key]) for key in keys if key in labels]
        )
        for rank, keys in ranks.items()
    }
    gap_edges: dict[int, list[_LayoutEdge]] = {}
    for edge in edges:
        rank = vertices[edge.source].rank
        gap_edges.setdefault(rank, []).append(edge)

    rank_starts: dict[int, int] = {}
    cursor = 0
    last_rank = max(ranks)
    for rank in range(last_rank + 1):
        rank_starts[rank] = cursor
        if rank < last_rank:
            cursor += rank_widths[rank] + max(5, len(gap_edges.get(rank, ())) + 3)
        else:
            cursor += rank_widths[rank]

    max_layer_size = max(len(keys) for keys in ranks.values())
    positions: dict[str, tuple[int, int]] = {}
    for rank, keys in ranks.items():
        offset = max_layer_size - len(keys)
        for index, key in enumerate(keys):
            positions[key] = (rank_starts[rank], offset + index * 2)

    connections: dict[tuple[int, int], set[str]] = {}
    arrows: set[tuple[int, int]] = set()
    text_cells: dict[tuple[int, int], str | None] = {}
    for key, label in labels.items():
        x, y = positions[key]
        _place_text(text_cells, y, x, label)

    for rank in range(last_rank):
        ranked_edges = sorted(
            gap_edges.get(rank, ()),
            key=lambda edge: (
                positions[edge.source][1], positions[edge.target][1],
                vertices[edge.source].sort_key, vertices[edge.target].sort_key,
            ),
        )
        for lane_index, edge in enumerate(ranked_edges):
            source = vertices[edge.source]
            target = vertices[edge.target]
            source_x, source_y = positions[edge.source]
            target_x, target_y = positions[edge.target]
            if source.ordinal is None:
                start_x = source_x + rank_widths[rank] // 2
            else:
                start_x = source_x + _display_width(labels[edge.source])
            lane_x = rank_starts[rank] + rank_widths[rank] + 1 + lane_index
            if target.ordinal is None:
                end_x = target_x + rank_widths[rank + 1] // 2
            else:
                end_x = target_x - 2
                arrows.add((target_y, end_x))
            _paint_polyline(
                connections,
                ((start_x, source_y), (lane_x, source_y),
                 (lane_x, target_y), (end_x, target_y)),
            )

    lines = _materialize_canvas(text_cells, connections, arrows)
    width = max((_display_width(line) for line in lines), default=0)
    return _HorizontalLayout(tuple(lines), width)


def _layered_vertices(
    graph: CodeGraphView,
) -> tuple[dict[int, list[str]], dict[str, _LayoutVertex], list[_LayoutEdge]] | None:
    nodes = {node.ordinal: node for node in graph.nodes}
    indegree = {ordinal: 0 for ordinal in nodes}
    children: dict[int, list[int]] = {ordinal: [] for ordinal in nodes}
    for node in graph.nodes:
        for dependency in node.dependency_ordinals:
            if dependency not in nodes:
                continue
            indegree[node.ordinal] += 1
            children[dependency].append(node.ordinal)
    ready = sorted(ordinal for ordinal, degree in indegree.items() if degree == 0)
    order: list[int] = []
    ranks_by_ordinal: dict[int, int] = {}
    while ready:
        ordinal = ready.pop(0)
        order.append(ordinal)
        node = nodes[ordinal]
        dependencies = [
            dependency for dependency in node.dependency_ordinals
            if dependency in ranks_by_ordinal
        ]
        ranks_by_ordinal[ordinal] = (
            max(ranks_by_ordinal[item] for item in dependencies) + 1
            if dependencies else 0
        )
        for child in sorted(children[ordinal]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        return None

    vertices: dict[str, _LayoutVertex] = {}
    ranks: dict[int, list[str]] = {}
    for ordinal in order:
        key = f"n:{ordinal}"
        rank = ranks_by_ordinal[ordinal]
        vertices[key] = _LayoutVertex(key, rank, ordinal, (ordinal, ordinal, 0))
        ranks.setdefault(rank, []).append(key)

    edges: list[_LayoutEdge] = []
    for target in order:
        for source in sorted(nodes[target].dependency_ordinals):
            if source not in nodes:
                continue
            previous = f"n:{source}"
            for rank in range(ranks_by_ordinal[source] + 1, ranks_by_ordinal[target]):
                key = f"d:{source}:{target}:{rank}"
                if key not in vertices:
                    vertices[key] = _LayoutVertex(
                        key, rank, None, (source, target, rank)
                    )
                    ranks.setdefault(rank, []).append(key)
                edges.append(_LayoutEdge(previous, key))
                previous = key
            edges.append(_LayoutEdge(previous, f"n:{target}"))
            if len(vertices) > _MAX_LAYOUT_VERTICES:
                return None

    for keys in ranks.values():
        keys.sort(key=lambda key: vertices[key].sort_key)
    _reduce_crossings(ranks, vertices, edges)
    return ranks, vertices, edges


def _reduce_crossings(
    ranks: dict[int, list[str]],
    vertices: dict[str, _LayoutVertex],
    edges: list[_LayoutEdge],
) -> None:
    predecessors: dict[str, list[str]] = {key: [] for key in vertices}
    successors: dict[str, list[str]] = {key: [] for key in vertices}
    for edge in edges:
        predecessors[edge.target].append(edge.source)
        successors[edge.source].append(edge.target)
    last_rank = max(ranks, default=0)
    for _ in range(4):
        positions = {key: index for keys in ranks.values() for index, key in enumerate(keys)}
        for rank in range(1, last_rank + 1):
            ranks[rank].sort(
                key=lambda key: _barycentric_key(
                    key, predecessors[key], positions, vertices
                )
            )
        positions = {key: index for keys in ranks.values() for index, key in enumerate(keys)}
        for rank in range(last_rank - 1, -1, -1):
            ranks[rank].sort(
                key=lambda key: _barycentric_key(
                    key, successors[key], positions, vertices
                )
            )


def _barycentric_key(
    key: str, neighbors: list[str], positions: dict[str, int],
    vertices: dict[str, _LayoutVertex],
) -> tuple[float, tuple[int, ...]]:
    value = (
        sum(positions[item] for item in neighbors) / len(neighbors)
        if neighbors else float(positions[key])
    )
    return value, vertices[key].sort_key


def _render_vertical_layout(
    graph: CodeGraphView, content_width: int
) -> tuple[str, ...]:
    nodes = _topological_nodes(graph)
    if not nodes:
        return ("(no nodes)",)
    index_by_ordinal = {node.ordinal: index for index, node in enumerate(nodes)}
    raw_edges = sorted(
        (index_by_ordinal[dependency], index_by_ordinal[node.ordinal])
        for node in nodes
        for dependency in node.dependency_ordinals
        if dependency in index_by_ordinal
        and index_by_ordinal[dependency] < index_by_ordinal[node.ordinal]
    )
    lane_ends: list[int] = []
    routed: list[tuple[int, int, int]] = []
    for source, target in raw_edges:
        lane = next(
            (index for index, end in enumerate(lane_ends) if end < source),
            len(lane_ends),
        )
        if lane == len(lane_ends):
            lane_ends.append(target)
        else:
            lane_ends[lane] = target
        routed.append((source, target, lane))
    max_lanes = max(1, (content_width - 9) // 2)
    visible_lanes = min(len(lane_ends), max_lanes)
    node_x = visible_lanes * 2 + 2
    connections: dict[tuple[int, int], set[str]] = {}
    arrows: set[tuple[int, int]] = set()
    text_cells: dict[tuple[int, int], str | None] = {}
    for index, node in enumerate(nodes):
        label = _fit_graph_line(
            _node_label(node, compact=True), max(4, content_width - node_x)
        )
        _place_text(text_cells, index * 2, node_x, label)
    for source, target, raw_lane in routed:
        lane = min(raw_lane, visible_lanes - 1)
        lane_x = lane * 2
        source_y = source * 2
        target_y = target * 2
        arrow_x = node_x - 2
        arrows.add((target_y, arrow_x))
        _paint_polyline(
            connections,
            ((node_x, source_y), (lane_x, source_y),
             (lane_x, target_y), (arrow_x, target_y)),
        )
    return tuple(_materialize_canvas(text_cells, connections, arrows))


def _topological_nodes(graph: CodeGraphView) -> list[CodeGraphNodeView]:
    nodes = {node.ordinal: node for node in graph.nodes}
    indegree = {ordinal: 0 for ordinal in nodes}
    children: dict[int, list[int]] = {ordinal: [] for ordinal in nodes}
    for node in graph.nodes:
        for dependency in node.dependency_ordinals:
            if dependency in nodes:
                indegree[node.ordinal] += 1
                children[dependency].append(node.ordinal)
    ready = sorted(ordinal for ordinal, degree in indegree.items() if degree == 0)
    result: list[CodeGraphNodeView] = []
    while ready:
        ordinal = ready.pop(0)
        result.append(nodes[ordinal])
        for child in sorted(children[ordinal]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(result) != len(nodes):
        return sorted(graph.nodes, key=lambda node: node.ordinal)
    return result


def _paint_polyline(
    connections: dict[tuple[int, int], set[str]],
    points: tuple[tuple[int, int], ...],
) -> None:
    previous = points[0]
    for point in points[1:]:
        x1, y1 = previous
        x2, y2 = point
        if x1 != x2 and y1 != y2:
            raise ValueError("graph route segment must be orthogonal")
        if x1 == x2:
            step = 1 if y2 >= y1 else -1
            for y in range(y1, y2, step):
                _connect_cells(connections, (x1, y), (x1, y + step))
        else:
            step = 1 if x2 >= x1 else -1
            for x in range(x1, x2, step):
                _connect_cells(connections, (x, y1), (x + step, y1))
        previous = point


def _connect_cells(
    connections: dict[tuple[int, int], set[str]],
    first: tuple[int, int], second: tuple[int, int],
) -> None:
    x1, y1 = first
    x2, y2 = second
    if x2 == x1 + 1:
        directions = ("R", "L")
    elif x2 == x1 - 1:
        directions = ("L", "R")
    elif y2 == y1 + 1:
        directions = ("D", "U")
    elif y2 == y1 - 1:
        directions = ("U", "D")
    else:
        raise ValueError("graph route cells must be adjacent")
    connections.setdefault((y1, x1), set()).add(directions[0])
    connections.setdefault((y2, x2), set()).add(directions[1])


def _place_text(
    cells: dict[tuple[int, int], str | None], y: int, x: int, text: str
) -> None:
    cursor = x
    for character in text:
        cells[(y, cursor)] = character
        character_width = _character_width(character)
        for continuation in range(1, character_width):
            cells[(y, cursor + continuation)] = None
        cursor += max(1, character_width)


def _materialize_canvas(
    text_cells: dict[tuple[int, int], str | None],
    connections: dict[tuple[int, int], set[str]],
    arrows: set[tuple[int, int]],
) -> list[str]:
    occupied = set(text_cells) | set(connections) | arrows
    if not occupied:
        return []
    max_y = max(y for y, _ in occupied)
    max_x = max(x for _, x in occupied)
    lines: list[str] = []
    for y in range(max_y + 1):
        parts: list[str] = []
        for x in range(max_x + 1):
            position = (y, x)
            if position in text_cells:
                character = text_cells[position]
                if character is not None:
                    parts.append(character)
            elif position in arrows:
                parts.append("▶")
            elif position in connections:
                parts.append(_connector_character(connections[position]))
            else:
                parts.append(" ")
        lines.append("".join(parts).rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _connector_character(directions: set[str]) -> str:
    table = {
        frozenset({"L", "R"}): "─",
        frozenset({"U", "D"}): "│",
        frozenset({"R", "D"}): "┌",
        frozenset({"L", "D"}): "┐",
        frozenset({"R", "U"}): "└",
        frozenset({"L", "U"}): "┘",
        frozenset({"U", "D", "R"}): "├",
        frozenset({"U", "D", "L"}): "┤",
        frozenset({"L", "R", "D"}): "┬",
        frozenset({"L", "R", "U"}): "┴",
        frozenset({"L", "R", "U", "D"}): "┼",
    }
    frozen = frozenset(directions)
    if frozen in table:
        return table[frozen]
    if frozen <= {"L", "R"}:
        return "─"
    if frozen <= {"U", "D"}:
        return "│"
    return "┼"


def _render_graph_summary(graph: CodeGraphView) -> str:
    labels = (
        "succeeded", "failed", "denied", "conflict", "timed_out",
        "interrupted", "upstream_failed", "not_executed",
        "outcome_unknown",
    )
    parts = [
        f"{graph.summary.get(label, 0)} {label}"
        for label in labels
        if graph.summary.get(label, 0)
    ]
    if not parts:
        parts.append(f"{graph.summary.get('planned', 0)} planned")
    if graph.elapsed_ms is not None:
        parts.append(f"wall {_format_elapsed(graph.elapsed_ms)}")
    return " · ".join(parts)


def _format_elapsed(milliseconds: int) -> str:
    if milliseconds < 1000:
        return f"{milliseconds} ms"
    return f"{milliseconds / 1000:.1f} s"


def _one_line(value: str) -> str:
    return _graph_text(value)


def _graph_text(value: str) -> str:
    return _escape_terminal_text(value)


def _fit_graph_line(line: str, width: int) -> str:
    if _display_width(line) <= width:
        return line
    budget = max(0, width - 1)
    rendered: list[str] = []
    used = 0
    for character in line:
        cell_width = _character_width(character)
        if used + cell_width > budget:
            break
        rendered.append(character)
        used += cell_width
    return "".join(rendered) + "…"


def _display_width(value: str) -> int:
    return sum(_character_width(character) for character in value)


def _character_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1


class MultiLineBuffer:
    """A minimal append/backspace editor for prompt text in raw terminal mode."""

    def __init__(self) -> None:
        self._characters: list[str] = []

    @property
    def value(self) -> str:
        return "".join(self._characters)

    def insert(self, character: str) -> None:
        if not isinstance(character, str) or len(character) != 1:
            raise ValueError("character must be exactly one character")
        self._characters.append(character)

    def newline(self) -> None:
        self._characters.append("\n")

    def backspace(self) -> None:
        if self._characters:
            self._characters.pop()


def is_ctrl_enter_sequence(value: str) -> bool:
    """Return true for the portable Ctrl+Enter extensions we explicitly support."""

    return value in {"\x1b[13;5u", "\x1b[27;5;13~"}


class TerminalInputError(RuntimeError):
    """The terminal cannot enter the requested interactive input mode."""


def read_multiline_prompt(
    *,
    input_stream: object = sys.stdin,
    output_stream: object = sys.stdout,
    prompt: str = "mca> ",
    continuation: str = "...  ",
) -> str:
    """Read in raw mode; Enter inserts a line break, Ctrl+Enter submits.

    Ctrl+Enter is recognized as CSI-u (``ESC [ 13 ; 5 u``) and xterm's
    modifyOtherKeys sequence (``ESC [ 27 ; 5 ; 13 ~``). Ctrl+S is retained as
    an explicit cross-terminal fallback because many terminals encode Ctrl+Enter
    as an ordinary CR, indistinguishable from Enter.
    """

    if not hasattr(input_stream, "isatty") or not input_stream.isatty():
        raise TerminalInputError("multiline input requires an interactive terminal")
    if not hasattr(input_stream, "fileno"):
        raise TerminalInputError("multiline input stream has no file descriptor")
    descriptor = input_stream.fileno()
    try:
        previous = termios.tcgetattr(descriptor)
    except termios.error as error:
        raise TerminalInputError("could not configure terminal input") from error

    buffer = MultiLineBuffer()
    _write(output_stream, prompt)
    try:
        tty.setraw(descriptor)
        # Ask compatible terminals to report modified key combinations in a
        # distinguishable form. Unsupported terminals ignore this request; the
        # documented Ctrl+S fallback remains available.
        _write(output_stream, "\x1b[>4;2m\x1b[>1u")
        escape = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            raw = os.read(descriptor, 1)
            if not raw:
                raise EOFError
            decoded = decoder.decode(raw, final=False)
            if not decoded:
                continue
            for character in decoded:
                submitted = _consume_character(
                    character, buffer, output_stream, prompt, continuation, escape
                )
                escape = submitted.escape
                if submitted.value is not None:
                    return submitted.value
    finally:
        # Restore the terminal's ordinary keyboard protocol before returning
        # control to prompts, shells, or approval input.
        _write(output_stream, "\x1b[>4;0m\x1b[<u")
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _write(stream: object, value: str) -> None:
    stream.write(value)
    stream.flush()


class _EditResult:
    def __init__(self, escape: str, value: str | None = None) -> None:
        self.escape = escape
        self.value = value


def _consume_character(
    character: str,
    buffer: MultiLineBuffer,
    output_stream: object,
    prompt: str,
    continuation: str,
    escape: str,
) -> _EditResult:
    """Consume one decoded Unicode character from the raw terminal."""

    if escape:
        escape += character
        if is_ctrl_enter_sequence(escape):
            _write(output_stream, "\r\n")
            return _EditResult("", buffer.value.strip())
        candidates = ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D")
        if escape in {"\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"}:
            return _EditResult("")
        if any(candidate.startswith(escape) for candidate in candidates):
            return _EditResult(escape)
        buffer.insert(escape)
        _write(output_stream, escape)
        return _EditResult("")
    if character == "\x1b":
        return _EditResult(character)
    if character == "\x13":
        _write(output_stream, "\r\n")
        return _EditResult("", buffer.value.strip())
    if character == "\x03":
        raise KeyboardInterrupt
    if character == "\x04" and not buffer.value:
        raise EOFError
    if character in {"\x7f", "\x08"}:
        if buffer.value:
            was_newline = buffer.value.endswith("\n")
            buffer.backspace()
            if was_newline:
                _write(output_stream, "\r\x1b[2K" + prompt + buffer.value.rsplit("\n", 1)[-1])
            else:
                _write(output_stream, "\b \b")
        return _EditResult("")
    if character in {"\r", "\n"}:
        buffer.newline()
        _write(output_stream, "\r\n" + continuation)
        return _EditResult("")
    if ord(character) >= 32:
        buffer.insert(character)
        _write(output_stream, character)
    return _EditResult("")


__all__ = [
    "render_code_graph_ansi",
    "render_code_graph_plain",
    "MultiLineBuffer",
    "TerminalInputError",
    "TerminalTheme",
    "is_ctrl_enter_sequence",
    "read_multiline_prompt",
]
