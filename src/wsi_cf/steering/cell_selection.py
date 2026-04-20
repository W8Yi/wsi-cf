from __future__ import annotations

import random


def parse_cell_spec(spec: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(spec).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid cell spec '{spec}'. Expected gx,gy")
    return int(parts[0]), int(parts[1])


def parse_cell_specs(specs: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for spec in specs:
        cell = parse_cell_spec(spec)
        if cell not in seen:
            out.append(cell)
            seen.add(cell)
    return out


def encode_cells(cells: list[tuple[int, int]]) -> str:
    return ";".join(f"{gx},{gy}" for gx, gy in cells)


def decode_cells(spec: str) -> list[tuple[int, int]]:
    if not str(spec).strip():
        return []
    return parse_cell_specs([item for item in str(spec).split(";") if item.strip()])


def validate_cells(cells: list[tuple[int, int]], *, grid_w: int, grid_h: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for gx, gy in cells:
        if not (0 <= int(gx) < int(grid_w) and 0 <= int(gy) < int(grid_h)):
            raise ValueError(f"Cell {(gx, gy)} out of bounds for grid {(grid_w, grid_h)}")
        out.append((int(gx), int(gy)))
    return out


def random_cells(*, grid_w: int, grid_h: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    coords = [(gx, gy) for gy in range(int(grid_h)) for gx in range(int(grid_w))]
    if int(count) <= 0 or int(count) > len(coords):
        raise ValueError(f"count must be in [1, {len(coords)}]")
    return sorted(rng.sample(coords, int(count)), key=lambda item: (item[1], item[0]))


def cell_neighbors(gx: int, gy: int, *, grid_w: int, grid_h: int) -> list[tuple[int, int]]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = int(gx) + dx
        ny = int(gy) + dy
        if 0 <= nx < int(grid_w) and 0 <= ny < int(grid_h):
            out.append((nx, ny))
    return out


def random_connected_cells(
    *,
    grid_w: int,
    grid_h: int,
    count: int,
    rng: random.Random,
    start: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    if int(count) <= 0:
        raise ValueError("count must be > 0")
    if start is None:
        start = (rng.randrange(int(grid_w)), rng.randrange(int(grid_h)))
    selected = [start]
    selected_set = {start}
    while len(selected) < int(count):
        frontier = []
        for gx, gy in selected:
            for cell in cell_neighbors(gx, gy, grid_w=int(grid_w), grid_h=int(grid_h)):
                if cell not in selected_set and cell not in frontier:
                    frontier.append(cell)
        if not frontier:
            raise ValueError(f"Could not build connected set of size {count} on grid {(grid_w, grid_h)}")
        nxt = rng.choice(frontier)
        selected.append(nxt)
        selected_set.add(nxt)
    return sorted(selected, key=lambda item: (item[1], item[0]))


def block_cells(
    *,
    origin_gx: int,
    origin_gy: int,
    width: int,
    height: int,
    grid_w: int,
    grid_h: int,
) -> list[tuple[int, int]]:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be > 0")
    cells = [(int(origin_gx) + dx, int(origin_gy) + dy) for dy in range(int(height)) for dx in range(int(width))]
    return validate_cells(cells, grid_w=int(grid_w), grid_h=int(grid_h))
