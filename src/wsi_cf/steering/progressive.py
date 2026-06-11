from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from wsi_cf.common.io import read_json
from wsi_cf.steering.cell_selection import parse_cell_spec, validate_cells

CENTER_2X2_LOCAL_CELLS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
)
EDIT_SUPPORT_CENTER_2X2 = "center_2x2"
EDIT_SUPPORT_BORDER_RELAXED = "border_relaxed"
EDIT_SUPPORT_CHOICES: tuple[str, ...] = (EDIT_SUPPORT_CENTER_2X2, EDIT_SUPPORT_BORDER_RELAXED)


@dataclass(frozen=True)
class ProgressiveEditRequest:
    run_id: str
    region_id: str
    target_cells: tuple[tuple[int, int], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ProgressiveWindow:
    window_id: str
    row_index: int
    col_index: int
    gx0: int
    gy0: int
    grid_w: int
    grid_h: int
    left: int
    top: int

    def global_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(self.gx0) + int(dx), int(self.gy0) + int(dy))
            for dy in range(int(self.grid_h))
            for dx in range(int(self.grid_w))
        )


@dataclass(frozen=True)
class PlannedProgressiveStep:
    step_index: int
    window: ProgressiveWindow
    edit_cells_global: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProgressiveCoverageState:
    target_cells: tuple[tuple[int, int], ...]
    edited_cells: tuple[tuple[int, int], ...]
    visited_cells: tuple[tuple[int, int], ...]
    window_history: tuple[str, ...]


def _normalize_cell(item: object) -> tuple[int, int]:
    if isinstance(item, dict):
        if "gx" not in item or "gy" not in item:
            raise ValueError("Cell dicts must contain gx and gy")
        return int(item["gx"]), int(item["gy"])
    if isinstance(item, str):
        return parse_cell_spec(item)
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return int(item[0]), int(item[1])
    raise ValueError(f"Unsupported cell spec: {item!r}")


def _stable_run_id(*, region_id: str, target_cells: Sequence[tuple[int, int]]) -> str:
    payload = ";".join(f"{gx},{gy}" for gx, gy in sorted(target_cells, key=lambda item: (item[1], item[0])))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{region_id}__targets_{digest}"


def load_progressive_edit_manifest(manifest_path: Path) -> list[ProgressiveEditRequest]:
    payload = read_json(manifest_path)
    if not isinstance(payload, list):
        raise ValueError(f"Progressive edit manifest must be a JSON list: {manifest_path}")
    requests: list[ProgressiveEditRequest] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Progressive edit manifest item {idx} must be an object")
        region_id = str(item.get("region_id", "")).strip()
        if not region_id:
            raise ValueError(f"Progressive edit manifest item {idx} must contain region_id")
        raw_cells = item.get("target_cells")
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ValueError(f"Progressive edit manifest item {idx} must contain a non-empty target_cells list")
        target_cells = [_normalize_cell(cell) for cell in raw_cells]
        seen: set[tuple[int, int]] = set()
        deduped: list[tuple[int, int]] = []
        for cell in target_cells:
            if cell not in seen:
                deduped.append((int(cell[0]), int(cell[1])))
                seen.add(cell)
        run_id = str(item.get("run_id", "")).strip() or _stable_run_id(region_id=region_id, target_cells=deduped)
        metadata = {
            str(key): value
            for key, value in item.items()
            if str(key) not in {"run_id", "region_id", "target_cells"}
        }
        requests.append(
            ProgressiveEditRequest(
                run_id=run_id,
                region_id=region_id,
                target_cells=tuple(deduped),
                metadata=metadata,
            )
        )
    return requests


def enumerate_progressive_windows(
    *,
    grid_w: int,
    grid_h: int,
    window_grid_side: int = 4,
    stride_cells: int = 2,
    grid_step_px: int = 256,
) -> list[ProgressiveWindow]:
    if int(window_grid_side) <= 0 or int(stride_cells) <= 0:
        raise ValueError("window_grid_side and stride_cells must be > 0")
    if int(grid_w) < int(window_grid_side) or int(grid_h) < int(window_grid_side):
        raise ValueError("Grid must be at least as large as the window size")

    def _starts(total: int) -> list[int]:
        starts = list(range(0, int(total) - int(window_grid_side) + 1, int(stride_cells)))
        if starts[-1] != int(total) - int(window_grid_side):
            starts.append(int(total) - int(window_grid_side))
        return starts

    xs = _starts(int(grid_w))
    ys = _starts(int(grid_h))
    windows: list[ProgressiveWindow] = []
    for row_index, gy0 in enumerate(ys):
        for col_index, gx0 in enumerate(xs):
            windows.append(
                ProgressiveWindow(
                    window_id=f"r{int(row_index)}_c{int(col_index)}",
                    row_index=int(row_index),
                    col_index=int(col_index),
                    gx0=int(gx0),
                    gy0=int(gy0),
                    grid_w=int(window_grid_side),
                    grid_h=int(window_grid_side),
                    left=int(gx0 * int(grid_step_px)),
                    top=int(gy0 * int(grid_step_px)),
                )
            )
    return windows


def _window_target_cells(
    window: ProgressiveWindow,
    target_set: set[tuple[int, int]],
    *,
    grid_w: int,
    grid_h: int,
    edit_support: str,
) -> set[tuple[int, int]]:
    return local_edit_support_global_cells(
        window,
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        edit_support=str(edit_support),
    ).intersection(target_set)


def center_support_global_cells(window: ProgressiveWindow) -> set[tuple[int, int]]:
    return {
        (int(window.gx0) + int(lx), int(window.gy0) + int(ly))
        for lx, ly in CENTER_2X2_LOCAL_CELLS
        if 0 <= int(lx) < int(window.grid_w) and 0 <= int(ly) < int(window.grid_h)
    }


def local_edit_support_global_cells(
    window: ProgressiveWindow,
    *,
    grid_w: int,
    grid_h: int,
    edit_support: str = EDIT_SUPPORT_CENTER_2X2,
) -> set[tuple[int, int]]:
    if str(edit_support) not in EDIT_SUPPORT_CHOICES:
        raise ValueError(f"Unsupported edit_support: {edit_support}")
    allowed = set(center_support_global_cells(window))
    if str(edit_support) == EDIT_SUPPORT_BORDER_RELAXED:
        for gx, gy in window.global_cells():
            if int(gx) in {0, int(grid_w) - 1} or int(gy) in {0, int(grid_h) - 1}:
                allowed.add((int(gx), int(gy)))
    return allowed


def split_cells_by_edit_support(
    *,
    target_cells: Sequence[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    window_grid_side: int = 4,
    stride_cells: int = 2,
    grid_step_px: int = 256,
    edit_support: str = EDIT_SUPPORT_CENTER_2X2,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    if str(edit_support) not in EDIT_SUPPORT_CHOICES:
        raise ValueError(f"Unsupported edit_support: {edit_support}")
    validated_targets = validate_cells(list(target_cells), grid_w=int(grid_w), grid_h=int(grid_h))
    windows = enumerate_progressive_windows(
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        window_grid_side=int(window_grid_side),
        stride_cells=int(stride_cells),
        grid_step_px=int(grid_step_px),
    )
    support: set[tuple[int, int]] = set()
    for window in windows:
        support.update(
            local_edit_support_global_cells(
                window,
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                edit_support=str(edit_support),
            )
        )
    supported: list[tuple[int, int]] = []
    unsupported: list[tuple[int, int]] = []
    for cell in sorted(set(validated_targets), key=lambda item: (int(item[1]), int(item[0]))):
        if cell in support:
            supported.append((int(cell[0]), int(cell[1])))
        else:
            unsupported.append((int(cell[0]), int(cell[1])))
    return tuple(supported), tuple(unsupported)


def plan_progressive_steps(
    *,
    target_cells: Sequence[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    window_grid_side: int = 4,
    stride_cells: int = 2,
    grid_step_px: int = 256,
    edit_support: str = EDIT_SUPPORT_CENTER_2X2,
) -> list[PlannedProgressiveStep]:
    if str(edit_support) not in EDIT_SUPPORT_CHOICES:
        raise ValueError(f"Unsupported edit_support: {edit_support}")
    validated_targets = validate_cells(list(target_cells), grid_w=int(grid_w), grid_h=int(grid_h))
    target_set = set(validated_targets)
    if not target_set:
        raise ValueError("At least one target cell is required")

    windows = enumerate_progressive_windows(
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        window_grid_side=int(window_grid_side),
        stride_cells=int(stride_cells),
        grid_step_px=int(grid_step_px),
    )
    window_targets = {
        window.window_id: _window_target_cells(
            window,
            target_set,
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            edit_support=str(edit_support),
        )
        for window in windows
    }
    candidate_windows = [window for window in windows if window_targets[window.window_id]]
    if not candidate_windows:
        raise RuntimeError(
            f"No progressive windows cover the requested target cells using edit_support={edit_support}"
        )
    covered_targets = set().union(*(window_targets[window.window_id] for window in candidate_windows))
    unsupported_targets = sorted(target_set - covered_targets, key=lambda item: (int(item[1]), int(item[0])))
    if unsupported_targets:
        raise RuntimeError(
            f"Some target cells are outside edit_support={edit_support}: {unsupported_targets}. "
            "Filter target_cells to supported cells or use edit_support=border_relaxed."
        )

    remaining = set(target_set)
    used_window_ids: set[str] = set()
    steps: list[PlannedProgressiveStep] = []
    current_window: ProgressiveWindow | None = None

    while remaining:
        viable = [
            window
            for window in candidate_windows
            if window.window_id not in used_window_ids and window_targets[window.window_id].intersection(remaining)
        ]
        if not viable:
            raise RuntimeError(
                f"Could not cover all requested target cells using edit_support={edit_support}"
            )

        if current_window is None:
            anchor = min(remaining, key=lambda item: (int(item[1]), int(item[0])))
            anchor_candidates = [window for window in viable if anchor in window_targets[window.window_id]]
            chosen = min(
                anchor_candidates,
                key=lambda window: (
                    -len(window_targets[window.window_id].intersection(remaining)),
                    int(window.gy0),
                    int(window.gx0),
                    str(window.window_id),
                ),
            )
        else:
            chosen = min(
                viable,
                key=lambda window: (
                    -len(window_targets[window.window_id].intersection(remaining)),
                    abs(int(window.gx0) - int(current_window.gx0)) + abs(int(window.gy0) - int(current_window.gy0)),
                    int(window.gy0),
                    int(window.gx0),
                    str(window.window_id),
                ),
            )

        edit_cells = tuple(
            sorted(
                window_targets[chosen.window_id].intersection(remaining),
                key=lambda item: (int(item[1]), int(item[0])),
            )
        )
        steps.append(
            PlannedProgressiveStep(
                step_index=len(steps),
                window=chosen,
                edit_cells_global=edit_cells,
            )
        )
        remaining.difference_update(edit_cells)
        used_window_ids.add(chosen.window_id)
        current_window = chosen

    return steps


def make_initial_progressive_state(
    *,
    target_cells: Sequence[tuple[int, int]],
) -> ProgressiveCoverageState:
    normalized = tuple(sorted({(int(gx), int(gy)) for gx, gy in target_cells}, key=lambda item: (item[1], item[0])))
    return ProgressiveCoverageState(
        target_cells=normalized,
        edited_cells=tuple(),
        visited_cells=tuple(),
        window_history=tuple(),
    )


def advance_progressive_state(
    state: ProgressiveCoverageState,
    *,
    window: ProgressiveWindow,
    edit_cells_global: Sequence[tuple[int, int]],
) -> ProgressiveCoverageState:
    edited = set(state.edited_cells)
    edited.update((int(gx), int(gy)) for gx, gy in edit_cells_global)
    visited = set(state.visited_cells)
    visited.update(window.global_cells())
    history = list(state.window_history)
    history.append(str(window.window_id))
    return ProgressiveCoverageState(
        target_cells=tuple(state.target_cells),
        edited_cells=tuple(sorted(edited, key=lambda item: (item[1], item[0]))),
        visited_cells=tuple(sorted(visited, key=lambda item: (item[1], item[0]))),
        window_history=tuple(history),
    )


def window_local_cells(
    *,
    window: ProgressiveWindow,
    global_cells: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for gx, gy in global_cells:
        if int(window.gx0) <= int(gx) < int(window.gx0) + int(window.grid_w) and int(window.gy0) <= int(gy) < int(window.gy0) + int(window.grid_h):
            out.append((int(gx) - int(window.gx0), int(gy) - int(window.gy0)))
    out.sort(key=lambda item: (item[1], item[0]))
    return out


def build_history_aware_preserve_map(
    *,
    width: int,
    height: int,
    grid_step_px: int,
    window: ProgressiveWindow,
    edit_cells_global: Sequence[tuple[int, int]],
    visited_cells_global: Sequence[tuple[int, int]],
    preserve_edit_strength: float,
    preserve_visited_strength: float,
    preserve_fresh_context_strength: float,
) -> torch.Tensor:
    for value, name in (
        (preserve_edit_strength, "preserve_edit_strength"),
        (preserve_visited_strength, "preserve_visited_strength"),
        (preserve_fresh_context_strength, "preserve_fresh_context_strength"),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be in [0,1]")

    canvas = torch.full((1, 1, int(height), int(width)), float(preserve_fresh_context_strength), dtype=torch.float32)
    visited_local = window_local_cells(window=window, global_cells=visited_cells_global)
    edit_local = window_local_cells(window=window, global_cells=edit_cells_global)

    for gx, gy in visited_local:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        canvas[:, :, y0:y1, x0:x1] = float(preserve_visited_strength)

    for gx, gy in edit_local:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        canvas[:, :, y0:y1, x0:x1] = float(preserve_edit_strength)

    return canvas


def preserve_map_preview(preserve_map: torch.Tensor) -> Image.Image:
    if preserve_map.dim() == 4:
        arr = preserve_map[0, 0].detach().cpu().numpy()
    elif preserve_map.dim() == 2:
        arr = preserve_map.detach().cpu().numpy()
    else:
        raise ValueError("preserve_map must have shape [1,1,H,W] or [H,W]")
    arr01 = np.clip(arr.astype(np.float32), 0.0, 1.0)
    grey = 208.0 + 42.0 * arr01
    rgb = np.stack([grey, grey, grey], axis=-1)
    edit_mask = arr01 <= 1e-6
    rgb[edit_mask] = np.asarray([252.0, 68.0, 68.0], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")


def draw_cells_overlay(
    img: Image.Image,
    *,
    cells: Sequence[tuple[int, int]],
    grid_step_px: int,
    outline: tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 6,
) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    drawn_edges: set[tuple[str, int, int, int]] = set()
    for idx, (gx, gy) in enumerate(cells, start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(canvas.size[0] - 1, x0 + int(grid_step_px))
        y1 = min(canvas.size[1] - 1, y0 + int(grid_step_px))
        edges = (
            (("h", y0, x0, x1), ((x0, y0), (x1, y0))),
            (("v", x1, y0, y1), ((x1, y0), (x1, y1))),
            (("h", y1, x0, x1), ((x0, y1), (x1, y1))),
            (("v", x0, y0, y1), ((x0, y0), (x0, y1))),
        )
        for edge_key, edge_points in edges:
            if edge_key not in drawn_edges:
                draw.line(edge_points, fill=outline, width=int(outline_width))
                drawn_edges.add(edge_key)
        draw.text((x0 + 8, y0 + 8), str(idx), fill=(255, 255, 0))
    return canvas


def draw_step_region_overlay(
    img: Image.Image,
    *,
    window: ProgressiveWindow,
    edit_cells_global: Sequence[tuple[int, int]],
    support_cells_global: Sequence[tuple[int, int]] | None = None,
    grid_step_px: int,
    grid_outline: tuple[int, int, int, int] = (0, 0, 0, 190),
    window_outline: tuple[int, int, int, int] = (255, 0, 0, 255),
    context_fill: tuple[int, int, int, int] = (235, 235, 235, 88),
    support_fill: tuple[int, int, int, int] = (255, 0, 0, 48),
    edit_fill: tuple[int, int, int, int] = (255, 0, 0, 92),
) -> Image.Image:
    canvas = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = canvas.size
    step = int(grid_step_px)

    edit_cells = {(int(gx), int(gy)) for gx, gy in edit_cells_global}
    support_cells = tuple(support_cells_global) if support_cells_global is not None else tuple(edit_cells_global)
    support_cell_set = {(int(gx), int(gy)) for gx, gy in support_cells}
    for gx, gy in window.global_cells():
        if (int(gx), int(gy)) in support_cell_set:
            continue
        x0 = int(gx) * step
        y0 = int(gy) * step
        x1 = min(width - 1, (int(gx) + 1) * step)
        y1 = min(height - 1, (int(gy) + 1) * step)
        draw.rectangle([x0, y0, x1, y1], fill=context_fill)

    for gx, gy in support_cells:
        x0 = int(gx) * step
        y0 = int(gy) * step
        x1 = min(width - 1, (int(gx) + 1) * step)
        y1 = min(height - 1, (int(gy) + 1) * step)
        fill = edit_fill if (int(gx), int(gy)) in edit_cells else support_fill
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    for x in range(0, width + 1, step):
        grid_x = min(width - 1, int(x))
        draw.line([(grid_x, 0), (grid_x, height - 1)], fill=grid_outline, width=4)
    for y in range(0, height + 1, step):
        grid_y = min(height - 1, int(y))
        draw.line([(0, grid_y), (width - 1, grid_y)], fill=grid_outline, width=4)

    drawn_edges: set[tuple[str, int, int, int]] = set()
    for gx, gy in support_cells:
        x0 = int(gx) * step
        y0 = int(gy) * step
        x1 = min(width - 1, (int(gx) + 1) * step)
        y1 = min(height - 1, (int(gy) + 1) * step)
        edges = (
            (("h", y0, x0, x1), ((x0, y0), (x1, y0))),
            (("v", x1, y0, y1), ((x1, y0), (x1, y1))),
            (("h", y1, x0, x1), ((x0, y1), (x1, y1))),
            (("v", x0, y0, y1), ((x0, y0), (x0, y1))),
        )
        for edge_key, edge_points in edges:
            if edge_key not in drawn_edges:
                draw.line(edge_points, fill=window_outline, width=6)
                drawn_edges.add(edge_key)

    x0 = int(window.left)
    y0 = int(window.top)
    x1 = min(width - 1, x0 + int(window.grid_w) * step)
    y1 = min(height - 1, y0 + int(window.grid_h) * step)
    dash_length = 88
    dash_gap = 52
    dash_width = 16
    radius = dash_width // 2

    def draw_rounded_dash(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line([start, end], fill=window_outline, width=dash_width)
        for x, y in (start, end):
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=window_outline)

    for x in range(x0, x1 + 1, dash_length + dash_gap):
        x_end = min(x + dash_length, x1)
        draw_rounded_dash((x, y0), (x_end, y0))
        draw_rounded_dash((x, y1), (x_end, y1))
    for y in range(y0, y1 + 1, dash_length + dash_gap):
        y_end = min(y + dash_length, y1)
        draw_rounded_dash((x0, y), (x0, y_end))
        draw_rounded_dash((x1, y), (x1, y_end))

    canvas.alpha_composite(overlay)
    return canvas.convert("RGB")


def draw_step_edit_area_zoom_4x4(
    img: Image.Image,
    *,
    window: ProgressiveWindow,
    edit_cells_global: Sequence[tuple[int, int]],
    support_cells_global: Sequence[tuple[int, int]],
    grid_step_px: int,
) -> Image.Image:
    full_overlay = draw_step_region_overlay(
        img,
        window=window,
        edit_cells_global=edit_cells_global,
        support_cells_global=support_cells_global,
        grid_step_px=int(grid_step_px),
    )
    step = int(grid_step_px)
    return full_overlay.crop(
        (
            int(window.gx0) * step,
            int(window.gy0) * step,
            (int(window.gx0) + int(window.grid_w)) * step,
            (int(window.gy0) + int(window.grid_h)) * step,
        )
    )
