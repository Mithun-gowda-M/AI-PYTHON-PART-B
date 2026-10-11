"""Conflict-Based Search for optimal paths on a small multi-robot grid."""

from heapq import heappop, heappush
from itertools import count

Cell = tuple[int, int]
Constraint = tuple[int, Cell, Cell | None]  # time, cell, previous cell (edge constraint)


def heuristic(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def low_level(grid: list[list[int]], start: Cell, goal: Cell, constraints: set[Constraint], max_time: int = 40) -> list[Cell] | None:
    blocked = {(t, cell, prev) for t, cell, prev in constraints}
    serial = count()
    frontier = [(heuristic(start, goal), 0, next(serial), start, (start,))]
    best = {(start, 0): 0}
    rows, cols = len(grid), len(grid[0])
    while frontier:
        _, time, _, cell, path = heappop(frontier)
        future_goal_blocks = any(
            blocked_cell == goal and blocked_time > time and (previous is None or previous == goal)
            for blocked_time, blocked_cell, previous in constraints
        )
        if cell == goal and not future_goal_blocks:
            return list(path)
        if time >= max_time:
            continue
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = cell[0] + dr, cell[1] + dc
            nt = time + 1
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or grid[nxt[0]][nxt[1]] == 1:
                continue
            if (nt, nxt, None) in blocked or (nt, nxt, cell) in blocked:
                continue
            state = (nxt, nt)
            if best.get(state, 10**9) <= nt:
                continue
            best[state] = nt
            new_path = path + (nxt,)
            heappush(frontier, (nt + heuristic(nxt, goal), nt, next(serial), nxt, new_path))
    return None


def first_conflict(paths: list[list[Cell]]) -> tuple[str, int, int, Cell, Cell | None] | None:
    horizon = max(map(len, paths))
    for t in range(horizon):
        positions = [p[min(t, len(p) - 1)] for p in paths]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if positions[i] == positions[j]:
                    return "vertex", t, i, positions[i], positions[j]
                if t > 0:
                    pi, pj = paths[i][min(t - 1, len(paths[i]) - 1)], paths[j][min(t - 1, len(paths[j]) - 1)]
                    if pi == positions[j] and pj == positions[i]:
                        return "edge", t, i, positions[i], pi
    return None


def cbs(grid: list[list[int]], starts: list[Cell], goals: list[Cell]) -> tuple[list[list[Cell]], int] | None:
    serial = count()
    root_constraints = [set() for _ in starts]
    root_paths = [low_level(grid, s, g, set()) for s, g in zip(starts, goals)]
    if any(path is None for path in root_paths):
        return None
    root = (sum(len(p) for p in root_paths), next(serial), root_constraints, root_paths)
    queue = [root]
    expanded = 0
    while queue:
        _, _, constraints, paths = heappop(queue)
        expanded += 1
        conflict = first_conflict(paths)
        if conflict is None:
            return paths, expanded
        kind, time, i, cell, other = conflict
        # Determine the second robot from the conflicting cell/time.
        positions = [p[min(time, len(p) - 1)] for p in paths]
        j = next(k for k in range(len(paths)) if k != i and positions[k] == other)
        for robot, forbidden, previous in ((i, cell, paths[i][time - 1] if kind == "edge" else None),
                                             (j, other, paths[j][time - 1] if kind == "edge" else None)):
            child_constraints = [set(c) for c in constraints]
            child_constraints[robot].add((time, forbidden, previous))
            child_paths = list(paths)
            child_paths[robot] = low_level(grid, starts[robot], goals[robot], child_constraints[robot])
            if child_paths[robot] is not None:
                heappush(queue, (sum(len(p) for p in child_paths), next(serial), child_constraints, child_paths))
    return None


def demo() -> None:
    grid = [[0] * 7 for _ in range(5)]
    for r, c in ((1, 3), (2, 3), (3, 3)):
        grid[r][c] = 1
    starts, goals = [(2, 0), (2, 6)], [(2, 6), (2, 0)]
    result = cbs(grid, starts, goals)
    if result is None:
        print("No collision-free plan found.")
        return
    paths, expanded = result
    for idx, path in enumerate(paths, 1):
        print(f"Robot {idx}: {path[:path.index(goals[idx - 1]) + 1]}")
    print("High-level CBS nodes expanded:", expanded)


if __name__ == "__main__":
    demo()
