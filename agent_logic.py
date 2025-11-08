from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

try:
    import torch
    from rl.model import DQN
except Exception:
    torch = None
    DQN = None  # type: ignore

from case_closed_game import Direction

# Candidate actions (dir, boost)
ACTION_SET: List[Tuple[Direction, bool]] = [
    (Direction.UP, False),
    (Direction.DOWN, False),
    (Direction.LEFT, False),
    (Direction.RIGHT, False),
    (Direction.UP, True),
    (Direction.DOWN, True),
    (Direction.LEFT, True),
    (Direction.RIGHT, True),
]

DIR_TO_STR = {
    Direction.UP: "UP",
    Direction.DOWN: "DOWN",
    Direction.LEFT: "LEFT",
    Direction.RIGHT: "RIGHT",
}


def _normalize_delta(delta: int, modulus: int) -> int:
    # Convert wrap-around delta to -1, 0, or 1 step
    if delta > 1:
        return -1
    if delta < -1:
        return 1
    return delta


def _get_current_direction_from_trail(trail: List[Tuple[int, int]]) -> Direction:
    if len(trail) < 2:
        return Direction.RIGHT
    x2, y2 = trail[-1]
    x1, y1 = trail[-2]
    dx = _normalize_delta(x2 - x1, modulus=9999)  # modulus not used here; we clamp to {-1,0,1}
    dy = _normalize_delta(y2 - y1, modulus=9999)
    if dx == 1:
        return Direction.RIGHT
    if dx == -1:
        return Direction.LEFT
    if dy == 1:
        return Direction.DOWN
    return Direction.UP


def _wrap(x: int, limit: int) -> int:
    return x % limit


def _is_cell_blocked(board: List[List[int]], x: int, y: int) -> bool:
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    return board[y % height][x % width] != 0


def _simulate_move_path(
    board: List[List[int]],
    start: Tuple[int, int],
    direction: Direction,
    steps: int,
) -> Tuple[Tuple[int, int], bool, List[Tuple[int, int]]]:
    """
    Simulate 'steps' forward moves on a torus grid.
    Returns (final_pos, survived_all_steps, visited_positions)
    """
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    x, y = start
    visited: List[Tuple[int, int]] = []
    for _ in range(steps):
        dx, dy = direction.value
        x = (x + dx) % width
        y = (y + dy) % height
        if _is_cell_blocked(board, x, y):
            return (x, y), False, visited
        visited.append((x, y))
    return (x, y), True, visited


def _reachable_area_after(
    board: List[List[int]],
    start: Tuple[int, int],
    additional_blocked: List[Tuple[int, int]],
) -> int:
    """BFS flood-fill from start on a torus, counting reachable empty cells."""
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    blocked = set(additional_blocked)

    def is_blocked(cx: int, cy: int) -> bool:
        if (cx % width, cy % height) in blocked:
            return True
        return _is_cell_blocked(board, cx, cy)

    from collections import deque

    q = deque()
    start_x, start_y = start
    if is_blocked(start_x, start_y):
        return 0
    q.append((start_x % width, start_y % height))
    seen = set([(start_x % width, start_y % height)])
    area = 0
    while q:
        cx, cy = q.popleft()
        area += 1
        for dx, dy in [d.value for d in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)]:
            nx = (cx + dx) % width
            ny = (cy + dy) % height
            if (nx, ny) in seen:
                continue
            if is_blocked(nx, ny):
                continue
            seen.add((nx, ny))
            q.append((nx, ny))
    return area


def _encode_state_for_q(
    state: Dict[str, Any],
    my_player_number: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Encode to (grid[C][H][W], scalars[S]) from the perspective of my_player_number.
    Channels: my trail, opp trail, my head, opp head.
    Scalars: my boosts, opp boosts, normalized turn count.
    """
    board = state["board"]
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    channels = 4
    grid = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]

    if my_player_number == 1:
        my_trail = state.get("agent1_trail", [])
        opp_trail = state.get("agent2_trail", [])
        my_boosts = float(state.get("agent1_boosts", 0))
        opp_boosts = float(state.get("agent2_boosts", 0))
    else:
        my_trail = state.get("agent2_trail", [])
        opp_trail = state.get("agent1_trail", [])
        my_boosts = float(state.get("agent2_boosts", 0))
        opp_boosts = float(state.get("agent1_boosts", 0))

    if my_trail:
        for (x, y) in my_trail:
            grid[0][y][x] = 1
        hx, hy = my_trail[-1]
        grid[2][hy][hx] = 1
    if opp_trail:
        for (x, y) in opp_trail:
            grid[1][y][x] = 1
        ox, oy = opp_trail[-1]
        grid[3][oy][ox] = 1

    turn_div = 500.0
    if config is not None:
        try:
            turn_div = float(config.get("turn_normalize_divisor", turn_div))
        except Exception:
            pass
    turn_count = float(state.get("turn_count", 0))
    scalars = [my_boosts, opp_boosts, min(turn_count / max(1.0, turn_div), 1.0)]
    return grid, scalars


def _mask_invalid_actions(
    state: Dict[str, Any],
    my_player_number: int,
) -> List[bool]:
    """
    Returns a boolean mask (len ACTION_SET) where True means action is valid.
    Invalid if:
      - immediate collision on first/second step
      - tries to use boost with no boosts left
    """
    board = state["board"]
    height = len(board)
    width = len(board[0]) if height > 0 else 0

    if my_player_number == 1:
        my_trail = state.get("agent1_trail", [])
        my_boosts = int(state.get("agent1_boosts", 0))
    else:
        my_trail = state.get("agent2_trail", [])
        my_boosts = int(state.get("agent2_boosts", 0))

    if not my_trail:
        # If unknown, allow basic 4 actions without boost
        return [True, True, True, True, False, False, False, False]

    head_x, head_y = my_trail[-1]
    valid_mask: List[bool] = []
    for (direction, use_boost) in ACTION_SET:
        if use_boost and my_boosts <= 0:
            valid_mask.append(False)
            continue
        steps = 2 if use_boost else 1
        (_, survived, _) = _simulate_move_path(board, (head_x, head_y), direction, steps)
        valid_mask.append(survived)
    return valid_mask


def _heuristic_score_action(
    state: Dict[str, Any],
    my_player_number: int,
    action_index: int,
) -> float:
    """Score via reachable empty area after taking the action; prefer non-boost unless it increases area."""
    board = state["board"]
    height = len(board)
    width = len(board[0]) if height > 0 else 0

    if my_player_number == 1:
        my_trail = state.get("agent1_trail", [])
    else:
        my_trail = state.get("agent2_trail", [])
    if not my_trail:
        return 0.0

    head_x, head_y = my_trail[-1]
    direction, use_boost = ACTION_SET[action_index]
    steps = 2 if use_boost else 1
    (final_pos, survived, visited) = _simulate_move_path(board, (head_x, head_y), direction, steps)
    if not survived:
        return -1e6

    # After moving, treat newly visited cells as blocked to approximate trail growth
    area = _reachable_area_after(board, final_pos, visited)
    # Penalize boost a bit to avoid reckless use unless it meaningfully increases area
    if use_boost:
        area -= 0.25
    return float(area)


def choose_move(
    state: Dict[str, Any],
    my_player_number: int,
    qnet: Optional["DQN"] = None,
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Decide the move string to return to judge, using Q-network if available, else heuristic fallback.
    """
    valid_mask = _mask_invalid_actions(state, my_player_number)
    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_indices:
        # As a last resort, choose a safe non-boost among the four dirs if any
        for i in range(4):
            if valid_mask[i]:
                direction, _ = ACTION_SET[i]
                return DIR_TO_STR[direction]
        # If truly nothing, default RIGHT
        return "RIGHT"

    # If Q-network provided, score valid actions
    if qnet is not None and torch is not None:
        grid, scalars = _encode_state_for_q(state, my_player_number, config=config)
        with torch.no_grad():
            grid_t = torch.as_tensor(grid, dtype=torch.float32, device=device).unsqueeze(0)
            scalars_t = torch.as_tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = qnet(grid_t, scalars_t).squeeze(0)  # (A,)
            # Mask invalid by setting to very low
            q_values_masked = q_values.clone()
            inv = torch.tensor([not v for v in valid_mask], dtype=torch.bool, device=device)
            q_values_masked[inv] = -1e9
            best_idx = int(torch.argmax(q_values_masked).item())
    else:
        # Heuristic: pick argmax of area score among valid
        best_idx = max(valid_indices, key=lambda i: _heuristic_score_action(state, my_player_number, i))

    direction, use_boost = ACTION_SET[best_idx]
    # Final conservative boost check: do not boost if it doesn't improve area
    if use_boost:
        conservative = True
        boost_margin = 0.1
        if config is not None:
            try:
                conservative = bool(config.get("conservative_boost", conservative))
                boost_margin = float(config.get("boost_margin", boost_margin))
            except Exception:
                pass
        if conservative:
            area_boost = _heuristic_score_action(state, my_player_number, best_idx)
            # Compare to non-boost same direction
            non_boost_idx = ACTION_SET.index((direction, False))
            area_noboost = _heuristic_score_action(state, my_player_number, non_boost_idx)
            if area_boost <= area_noboost + boost_margin:
                use_boost = False

    return f"{DIR_TO_STR[direction]}:BOOST" if use_boost else DIR_TO_STR[direction]


