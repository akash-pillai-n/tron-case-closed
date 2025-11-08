"""
Safety shield for multi-agent Tron: action masking, collision checks, deadlock avoidance.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from case_closed_game import Direction, AGENT, EMPTY


DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
DIR_TO_STR = {
    Direction.UP: "UP",
    Direction.DOWN: "DOWN",
    Direction.LEFT: "LEFT",
    Direction.RIGHT: "RIGHT",
}


def compute_safe_action_mask(
    obs: Dict[str, Any],
    width: int = 20,
    height: int = 18,
    min_area_threshold: int = 15,
) -> List[bool]:
    """
    Compute safety mask for 8 actions (4 dirs × {no-boost, boost}).
    
    Masks out:
    - Opposite direction
    - Immediate collisions (1-step or 2-step for boost)
    - Actions leading to very low reachable area (deadlock risk)
    - Boosts when no boosts remaining
    
    Returns:
        List of 8 booleans (True = safe/allowed)
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    my_direction = obs.get('my_direction', Direction.RIGHT)
    my_boosts = obs.get('my_boosts', 0)
    alive = obs.get('alive', True)
    
    if not alive or my_head is None:
        return [False] * 8
    
    mask = []
    
    for direction in DIRECTIONS:
        # Check no-boost action
        safe_no_boost = _is_action_safe(
            grid, my_head, direction, my_direction, width, height, steps=1
        )
        
        # Additional check: does it lead to deadlock?
        if safe_no_boost:
            final_pos = _simulate_move(my_head, direction, width, height, steps=1)
            area = _compute_reachable_area(grid, final_pos, width, height, max_depth=50)
            if area < min_area_threshold:
                safe_no_boost = False
        
        mask.append(safe_no_boost)
        
        # Check boost action
        has_boosts = my_boosts > 0
        safe_boost = has_boosts and _is_action_safe(
            grid, my_head, direction, my_direction, width, height, steps=2
        )
        
        # Additional check for boost: must not reduce area
        if safe_boost:
            final_pos = _simulate_move(my_head, direction, width, height, steps=2)
            area = _compute_reachable_area(grid, final_pos, width, height, max_depth=50)
            if area < min_area_threshold:
                safe_boost = False
        
        mask.append(safe_boost)
    
    # If all masked, allow least-bad (continuing current direction)
    if not any(mask):
        for i, direction in enumerate(DIRECTIONS):
            if direction == my_direction:
                mask[i * 2] = True  # Allow no-boost in current direction
                break
    
    return mask


def _is_action_safe(
    grid: List[List[int]],
    start: Tuple[int, int],
    direction: Direction,
    current_dir: Direction,
    width: int,
    height: int,
    steps: int,
) -> bool:
    """Check if action is safe (no immediate collision, not opposite direction)."""
    # Check opposite direction
    cur_dx, cur_dy = current_dir.value
    req_dx, req_dy = direction.value
    if (req_dx, req_dy) == (-cur_dx, -cur_dy):
        return False
    
    # Simulate steps
    x, y = start
    for _ in range(steps):
        dx, dy = direction.value
        x = (x + dx) % width
        y = (y + dy) % height
        
        if grid[y][x] == AGENT:
            return False
    
    return True


def _simulate_move(
    start: Tuple[int, int],
    direction: Direction,
    width: int,
    height: int,
    steps: int,
) -> Tuple[int, int]:
    """Simulate movement and return final position."""
    x, y = start
    for _ in range(steps):
        dx, dy = direction.value
        x = (x + dx) % width
        y = (y + dy) % height
    return (x, y)


def _compute_reachable_area(
    grid: List[List[int]],
    start: Tuple[int, int],
    width: int,
    height: int,
    max_depth: int = 100,
) -> int:
    """BFS flood-fill to compute reachable empty cells from start."""
    if grid[start[1]][start[0]] == AGENT:
        return 0
    
    visited = set([start])
    queue = deque([start])
    area = 0
    
    while queue and area < max_depth:
        x, y = queue.popleft()
        area += 1
        
        for direction in DIRECTIONS:
            dx, dy = direction.value
            nx = (x + dx) % width
            ny = (y + dy) % height
            
            if (nx, ny) in visited:
                continue
            
            if grid[ny][nx] == EMPTY:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return area


def detect_deadlock_risk(
    obs: Dict[str, Any],
    width: int = 20,
    height: int = 18,
    critical_area: int = 10,
) -> bool:
    """
    Detect if agent is at risk of deadlock (trapped with very low area).
    
    Returns:
        True if in critical deadlock risk
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    
    if my_head is None:
        return True
    
    area = _compute_reachable_area(grid, my_head, width, height, max_depth=critical_area + 5)
    return area <= critical_area


def find_escape_actions(
    obs: Dict[str, Any],
    width: int = 20,
    height: int = 18,
) -> List[Tuple[int, bool, int]]:
    """
    Find actions that maximize reachable area (escape from traps).
    
    Returns:
        List of (direction_idx, use_boost, area_score) sorted by area (descending)
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    my_direction = obs.get('my_direction', Direction.RIGHT)
    my_boosts = obs.get('my_boosts', 0)
    
    if my_head is None:
        return []
    
    escape_actions = []
    
    for dir_idx, direction in enumerate(DIRECTIONS):
        # Check no-boost
        if _is_action_safe(grid, my_head, direction, my_direction, width, height, steps=1):
            final_pos = _simulate_move(my_head, direction, width, height, steps=1)
            area = _compute_reachable_area(grid, final_pos, width, height, max_depth=100)
            escape_actions.append((dir_idx, False, area))
        
        # Check boost
        if my_boosts > 0 and _is_action_safe(grid, my_head, direction, my_direction, width, height, steps=2):
            final_pos = _simulate_move(my_head, direction, width, height, steps=2)
            area = _compute_reachable_area(grid, final_pos, width, height, max_depth=100)
            escape_actions.append((dir_idx, True, area))
    
    # Sort by area (descending)
    escape_actions.sort(key=lambda x: x[2], reverse=True)
    
    return escape_actions


def apply_safety_shield(
    obs: Dict[str, Any],
    q_values: Optional[List[float]],
    width: int = 20,
    height: int = 18,
    conservative_boost: bool = True,
    boost_area_margin: float = 5.0,
) -> Tuple[int, bool]:
    """
    Apply safety shield to select safe action.
    
    Args:
        obs: Observation dict
        q_values: Optional Q-values for 8 actions (or None for heuristic)
        width, height: Grid dimensions
        conservative_boost: If True, only boost if it increases area
        boost_area_margin: Min area gain required to justify boost
    
    Returns:
        (direction_idx, use_boost)
    """
    # Get safety mask
    mask = compute_safe_action_mask(obs, width, height)
    
    # Check if in deadlock risk
    in_deadlock = detect_deadlock_risk(obs, width, height)
    
    if in_deadlock:
        # Escape mode: prioritize area maximization
        escape_actions = find_escape_actions(obs, width, height)
        if escape_actions:
            dir_idx, use_boost, _ = escape_actions[0]
            return dir_idx, use_boost
    
    # Normal mode: use Q-values or heuristic
    valid_indices = [i for i, safe in enumerate(mask) if safe]
    
    if not valid_indices:
        # Emergency: pick least-bad (current direction, no boost)
        my_direction = obs.get('my_direction', Direction.RIGHT)
        for i, direction in enumerate(DIRECTIONS):
            if direction == my_direction:
                return i, False
        return 0, False  # Default to UP, no boost
    
    if q_values is not None:
        # Pick best valid action by Q-value
        best_idx = max(valid_indices, key=lambda i: q_values[i])
    else:
        # Heuristic: pick action with max area
        grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
        my_head = obs.get('my_head')
        my_direction = obs.get('my_direction', Direction.RIGHT)
        
        best_idx = valid_indices[0]
        best_area = -1
        
        for idx in valid_indices:
            dir_idx = idx // 2
            use_boost = (idx % 2 == 1)
            direction = DIRECTIONS[dir_idx]
            steps = 2 if use_boost else 1
            
            if _is_action_safe(grid, my_head, direction, my_direction, width, height, steps):
                final_pos = _simulate_move(my_head, direction, width, height, steps)
                area = _compute_reachable_area(grid, final_pos, width, height, max_depth=100)
                
                if area > best_area:
                    best_area = area
                    best_idx = idx
    
    # Decode action
    dir_idx = best_idx // 2
    use_boost = (best_idx % 2 == 1)
    
    # Conservative boost check
    if use_boost and conservative_boost:
        grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
        my_head = obs.get('my_head')
        direction = DIRECTIONS[dir_idx]
        
        # Compare boost vs no-boost area
        pos_boost = _simulate_move(my_head, direction, width, height, steps=2)
        area_boost = _compute_reachable_area(grid, pos_boost, width, height, max_depth=100)
        
        pos_no_boost = _simulate_move(my_head, direction, width, height, steps=1)
        area_no_boost = _compute_reachable_area(grid, pos_no_boost, width, height, max_depth=100)
        
        if area_boost <= area_no_boost + boost_area_margin:
            use_boost = False
    
    return dir_idx, use_boost


def format_action_string(dir_idx: int, use_boost: bool) -> str:
    """Convert action to judge-compatible string."""
    direction = DIRECTIONS[dir_idx]
    dir_str = DIR_TO_STR[direction]
    
    if use_boost:
        return f"{dir_str}:BOOST"
    return dir_str

