"""
Blocking and trap strategy for multi-agent Tron.
Uses Voronoi partitioning and choke detection to actively trap opponents.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from case_closed_game import Direction, AGENT, EMPTY


DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


def compute_blocking_score(
    obs: Dict[str, Any],
    action_idx: int,
    width: int = 20,
    height: int = 18,
) -> float:
    """
    Compute blocking score for an action: how much it reduces opponent territory.
    
    Higher score = better blocking move.
    
    Returns:
        Blocking score (0.0 to 1.0+)
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    opponent_heads = obs.get('opponent_heads', [])
    
    if my_head is None or not opponent_heads:
        return 0.0
    
    # Decode action
    dir_idx = action_idx // 2
    use_boost = (action_idx % 2 == 1)
    direction = DIRECTIONS[dir_idx]
    steps = 2 if use_boost else 1
    
    # Simulate move
    new_positions = _simulate_move_path(my_head, direction, width, height, steps)
    
    # Compute opponent territories before and after
    all_heads = [my_head] + opponent_heads
    voronoi_before = compute_voronoi_partition(grid, all_heads, width, height)
    territory_before = compute_territory_sizes(voronoi_before, len(all_heads))
    
    # Simulate placing trail at new positions
    grid_after = [row[:] for row in grid]
    for pos in new_positions:
        grid_after[pos[1]][pos[0]] = AGENT
    
    # Update my head position
    all_heads_after = [new_positions[-1]] + opponent_heads
    voronoi_after = compute_voronoi_partition(grid_after, all_heads_after, width, height)
    territory_after = compute_territory_sizes(voronoi_after, len(all_heads_after))
    
    # Compute total opponent territory reduction
    total_reduction = 0.0
    for i in range(1, len(all_heads)):  # Skip index 0 (my territory)
        before = territory_before[i]
        after = territory_after[i]
        reduction = max(0, before - after)
        total_reduction += reduction
    
    # Normalize by total opponent territory
    total_opp_territory = sum(territory_before[1:])
    if total_opp_territory > 0:
        return total_reduction / total_opp_territory
    
    return 0.0


def find_best_blocking_action(
    obs: Dict[str, Any],
    valid_actions: List[int],
    width: int = 20,
    height: int = 18,
) -> Tuple[int, float]:
    """
    Find the action that maximizes blocking score among valid actions.
    
    Returns:
        (best_action_idx, blocking_score)
    """
    if not valid_actions:
        return 0, 0.0
    
    best_action = valid_actions[0]
    best_score = 0.0
    
    for action_idx in valid_actions:
        score = compute_blocking_score(obs, action_idx, width, height)
        if score > best_score:
            best_score = score
            best_action = action_idx
    
    return best_action, best_score


def detect_trap_opportunities(
    obs: Dict[str, Any],
    width: int = 20,
    height: int = 18,
    min_impact: float = 0.15,
) -> List[Tuple[int, int, int, float]]:
    """
    Detect trap opportunities: positions where we can significantly reduce opponent area.
    
    Returns:
        List of (x, y, opponent_idx, impact_score) sorted by impact (descending)
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    opponent_heads = obs.get('opponent_heads', [])
    
    if my_head is None or not opponent_heads:
        return []
    
    traps = []
    
    # For each opponent
    for opp_idx, opp_head in enumerate(opponent_heads):
        # Compute opponent's reachable area
        opp_area_before = _compute_reachable_area(grid, opp_head, width, height)
        
        # Check cells near my head that could block opponent
        for direction in DIRECTIONS:
            dx, dy = direction.value
            nx = (my_head[0] + dx) % width
            ny = (my_head[1] + dy) % height
            
            if grid[ny][nx] != EMPTY:
                continue
            
            # Simulate placing trail at (nx, ny)
            grid[ny][nx] = AGENT
            opp_area_after = _compute_reachable_area(grid, opp_head, width, height)
            grid[ny][nx] = EMPTY
            
            # Compute impact
            if opp_area_before > 0:
                impact = (opp_area_before - opp_area_after) / opp_area_before
                if impact >= min_impact:
                    traps.append((nx, ny, opp_idx, impact))
    
    # Sort by impact (descending)
    traps.sort(key=lambda x: x[3], reverse=True)
    
    return traps


def compute_voronoi_partition(
    grid: List[List[int]],
    agent_heads: List[Tuple[int, int]],
    width: int,
    height: int,
) -> List[List[int]]:
    """
    Compute Voronoi partition: assign each empty cell to nearest agent head.
    Returns grid where each cell contains agent_id (0-indexed) or -1 if occupied.
    """
    voronoi = [[-1 for _ in range(width)] for _ in range(height)]
    
    # Multi-source BFS
    queue = deque()
    for agent_id, (hx, hy) in enumerate(agent_heads):
        if 0 <= hx < width and 0 <= hy < height:
            if grid[hy][hx] == EMPTY:
                voronoi[hy][hx] = agent_id
                queue.append((hx, hy, agent_id))
    
    visited = set([(hx, hy) for hx, hy in agent_heads if 0 <= hx < width and 0 <= hy < height])
    
    while queue:
        x, y, agent_id = queue.popleft()
        
        for direction in DIRECTIONS:
            dx, dy = direction.value
            nx = (x + dx) % width
            ny = (y + dy) % height
            
            if (nx, ny) in visited:
                continue
            
            if grid[ny][nx] == EMPTY:
                voronoi[ny][nx] = agent_id
                visited.add((nx, ny))
                queue.append((nx, ny, agent_id))
    
    return voronoi


def compute_territory_sizes(voronoi: List[List[int]], num_agents: int) -> List[int]:
    """Count territory size for each agent from Voronoi partition."""
    sizes = [0] * num_agents
    for row in voronoi:
        for cell in row:
            if 0 <= cell < num_agents:
                sizes[cell] += 1
    return sizes


def identify_choke_points(
    grid: List[List[int]],
    my_head: Tuple[int, int],
    opponent_head: Tuple[int, int],
    width: int,
    height: int,
) -> List[Tuple[int, int, float]]:
    """
    Identify choke points between me and a specific opponent.
    
    Returns:
        List of (x, y, criticality) where criticality indicates how important the choke is
    """
    # Find shortest path between my head and opponent head
    path = _find_shortest_path(grid, my_head, opponent_head, width, height)
    
    if not path or len(path) < 3:
        return []
    
    chokes = []
    
    # Check each cell in the path
    for i in range(1, len(path) - 1):
        x, y = path[i]
        
        # Count adjacent empty cells (narrow = choke)
        adjacent_empty = 0
        for direction in DIRECTIONS:
            dx, dy = direction.value
            nx = (x + dx) % width
            ny = (y + dy) % height
            if grid[ny][nx] == EMPTY:
                adjacent_empty += 1
        
        # Lower adjacent_empty = higher criticality
        if adjacent_empty <= 2:
            criticality = 1.0 / max(adjacent_empty, 1)
            chokes.append((x, y, criticality))
    
    return chokes


def _find_shortest_path(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """BFS to find shortest path between two points."""
    if grid[start[1]][start[0]] == AGENT or grid[goal[1]][goal[0]] == AGENT:
        return []
    
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) == goal:
            return path
        
        for direction in DIRECTIONS:
            dx, dy = direction.value
            nx = (x + dx) % width
            ny = (y + dy) % height
            
            if (nx, ny) in visited:
                continue
            
            if grid[ny][nx] == EMPTY:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return []


def _simulate_move_path(
    start: Tuple[int, int],
    direction: Direction,
    width: int,
    height: int,
    steps: int,
) -> List[Tuple[int, int]]:
    """Simulate movement and return all positions visited."""
    positions = []
    x, y = start
    
    for _ in range(steps):
        dx, dy = direction.value
        x = (x + dx) % width
        y = (y + dy) % height
        positions.append((x, y))
    
    return positions


def _compute_reachable_area(
    grid: List[List[int]],
    start: Tuple[int, int],
    width: int,
    height: int,
    max_depth: int = 200,
) -> int:
    """BFS flood-fill to compute reachable empty cells."""
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


def compute_aggressive_action_score(
    obs: Dict[str, Any],
    action_idx: int,
    width: int = 20,
    height: int = 18,
    blocking_weight: float = 0.4,
    area_weight: float = 0.6,
) -> float:
    """
    Compute combined score: area control + blocking.
    
    Returns:
        Combined score (higher = better)
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    
    if my_head is None:
        return 0.0
    
    # Decode action
    dir_idx = action_idx // 2
    use_boost = (action_idx % 2 == 1)
    direction = DIRECTIONS[dir_idx]
    steps = 2 if use_boost else 1
    
    # Simulate move
    new_positions = _simulate_move_path(my_head, direction, width, height, steps)
    
    if not new_positions:
        return 0.0
    
    final_pos = new_positions[-1]
    
    # Area score
    area = _compute_reachable_area(grid, final_pos, width, height, max_depth=100)
    area_score = min(area / 100.0, 1.0)
    
    # Blocking score
    blocking_score = compute_blocking_score(obs, action_idx, width, height)
    
    # Combined score
    combined = area_weight * area_score + blocking_weight * blocking_score
    
    return combined

