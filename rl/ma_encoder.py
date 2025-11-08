"""
Multi-agent observation encoder with occupancy, heads, danger maps, and scalars.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

try:
    import torch
except ImportError:
    torch = None

from case_closed_game import Direction, AGENT, EMPTY


DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


def encode_ma_observation(
    obs: Dict[str, Any],
    height: int = 18,
    width: int = 20,
    include_danger_maps: bool = True,
    include_local_crop: bool = False,
    crop_size: int = 7,
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Encode multi-agent observation into grid channels and scalar features.
    
    Channels (C×H×W):
    0: Occupancy (all trails)
    1: My head
    2: Opponent heads (all opponents)
    3: My trail (optional, helps with self-awareness)
    4: Danger map 1-step (cells opponents can enter next tick)
    5: Danger map 2-step (conservative boost hazard, optional)
    
    Scalars:
    - my_boosts / 3.0
    - turn_count / 500.0
    - local_density (empty cells in 5×5 around head / 25.0)
    - min_distance_to_hazard / max(H, W)
    
    Returns:
        (grid_channels, scalars)
    """
    grid = obs.get('grid', [[EMPTY] * width for _ in range(height)])
    my_head = obs.get('my_head')
    my_trail = obs.get('my_trail', [])
    my_boosts = obs.get('my_boosts', 0)
    opponent_heads = obs.get('opponent_heads', [])
    turn_count = obs.get('turn_count', 0)
    alive = obs.get('alive', True)
    
    # Initialize channels
    num_channels = 6 if include_danger_maps else 4
    channels = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(num_channels)]
    
    # Channel 0: Occupancy (all trails)
    for y in range(height):
        for x in range(width):
            if grid[y][x] == AGENT:
                channels[0][y][x] = 1
    
    # Channel 1: My head
    if my_head is not None and alive:
        hx, hy = my_head
        channels[1][hy][hx] = 1
    
    # Channel 2: Opponent heads
    for opp_head in opponent_heads:
        ox, oy = opp_head
        channels[2][oy][ox] = 1
    
    # Channel 3: My trail
    for tx, ty in my_trail:
        channels[3][ty][tx] = 1
    
    # Channel 4: Danger map 1-step (cells opponents can enter next tick)
    if include_danger_maps and len(channels) > 4:
        danger_1step = _compute_danger_map(opponent_heads, grid, width, height, steps=1)
        channels[4] = danger_1step
    
    # Channel 5: Danger map 2-step (boost hazard)
    if include_danger_maps and len(channels) > 5:
        danger_2step = _compute_danger_map(opponent_heads, grid, width, height, steps=2)
        channels[5] = danger_2step
    
    # Compute scalars
    scalars = []
    
    # Boosts normalized
    scalars.append(float(my_boosts) / 3.0)
    
    # Turn count normalized
    scalars.append(min(float(turn_count) / 500.0, 1.0))
    
    # Local density (5×5 around head)
    local_density = 0.0
    if my_head is not None:
        hx, hy = my_head
        empty_count = 0
        total_count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx = (hx + dx) % width
                ny = (hy + dy) % height
                total_count += 1
                if grid[ny][nx] == EMPTY:
                    empty_count += 1
        local_density = float(empty_count) / max(total_count, 1)
    scalars.append(local_density)
    
    # Min distance to hazard (nearest trail or opponent head)
    min_dist = float(max(height, width))
    if my_head is not None:
        hx, hy = my_head
        for y in range(height):
            for x in range(width):
                if grid[y][x] == AGENT and (x, y) not in my_trail:
                    dist = _torus_distance(hx, hy, x, y, width, height)
                    min_dist = min(min_dist, dist)
        for ox, oy in opponent_heads:
            dist = _torus_distance(hx, hy, ox, oy, width, height)
            min_dist = min(min_dist, dist)
    scalars.append(min_dist / max(height, width))
    
    return channels, scalars


def _compute_danger_map(
    opponent_heads: List[Tuple[int, int]],
    grid: List[List[int]],
    width: int,
    height: int,
    steps: int = 1,
) -> List[List[int]]:
    """
    Compute danger map: cells that opponents can reach in 'steps' moves.
    Conservative estimate (assumes opponents move optimally toward all directions).
    """
    danger = [[0 for _ in range(width)] for _ in range(height)]
    
    for opp_head in opponent_heads:
        ox, oy = opp_head
        # BFS from opponent head for 'steps' moves
        from collections import deque
        queue = deque([(ox, oy, 0)])
        visited = set([(ox, oy)])
        
        while queue:
            x, y, dist = queue.popleft()
            
            if dist >= steps:
                continue
            
            for direction in DIRECTIONS:
                dx, dy_delta = direction.value
                nx = (x + dx) % width
                ny = (y + dy_delta) % height
                
                if (nx, ny) in visited:
                    continue
                
                # Don't expand through walls (but mark them as reachable if within steps)
                if grid[ny][nx] != AGENT or dist + 1 <= steps:
                    danger[ny][nx] = 1
                    visited.add((nx, ny))
                    
                    if grid[ny][nx] == EMPTY:
                        queue.append((nx, ny, dist + 1))
    
    return danger


def _torus_distance(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> float:
    """Compute torus (wrapped) Manhattan distance."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dx = min(dx, width - dx)
    dy = min(dy, height - dy)
    return float(dx + dy)


def encode_ma_batch(
    observations: List[Dict[str, Any]],
    height: int = 18,
    width: int = 20,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """
    Encode a batch of multi-agent observations into tensors.
    
    Returns:
        (grid_tensor, scalars_tensor) if torch available, else (list, list)
    """
    batch_grids = []
    batch_scalars = []
    
    for obs in observations:
        grid, scalars = encode_ma_observation(obs, height, width)
        batch_grids.append(grid)
        batch_scalars.append(scalars)
    
    if torch is not None:
        grid_tensor = torch.tensor(batch_grids, dtype=torch.float32, device=device)
        scalars_tensor = torch.tensor(batch_scalars, dtype=torch.float32, device=device)
        return grid_tensor, scalars_tensor
    
    return batch_grids, batch_scalars


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
    from collections import deque
    
    voronoi = [[-1 for _ in range(width)] for _ in range(height)]
    
    # Multi-source BFS
    queue = deque()
    for agent_id, (hx, hy) in enumerate(agent_heads):
        if grid[hy][hx] == EMPTY:
            voronoi[hy][hx] = agent_id
            queue.append((hx, hy, agent_id))
    
    visited = set([(hx, hy) for hx, hy in agent_heads])
    
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


def detect_choke_points(
    grid: List[List[int]],
    my_head: Tuple[int, int],
    opponent_heads: List[Tuple[int, int]],
    width: int,
    height: int,
) -> List[Tuple[int, int, float]]:
    """
    Detect choke points: cells where moving there would significantly reduce opponent territory.
    
    Returns:
        List of (x, y, impact_score) where impact_score is estimated opponent area reduction.
    """
    choke_points = []
    
    # For each empty cell adjacent to my head
    hx, hy = my_head
    for direction in DIRECTIONS:
        dx, dy = direction.value
        nx = (hx + dx) % width
        ny = (hy + dy) % height
        
        if grid[ny][nx] != EMPTY:
            continue
        
        # Simulate placing a trail at (nx, ny)
        grid[ny][nx] = AGENT
        
        # Compute new Voronoi and check opponent territory reduction
        all_heads = [my_head] + opponent_heads
        voronoi_before = compute_voronoi_partition(grid, all_heads, width, height)
        territory_before = compute_territory_sizes(voronoi_before, len(all_heads))
        
        # Estimate impact (sum of opponent territory reductions)
        impact = 0.0
        for i in range(1, len(all_heads)):  # Skip my territory (index 0)
            impact += max(0, territory_before[i])
        
        # Restore grid
        grid[ny][nx] = EMPTY
        
        if impact > 0:
            choke_points.append((nx, ny, impact))
    
    # Sort by impact (descending)
    choke_points.sort(key=lambda x: x[2], reverse=True)
    
    return choke_points

