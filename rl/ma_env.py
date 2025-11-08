"""
Multi-agent Tron environment for N agents on 18x20 torus grid.
Supports boosts, simultaneous moves, and 500-turn episodes.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import random

from case_closed_game import Direction, EMPTY, AGENT

# Direction mappings
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
DIR_TO_IDX = {Direction.UP: 0, Direction.DOWN: 1, Direction.LEFT: 2, Direction.RIGHT: 3}


class MAAgent:
    """Multi-agent version of Agent with tracking for blocking/trap metrics."""
    
    def __init__(self, agent_id: int, start_pos: Tuple[int, int], start_dir: Direction, width: int, height: int):
        self.agent_id = agent_id
        self.width = width
        self.height = height
        
        # Initialize trail with 2 cells
        second = self._torus_wrap(start_pos[0] + start_dir.value[0], start_pos[1] + start_dir.value[1])
        self.trail = deque([start_pos, second])
        self.direction = start_dir
        self.alive = True
        self.length = 2
        self.boosts_remaining = 3
        
        # Metrics
        self.turns_survived = 0
        self.blocks_caused = 0  # Opponents killed by hitting our trail
        
    def _torus_wrap(self, x: int, y: int) -> Tuple[int, int]:
        return (x % self.width, y % self.height)
    
    def get_head(self) -> Tuple[int, int]:
        return self.trail[-1] if self.trail else (0, 0)
    
    def is_head(self, pos: Tuple[int, int]) -> bool:
        return pos == self.get_head()


class MultiAgentTronEnv:
    """
    N-agent Tron environment with:
    - 18x20 torus grid
    - Simultaneous moves
    - Boosts (3 per agent)
    - 500 turn limit
    - Per-agent observations, rewards, dones
    """
    
    def __init__(self, num_agents: int = 6, height: int = 18, width: int = 20, max_turns: int = 500):
        self.num_agents = num_agents
        self.height = height
        self.width = width
        self.max_turns = max_turns
        
        self.grid: List[List[int]] = []
        self.agents: List[MAAgent] = []
        self.turn_count = 0
        
        # Tracking for rewards
        self.prev_opponent_areas: Dict[int, int] = {}
        
    def reset(self) -> Tuple[List[Dict[str, Any]], List[List[bool]]]:
        """Reset environment and return initial observations and action masks."""
        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.agents = []
        self.turn_count = 0
        self.prev_opponent_areas = {}
        
        # Spawn agents in spread-out positions
        start_positions = self._generate_start_positions()
        start_directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for i in range(self.num_agents):
            start_pos = start_positions[i]
            start_dir = start_directions[i % len(start_directions)]
            agent = MAAgent(i, start_pos, start_dir, self.width, self.height)
            self.agents.append(agent)
            
            # Mark initial trail on grid
            for pos in agent.trail:
                self.grid[pos[1]][pos[0]] = AGENT
        
        observations = self._get_observations()
        action_masks = self._get_action_masks()
        
        return observations, action_masks
    
    def _generate_start_positions(self) -> List[Tuple[int, int]]:
        """Generate spread-out starting positions for N agents."""
        positions = []
        
        if self.num_agents <= 4:
            # Corners
            corners = [(2, 2), (self.width - 3, 2), (2, self.height - 3), (self.width - 3, self.height - 3)]
            positions = corners[:self.num_agents]
        else:
            # Distribute around perimeter
            for i in range(self.num_agents):
                angle_frac = i / self.num_agents
                if angle_frac < 0.25:
                    # Top edge
                    x = int(2 + (self.width - 4) * (angle_frac / 0.25))
                    y = 2
                elif angle_frac < 0.5:
                    # Right edge
                    x = self.width - 3
                    y = int(2 + (self.height - 4) * ((angle_frac - 0.25) / 0.25))
                elif angle_frac < 0.75:
                    # Bottom edge
                    x = int(self.width - 3 - (self.width - 4) * ((angle_frac - 0.5) / 0.25))
                    y = self.height - 3
                else:
                    # Left edge
                    x = 2
                    y = int(self.height - 3 - (self.height - 4) * ((angle_frac - 0.75) / 0.25))
                positions.append((x, y))
        
        return positions
    
    def step(self, actions: List[Tuple[int, bool]]) -> Tuple[
        List[Dict[str, Any]], 
        List[float], 
        List[bool], 
        List[List[bool]],
        Dict[str, Any]
    ]:
        """
        Execute one step with simultaneous moves.
        
        Args:
            actions: List of (direction_idx, use_boost) for each agent
        
        Returns:
            observations, rewards, dones, action_masks, info
        """
        self.turn_count += 1
        
        # Store previous reachable areas for reward calculation
        self.prev_opponent_areas = {}
        for agent in self.agents:
            if agent.alive:
                self.prev_opponent_areas[agent.agent_id] = self._compute_reachable_area(agent.get_head(), [])
        
        # Execute moves simultaneously
        new_heads: List[Optional[Tuple[int, int]]] = [None] * self.num_agents
        move_paths: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_agents)]
        
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            
            dir_idx, use_boost = actions[i]
            direction = DIRECTIONS[dir_idx]
            
            # Check boost validity
            if use_boost and agent.boosts_remaining <= 0:
                use_boost = False
            
            steps = 2 if use_boost else 1
            if use_boost:
                agent.boosts_remaining -= 1
            
            # Simulate movement
            head = agent.get_head()
            path = []
            final_head = head
            survived = True
            
            for step in range(steps):
                dx, dy = direction.value
                next_x = (final_head[0] + dx) % self.width
                next_y = (final_head[1] + dy) % self.height
                next_pos = (next_x, next_y)
                
                # Check collision with existing trails (but not with new heads yet)
                if self.grid[next_y][next_x] == AGENT:
                    survived = False
                    break
                
                path.append(next_pos)
                final_head = next_pos
            
            if survived:
                new_heads[i] = final_head
                move_paths[i] = path
                agent.direction = direction
            else:
                agent.alive = False
        
        # Check for head-on collisions (multiple agents entering same cell)
        head_collisions: Dict[Tuple[int, int], List[int]] = {}
        for i, head in enumerate(new_heads):
            if head is not None:
                if head not in head_collisions:
                    head_collisions[head] = []
                head_collisions[head].append(i)
        
        # Kill agents in head-on collisions
        for pos, agent_ids in head_collisions.items():
            if len(agent_ids) > 1:
                for aid in agent_ids:
                    self.agents[aid].alive = False
                    new_heads[aid] = None
        
        # Update trails and grid
        for i, agent in enumerate(self.agents):
            if new_heads[i] is not None:
                for pos in move_paths[i]:
                    agent.trail.append(pos)
                    agent.length += 1
                    self.grid[pos[1]][pos[0]] = AGENT
                agent.turns_survived += 1
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check episode termination
        alive_count = sum(1 for a in self.agents if a.alive)
        episode_done = (alive_count <= 1) or (self.turn_count >= self.max_turns)
        dones = [not agent.alive or episode_done for agent in self.agents]
        
        # Get new observations and masks
        observations = self._get_observations()
        action_masks = self._get_action_masks()
        
        info = {
            'turn': self.turn_count,
            'alive_count': alive_count,
            'episode_done': episode_done,
        }
        
        return observations, rewards, dones, action_masks, info
    
    def _get_observations(self) -> List[Dict[str, Any]]:
        """Generate per-agent observations."""
        observations = []
        
        for agent in self.agents:
            obs = {
                'agent_id': agent.agent_id,
                'grid': [row[:] for row in self.grid],  # Copy
                'my_head': agent.get_head() if agent.alive else None,
                'my_trail': list(agent.trail) if agent.alive else [],
                'my_boosts': agent.boosts_remaining,
                'my_direction': agent.direction,
                'opponent_heads': [a.get_head() for a in self.agents if a.alive and a.agent_id != agent.agent_id],
                'opponent_trails': [list(a.trail) for a in self.agents if a.alive and a.agent_id != agent.agent_id],
                'turn_count': self.turn_count,
                'alive': agent.alive,
            }
            observations.append(obs)
        
        return observations
    
    def _get_action_masks(self) -> List[List[bool]]:
        """Generate per-agent action masks (8 actions: 4 dirs × {no-boost, boost})."""
        masks = []
        
        for agent in self.agents:
            if not agent.alive:
                masks.append([False] * 8)
                continue
            
            mask = []
            head = agent.get_head()
            
            for dir_idx, direction in enumerate(DIRECTIONS):
                # Check no-boost action
                valid_no_boost = self._is_action_safe(head, direction, agent.direction, steps=1)
                mask.append(valid_no_boost)
                
                # Check boost action
                has_boosts = agent.boosts_remaining > 0
                valid_boost = has_boosts and self._is_action_safe(head, direction, agent.direction, steps=2)
                mask.append(valid_boost)
            
            # If all masked, allow at least one (least-bad)
            if not any(mask):
                # Allow continuing current direction without boost as fallback
                current_idx = DIR_TO_IDX.get(agent.direction, 0)
                mask[current_idx * 2] = True
            
            masks.append(mask)
        
        return masks
    
    def _is_action_safe(self, start: Tuple[int, int], direction: Direction, current_dir: Direction, steps: int) -> bool:
        """Check if an action is safe (doesn't immediately collide)."""
        # Check if opposite direction
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            return False
        
        # Simulate steps
        x, y = start
        for _ in range(steps):
            dx, dy = direction.value
            x = (x + dx) % self.width
            y = (y + dy) % self.height
            
            if self.grid[y][x] == AGENT:
                return False
        
        return True
    
    def _compute_rewards(self) -> List[float]:
        """Compute per-agent rewards based on survival, blocking, area control."""
        rewards = [0.0] * self.num_agents
        
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                rewards[i] = -1.0
                continue
            
            # Base survival reward
            rewards[i] += 0.01
            
            # Area control reward
            current_area = self._compute_reachable_area(agent.get_head(), [])
            
            # Penalize low area (deadlock risk) - less aggressive
            if current_area < 20:
                rewards[i] -= 0.05
            
            # Reward for reducing opponent area (blocking)
            for opp in self.agents:
                if opp.agent_id == agent.agent_id or not opp.alive:
                    continue
                
                prev_area = self.prev_opponent_areas.get(opp.agent_id, 0)
                curr_opp_area = self._compute_reachable_area(opp.get_head(), [])
                
                if prev_area > 0:
                    area_reduction = (prev_area - curr_opp_area) / max(prev_area, 1)
                    if area_reduction > 0.05:  # Significant blocking
                        rewards[i] += min(0.05 * area_reduction, 0.05)
        
        # Check if episode ended
        alive_count = sum(1 for a in self.agents if a.alive)
        episode_done = (alive_count <= 1) or (self.turn_count >= self.max_turns)
        
        if episode_done:
            for i, agent in enumerate(self.agents):
                if agent.alive:
                    rewards[i] += 2.0  # Win bonus (less aggressive than 5.0)
                else:
                    rewards[i] -= 2.0  # Loss penalty (less aggressive than 5.0)
        
        # Credit blocks caused (opponent died by hitting our trail)
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                # Check if this agent died by hitting another agent's trail
                for j, other in enumerate(self.agents):
                    if i != j and other.alive:
                        # Simple heuristic: if agent died and other is alive, credit partial block
                        rewards[j] += 0.3 / max(alive_count, 1)
        
        return rewards
    
    def _compute_reachable_area(self, start: Tuple[int, int], additional_blocked: List[Tuple[int, int]]) -> int:
        """BFS flood-fill to compute reachable area from start position."""
        from collections import deque
        
        blocked_set = set(additional_blocked)
        visited = set([start])
        queue = deque([start])
        area = 0
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) in blocked_set or self.grid[y][x] == AGENT:
                continue
            
            area += 1
            
            for direction in DIRECTIONS:
                dx, dy = direction.value
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return area
    
    def render(self) -> str:
        """Simple text rendering of the grid."""
        display = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark trails
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == AGENT:
                    display[y][x] = '#'
        
        # Mark heads
        for i, agent in enumerate(self.agents):
            if agent.alive:
                hx, hy = agent.get_head()
                display[hy][hx] = str(i)
        
        lines = [''.join(row) for row in display]
        return '\n'.join(lines)

