"""
Multi-agent aware agent logic with safety shield and blocking strategy.
Backward compatible with 2-agent judge payloads.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

try:
    import torch
    from rl.model import DQN
    from rl.ma_encoder import encode_ma_observation
    from rl.ma_safety_shield import apply_safety_shield, format_action_string
    from rl.ma_blocking import compute_aggressive_action_score
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    DQN = None
    TORCH_AVAILABLE = False

from case_closed_game import Direction


def convert_judge_state_to_ma_obs(state: Dict[str, Any], my_player_number: int) -> Dict[str, Any]:
    """
    Convert judge state payload to multi-agent observation format.
    Handles both 2-agent and N-agent scenarios.
    """
    board = state.get("board", [])
    turn_count = state.get("turn_count", 0)
    
    # Extract my info
    if my_player_number == 1:
        my_trail = state.get("agent1_trail", [])
        my_boosts = state.get("agent1_boosts", 0)
        my_alive = state.get("agent1_alive", True)
    else:
        my_trail = state.get("agent2_trail", [])
        my_boosts = state.get("agent2_boosts", 0)
        my_alive = state.get("agent2_alive", True)
    
    # Extract opponent info (handle 2-agent case)
    opponent_heads = []
    opponent_trails = []
    
    if my_player_number == 1:
        opp_trail = state.get("agent2_trail", [])
        if opp_trail:
            opponent_heads.append(opp_trail[-1])
            opponent_trails.append(opp_trail)
    else:
        opp_trail = state.get("agent1_trail", [])
        if opp_trail:
            opponent_heads.append(opp_trail[-1])
            opponent_trails.append(opp_trail)
    
    # Infer my direction from trail
    my_direction = Direction.RIGHT
    if len(my_trail) >= 2:
        x2, y2 = my_trail[-1]
        x1, y1 = my_trail[-2]
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize for torus wrap
        if abs(dx) > 1:
            dx = -1 if dx > 0 else 1
        if abs(dy) > 1:
            dy = -1 if dy > 0 else 1
        
        if dx == 1:
            my_direction = Direction.RIGHT
        elif dx == -1:
            my_direction = Direction.LEFT
        elif dy == 1:
            my_direction = Direction.DOWN
        elif dy == -1:
            my_direction = Direction.UP
    
    # Build MA observation
    obs = {
        'agent_id': my_player_number - 1,
        'grid': board,
        'my_head': my_trail[-1] if my_trail else None,
        'my_trail': my_trail,
        'my_boosts': my_boosts,
        'my_direction': my_direction,
        'opponent_heads': opponent_heads,
        'opponent_trails': opponent_trails,
        'turn_count': turn_count,
        'alive': my_alive,
    }
    
    return obs


def choose_move_ma(
    state: Dict[str, Any],
    my_player_number: int,
    qnet: Optional[DQN] = None,
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Choose move using multi-agent safety shield and optional Q-network.
    
    Args:
        state: Judge state payload
        my_player_number: 1 or 2
        qnet: Optional trained DQN
        device: torch device
        config: Optional config dict
    
    Returns:
        Move string (e.g., "UP", "RIGHT:BOOST")
    """
    # Convert to MA observation format
    obs = convert_judge_state_to_ma_obs(state, my_player_number)
    
    # Get Q-values if network available
    q_values = None
    if qnet is not None and TORCH_AVAILABLE:
        try:
            grid, scalars = encode_ma_observation(obs)
            with torch.no_grad():
                grid_t = torch.tensor([grid], dtype=torch.float32, device=device)
                scalars_t = torch.tensor([scalars], dtype=torch.float32, device=device)
                q_vals = qnet(grid_t, scalars_t).squeeze(0)
                q_values = q_vals.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error computing Q-values: {e}")
            q_values = None
    
    # Apply safety shield
    conservative_boost = True
    boost_margin = 5.0
    
    if config is not None:
        conservative_boost = config.get("conservative_boost", True)
        boost_margin = config.get("boost_area_margin", 5.0)
    
    dir_idx, use_boost = apply_safety_shield(
        obs,
        q_values,
        width=20,
        height=18,
        conservative_boost=conservative_boost,
        boost_area_margin=boost_margin,
    )
    
    # Format and return
    return format_action_string(dir_idx, use_boost)


def choose_move_ma_aggressive(
    state: Dict[str, Any],
    my_player_number: int,
    qnet: Optional[DQN] = None,
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
    blocking_weight: float = 0.5,
) -> str:
    """
    Choose move with aggressive blocking strategy.
    
    Prioritizes actions that reduce opponent territory while maintaining safety.
    """
    obs = convert_judge_state_to_ma_obs(state, my_player_number)
    
    # Get Q-values if available
    q_values = None
    if qnet is not None and TORCH_AVAILABLE:
        try:
            grid, scalars = encode_ma_observation(obs)
            with torch.no_grad():
                grid_t = torch.tensor([grid], dtype=torch.float32, device=device)
                scalars_t = torch.tensor([scalars], dtype=torch.float32, device=device)
                q_vals = qnet(grid_t, scalars_t).squeeze(0)
                q_values = q_vals.cpu().numpy().tolist()
        except Exception:
            q_values = None
    
    # Compute aggressive scores (area + blocking)
    from rl.ma_safety_shield import compute_safe_action_mask
    from rl.ma_blocking import compute_aggressive_action_score
    
    mask = compute_safe_action_mask(obs, width=20, height=18)
    valid_actions = [i for i, m in enumerate(mask) if m]
    
    if not valid_actions:
        # Fallback to safety shield
        dir_idx, use_boost = apply_safety_shield(obs, q_values, width=20, height=18)
        return format_action_string(dir_idx, use_boost)
    
    # Combine Q-values with aggressive scores
    best_action = valid_actions[0]
    best_score = -1e9
    
    for action_idx in valid_actions:
        # Aggressive score (area + blocking)
        agg_score = compute_aggressive_action_score(obs, action_idx, width=20, height=18)
        
        # Combine with Q-value if available
        if q_values is not None:
            combined_score = (1 - blocking_weight) * q_values[action_idx] + blocking_weight * agg_score
        else:
            combined_score = agg_score
        
        if combined_score > best_score:
            best_score = combined_score
            best_action = action_idx
    
    # Decode action
    dir_idx = best_action // 2
    use_boost = (best_action % 2 == 1)
    
    # Conservative boost check
    conservative_boost = config.get("conservative_boost", True) if config else True
    if use_boost and conservative_boost:
        from rl.ma_safety_shield import _simulate_move, _compute_reachable_area
        from rl.ma_safety_shield import DIRECTIONS
        
        grid = obs.get('grid', [])
        my_head = obs.get('my_head')
        direction = DIRECTIONS[dir_idx]
        
        if my_head and grid:
            pos_boost = _simulate_move(my_head, direction, 20, 18, steps=2)
            area_boost = _compute_reachable_area(grid, pos_boost, 20, 18, max_depth=100)
            
            pos_no_boost = _simulate_move(my_head, direction, 20, 18, steps=1)
            area_no_boost = _compute_reachable_area(grid, pos_no_boost, 20, 18, max_depth=100)
            
            boost_margin = config.get("boost_area_margin", 5.0) if config else 5.0
            if area_boost <= area_no_boost + boost_margin:
                use_boost = False
    
    return format_action_string(dir_idx, use_boost)

