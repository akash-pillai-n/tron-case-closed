from __future__ import annotations

import os
import random
from typing import Dict, Any, Tuple

try:
    import torch
except Exception:
    torch = None

from case_closed_game import Direction
from rl.env_wrapper import CaseClosedEnv
from rl.model import DQN

# Match training action layout
ACTIONS = [
    (Direction.UP, False),
    (Direction.DOWN, False),
    (Direction.LEFT, False),
    (Direction.RIGHT, False),
    (Direction.UP, True),
    (Direction.DOWN, True),
    (Direction.LEFT, True),
    (Direction.RIGHT, True),
]


def encode_state(state: Dict[str, Any]):
    height = len(state["board"])
    width = len(state["board"][0]) if height > 0 else 0
    channels = 4
    grid = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]
    my_trail = state["agent1_trail"]
    opp_trail = state["agent2_trail"]
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
    scalars = [
        float(state.get("agent1_boosts", 0)),
        float(state.get("agent2_boosts", 0)),
        min(float(state.get("turn_count", 0)) / 500.0, 1.0),
    ]
    return grid, scalars


def opponent_policy_simple(state: Dict[str, Any]) -> Tuple[Direction, bool]:
    # Same as training opponent
    trail = state["agent2_trail"]
    if len(trail) >= 2:
        (x2, y2) = trail[-1]
        (x1, y1) = trail[-2]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 1:
            dx = -1 if dx > 0 else 1
        if abs(dy) > 1:
            dy = -1 if dy > 0 else 1
        if dx == 1:
            return Direction.RIGHT, False
        if dx == -1:
            return Direction.LEFT, False
        if dy == 1:
            return Direction.DOWN, False
        if dy == -1:
            return Direction.UP, False
    return random.choice(ACTIONS[:4])


def main():
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate. Please install CPU-only PyTorch locally.")

    # Load checkpoint
    ckpt_path = os.path.join("checkpoints", "latest.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    data = torch.load(ckpt_path, map_location="cpu")
    in_channels = int(data.get("in_channels", 4))
    num_actions = int(data.get("num_actions", 8))
    num_scalars = int(data.get("num_scalars", 3))
    model = DQN(in_channels=in_channels, num_actions=num_actions, num_scalar_features=num_scalars)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    env = CaseClosedEnv(max_turns=500)
    episodes = 20
    wins = 0
    losses = 0
    draws = 0
    for ep in range(episodes):
        state = env.reset()
        while True:
            grid, scalars = encode_state(state)
            with torch.no_grad():
                grid_t = torch.as_tensor(grid, dtype=torch.float32).unsqueeze(0)
                scalars_t = torch.as_tensor(scalars, dtype=torch.float32).unsqueeze(0)
                q_values = model(grid_t, scalars_t).squeeze(0)
                action_idx = int(torch.argmax(q_values).item())
            my_dir, my_boost = ACTIONS[action_idx]
            opp_dir, opp_boost = opponent_policy_simple(state)
            result = env.step(my_dir, opp_dir, my_boost, opp_boost)
            state = result.next_state
            if result.done:
                res = result.info.get("result")
                if res == "AGENT1_WIN":
                    wins += 1
                elif res == "AGENT2_WIN":
                    losses += 1
                else:
                    draws += 1
                break
        print(f"Episode {ep+1}/{episodes} => W:{wins} L:{losses} D:{draws}")
    print(f"Final: W:{wins} L:{losses} D:{draws}")


if __name__ == "__main__":
    main()


