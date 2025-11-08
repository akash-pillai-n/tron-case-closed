from __future__ import annotations

import os
import random
import time
from typing import Dict, Any, List, Tuple

try:
    import torch
    from torch import nn
    from torch.optim import Adam
except Exception:
    torch = None
    nn = None  # type: ignore
    Adam = None  # type: ignore

from case_closed_game import Direction
from rl.env_wrapper import CaseClosedEnv
from rl.model import DQN
from rl.replay_buffer import ReplayBuffer, Transition


# Discrete action space: 8 actions (4 dirs x {no-boost, boost})
ACTIONS: List[Tuple[Direction, bool]] = [
    (Direction.UP, False),
    (Direction.DOWN, False),
    (Direction.LEFT, False),
    (Direction.RIGHT, False),
    (Direction.UP, True),
    (Direction.DOWN, True),
    (Direction.LEFT, True),
    (Direction.RIGHT, True),
]


def encode_state(state: Dict[str, Any]) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Build grid feature planes and scalar features.
      Channels:
        0: my trail (1 where occupied by agent1)
        1: opponent trail (1 where occupied by agent2)
        2: my head (1 at head location)
        3: opponent head (1 at head location)
      Scalars:
        boosts_remaining (mine), opp_boosts_remaining, normalized_turn_count
    Returns:
      (grid_planes[C][H][W], scalars[S])
    """
    height = len(state["board"])
    width = len(state["board"][0]) if height > 0 else 0
    channels = 4
    grid = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]

    my_trail = state["agent1_trail"]
    opp_trail = state["agent2_trail"]
    if my_trail:
        for (x, y) in my_trail:
            grid[0][y][x] = 1
        head_x, head_y = my_trail[-1]
        grid[2][head_y][head_x] = 1
    if opp_trail:
        for (x, y) in opp_trail:
            grid[1][y][x] = 1
        head_x, head_y = opp_trail[-1]
        grid[3][head_y][head_x] = 1

    turn_count = float(state.get("turn_count", 0))
    scalars = [
        float(state.get("agent1_boosts", 0)),
        float(state.get("agent2_boosts", 0)),
        min(turn_count / 500.0, 1.0),
    ]
    return grid, scalars


def select_action(
    qnet: DQN,
    grid,
    scalars,
    epsilon: float,
    device: str = "cpu",
) -> int:
    if torch is None:
        # Fallback random if torch not installed (training expects torch)
        return random.randrange(len(ACTIONS))
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    with torch.no_grad():
        grid_t = torch.as_tensor(grid, dtype=torch.float32, device=device).unsqueeze(0)  # (1, C, H, W)
        scalars_t = torch.as_tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)  # (1, S)
        q_values = qnet(grid_t, scalars_t)  # (1, A)
        action = int(torch.argmax(q_values, dim=1).item())
    return action


def opponent_policy_simple(state: Dict[str, Any]) -> Tuple[Direction, bool]:
    """
    A lightweight scripted opponent:
      - Prefer continuing direction; otherwise pick a random direction without boost.
      - No boost usage.
    """
    # We do not have direction directly; approximate by last two trail points
    trail = state["agent2_trail"]
    if len(trail) >= 2:
        (x2, y2) = trail[-1]
        (x1, y1) = trail[-2]
        dx, dy = x2 - x1, y2 - y1
        # Normalize wrap deltas for torus (18x20 grid → width=20, height=18)
        width = len(state["board"][0])
        height = len(state["board"])
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
    # Default random
    return random.choice(ACTIONS[:4])


def main():
    if torch is None:
        raise RuntimeError("PyTorch is required to run training. Please install CPU-only PyTorch locally.")

    device = "cpu"
    random.seed(0)
    torch.manual_seed(0)

    env = CaseClosedEnv(max_turns=500)
    replay = ReplayBuffer(capacity=50_000)

    in_channels = 4  # as defined in encode_state
    num_actions = len(ACTIONS)
    num_scalars = 3

    qnet = DQN(in_channels=in_channels, num_actions=num_actions, num_scalar_features=num_scalars).to(device)
    target = DQN(in_channels=in_channels, num_actions=num_actions, num_scalar_features=num_scalars).to(device)
    target.load_state_dict(qnet.state_dict())
    target.eval()

    optimizer = Adam(qnet.parameters(), lr=2e-4)

    # Training settings (keep small by default; tune offline)
    total_episodes = 100
    max_steps_per_episode = 600  # env caps at 500
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 80
    target_update_interval = 500  # steps

    global_step = 0
    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(1, total_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        # Epsilon schedule (per-episode linear anneal)
        t = min(episode / float(max(1, epsilon_decay_episodes)), 1.0)
        epsilon = epsilon_start + t * (epsilon_end - epsilon_start)

        for step in range(max_steps_per_episode):
            grid, scalars = encode_state(state)
            action_idx = select_action(qnet, grid, scalars, epsilon, device=device)
            p1_dir, p1_boost = ACTIONS[action_idx]

            opp_dir, opp_boost = opponent_policy_simple(state)

            step_result = env.step(p1_dir, opp_dir, p1_boost, opp_boost)
            next_state = step_result.next_state
            reward = step_result.reward
            done = step_result.done

            replay.push(Transition(state=state, action=action_idx, reward=reward, next_state=next_state, done=done))
            state = next_state
            ep_reward += reward
            global_step += 1

            # Optimize
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)

                # Build tensors
                grid_batch = []
                scalar_batch = []
                action_batch = []
                reward_batch = []
                grid_next_batch = []
                scalar_next_batch = []
                done_batch = []

                for tr in batch:
                    g, s = encode_state(tr.state)
                    grid_batch.append(g)
                    scalar_batch.append(s)
                    action_batch.append(tr.action)
                    reward_batch.append(tr.reward)
                    g2, s2 = encode_state(tr.next_state)
                    grid_next_batch.append(g2)
                    scalar_next_batch.append(s2)
                    done_batch.append(1.0 if tr.done else 0.0)

                grid_t = torch.as_tensor(grid_batch, dtype=torch.float32, device=device)  # (B, C, H, W)
                scalars_t = torch.as_tensor(scalar_batch, dtype=torch.float32, device=device)  # (B, S)
                actions_t = torch.as_tensor(action_batch, dtype=torch.int64, device=device)
                rewards_t = torch.as_tensor(reward_batch, dtype=torch.float32, device=device)
                grid_next_t = torch.as_tensor(grid_next_batch, dtype=torch.float32, device=device)
                scalars_next_t = torch.as_tensor(scalar_next_batch, dtype=torch.float32, device=device)
                done_t = torch.as_tensor(done_batch, dtype=torch.float32, device=device)

                q_values = qnet(grid_t, scalars_t).gather(1, actions_t.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(grid_next_t, scalars_next_t).max(dim=1).values
                    target_q = rewards_t + (1.0 - done_t) * gamma * next_q

                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=1.0)
                optimizer.step()

                # Target update
                if global_step % target_update_interval == 0:
                    target.load_state_dict(qnet.state_dict())

            if done:
                break

        # Save checkpoint periodically
        if episode % 5 == 0:
            ckpt_path = os.path.join("checkpoints", "latest.pt")
            torch.save(
                {
                    "model_state_dict": qnet.state_dict(),
                    "in_channels": in_channels,
                    "num_actions": num_actions,
                    "num_scalars": num_scalars,
                },
                ckpt_path,
            )
            print(f"[Episode {episode}] Reward={ep_reward:.2f} | Saved {ckpt_path}")
        else:
            print(f"[Episode {episode}] Reward={ep_reward:.2f}")

    # Final save
    ckpt_path = os.path.join("checkpoints", "final.pt")
    torch.save(
        {
            "model_state_dict": qnet.state_dict(),
            "in_channels": in_channels,
            "num_actions": num_actions,
            "num_scalars": num_scalars,
        },
        ckpt_path,
    )
    print(f"Training complete. Saved {ckpt_path}")


if __name__ == "__main__":
    main()


