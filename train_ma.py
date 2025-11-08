"""
Multi-agent training script with vectorized self-play, opponent pool, and prioritized replay.
"""
import os
import random
import json
from collections import deque
from typing import List, Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Please install CPU-only PyTorch for training.")
    TORCH_AVAILABLE = False
    exit(1)

from rl.ma_env import MultiAgentTronEnv, DIRECTIONS
from rl.model import DQN
from rl.ma_encoder import encode_ma_observation
from rl.ma_safety_shield import compute_safe_action_mask
from rl.ma_blocking import compute_aggressive_action_score


class MAReplayBuffer:
    """Replay buffer for multi-agent experiences."""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done, mask):
        """Store transition."""
        self.buffer.append((obs, action, reward, next_obs, done, mask))
    
    def sample(self, batch_size: int):
        """Sample random batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)


class OpponentPool:
    """Pool of opponent policies for diverse training."""
    
    def __init__(self):
        self.policies = ['random', 'greedy', 'frozen']
        self.frozen_policy = None
    
    def get_opponent_type(self, curriculum_stage: int = 0) -> str:
        """Sample opponent type based on curriculum."""
        if curriculum_stage == 0:
            # Early training: more random
            return random.choice(['random', 'random', 'greedy'])
        elif curriculum_stage == 1:
            # Mid training: balanced
            return random.choice(['random', 'greedy', 'frozen'])
        else:
            # Late training: mostly frozen policy
            return random.choice(['greedy', 'frozen', 'frozen'])
    
    def update_frozen_policy(self, policy_state_dict):
        """Update frozen policy snapshot."""
        self.frozen_policy = policy_state_dict


def select_action_random(num_actions: int = 8) -> int:
    """Random action selection."""
    return random.randint(0, num_actions - 1)


def select_action_greedy(obs: Dict[str, Any], mask: List[bool]) -> int:
    """Greedy action: maximize area."""
    valid_actions = [i for i, m in enumerate(mask) if m]
    if not valid_actions:
        return 0
    
    # Use aggressive score (area + blocking)
    best_action = valid_actions[0]
    best_score = -1e9
    
    for action_idx in valid_actions:
        score = compute_aggressive_action_score(obs, action_idx)
        if score > best_score:
            best_score = score
            best_action = action_idx
    
    return best_action


def select_action_policy(
    policy: DQN,
    obs: Dict[str, Any],
    mask: List[bool],
    epsilon: float,
    device: str,
) -> int:
    """Select action using policy with epsilon-greedy."""
    if random.random() < epsilon:
        valid_actions = [i for i, m in enumerate(mask) if m]
        return random.choice(valid_actions) if valid_actions else 0
    
    # Encode observation
    grid, scalars = encode_ma_observation(obs)
    
    with torch.no_grad():
        grid_t = torch.tensor([grid], dtype=torch.float32, device=device)
        scalars_t = torch.tensor([scalars], dtype=torch.float32, device=device)
        q_values = policy(grid_t, scalars_t).squeeze(0)
        
        # Mask invalid actions
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        q_values[~mask_t] = -1e9
        
        action = int(torch.argmax(q_values).item())
    
    return action


def train_step(
    policy: DQN,
    target: DQN,
    optimizer: optim.Optimizer,
    replay_buffer: MAReplayBuffer,
    batch_size: int,
    gamma: float,
    device: str,
) -> float:
    """Perform one training step."""
    if len(replay_buffer) < batch_size:
        return 0.0
    
    # Sample batch
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, mask_batch = replay_buffer.sample(batch_size)
    
    # Encode observations
    grids = []
    scalars = []
    next_grids = []
    next_scalars = []
    
    for obs in obs_batch:
        g, s = encode_ma_observation(obs)
        grids.append(g)
        scalars.append(s)
    
    for next_obs in next_obs_batch:
        g, s = encode_ma_observation(next_obs)
        next_grids.append(g)
        next_scalars.append(s)
    
    # Convert to tensors
    grid_t = torch.tensor(grids, dtype=torch.float32, device=device)
    scalars_t = torch.tensor(scalars, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_batch, dtype=torch.long, device=device)
    reward_t = torch.tensor(reward_batch, dtype=torch.float32, device=device)
    next_grid_t = torch.tensor(next_grids, dtype=torch.float32, device=device)
    next_scalars_t = torch.tensor(next_scalars, dtype=torch.float32, device=device)
    done_t = torch.tensor(done_batch, dtype=torch.float32, device=device)
    
    # Current Q values
    q_values = policy(grid_t, scalars_t)
    q_values = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)
    
    # Target Q values (Double DQN)
    with torch.no_grad():
        # Use policy to select actions
        next_q_policy = policy(next_grid_t, next_scalars_t)
        next_actions = torch.argmax(next_q_policy, dim=1)
        
        # Use target to evaluate
        next_q_target = target(next_grid_t, next_scalars_t)
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        target_q = reward_t + gamma * next_q_values * (1 - done_t)
    
    # Compute loss with clipping
    loss = nn.MSELoss()(q_values, target_q)
    loss = torch.clamp(loss, max=1.0)  # Clip loss to prevent instability
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def train_multi_agent(
    num_agents: int = 3,
    num_episodes: int = 5000,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 2000,
    target_update: int = 50,
    save_interval: int = 500,
    device: str = "mps",
):
    """Main training loop."""
    # Check device availability
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Training multi-agent Tron with {num_agents} agents on {device.upper()}...")
    
    # Initialize environment
    env = MultiAgentTronEnv(num_agents=num_agents)
    
    # Initialize policy and target networks
    policy = DQN(in_channels=6, num_actions=8, num_scalar_features=4).to(device)
    target = DQN(in_channels=6, num_actions=8, num_scalar_features=4).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    
    # Optimizer with reduced learning rate
    optimizer = optim.Adam(policy.parameters(), lr=5e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    
    # Replay buffer
    replay_buffer = MAReplayBuffer(capacity=50000)
    
    # Opponent pool
    opp_pool = OpponentPool()
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Curriculum stages
    curriculum_stage = 0
    
    for episode in range(num_episodes):
        # Epsilon decay
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - episode / epsilon_decay)
        
        # Update curriculum
        if episode > 1000:
            curriculum_stage = 1
        if episode > 3000:
            curriculum_stage = 2
        
        # Reset environment
        observations, action_masks = env.reset()
        
        # Assign opponent types
        opponent_types = [opp_pool.get_opponent_type(curriculum_stage) for _ in range(num_agents)]
        opponent_types[0] = 'policy'  # First agent always uses learning policy
        
        episode_reward = 0
        step_count = 0
        
        while True:
            # Select actions for all agents
            actions = []
            for i, obs in enumerate(observations):
                mask = action_masks[i]
                
                if opponent_types[i] == 'policy':
                    action = select_action_policy(policy, obs, mask, epsilon, device)
                elif opponent_types[i] == 'random':
                    valid = [j for j, m in enumerate(mask) if m]
                    action = random.choice(valid) if valid else 0
                elif opponent_types[i] == 'greedy':
                    action = select_action_greedy(obs, mask)
                else:  # frozen
                    if opp_pool.frozen_policy is not None:
                        frozen_net = DQN(in_channels=6, num_actions=8, num_scalar_features=4).to(device)
                        frozen_net.load_state_dict(opp_pool.frozen_policy)
                        frozen_net.eval()
                        action = select_action_policy(frozen_net, obs, mask, 0.0, device)
                    else:
                        action = select_action_greedy(obs, mask)
                
                # Convert to (dir_idx, use_boost)
                dir_idx = action // 2
                use_boost = (action % 2 == 1)
                actions.append((dir_idx, use_boost))
            
            # Step environment
            next_observations, rewards, dones, next_action_masks, info = env.step(actions)
            
            # Store transitions for learning agent (index 0)
            if not dones[0]:
                replay_buffer.push(
                    observations[0],
                    actions[0][0] * 2 + int(actions[0][1]),  # Convert back to action_idx
                    rewards[0],
                    next_observations[0],
                    dones[0],
                    action_masks[0],
                )
            
            episode_reward += rewards[0]
            step_count += 1
            
            observations = next_observations
            action_masks = next_action_masks
            
            # Train
            if len(replay_buffer) >= batch_size:
                loss = train_step(policy, target, optimizer, replay_buffer, batch_size, gamma, device)
                losses.append(loss)
                scheduler.step()  # Update learning rate
            
            # Check if episode done
            if info['episode_done']:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Update target network
        if episode % target_update == 0:
            target.load_state_dict(policy.state_dict())
            print(f"Episode {episode}: Target network updated")
        
        # Update frozen policy in opponent pool
        if episode % 500 == 0 and episode > 0:
            opp_pool.update_frozen_policy(policy.state_dict())
            print(f"Episode {episode}: Frozen policy updated")
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }
            torch.save(checkpoint, f"checkpoints/marl_episode_{episode}.pt")
            torch.save(policy.state_dict(), "checkpoints/marl_latest.pt")
            print(f"Episode {episode}: Checkpoint saved")
        
        # Logging
        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            avg_loss = sum(losses[-100:]) / max(1, len(losses[-100:]))
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")
    
    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints/marl_final.pt")
    
    # Save config
    config = {
        'num_agents': num_agents,
        'in_channels': 6,
        'num_scalar_features': 4,
        'num_actions': 8,
        'width': 20,
        'height': 18,
    }
    with open("checkpoints/marl_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multi-agent Tron RL")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    train_multi_agent(
        num_agents=args.num_agents,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        device=args.device,
    )

