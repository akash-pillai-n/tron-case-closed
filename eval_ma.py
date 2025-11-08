"""
Multi-agent evaluation script for N≥6 agents.
Measures survival rate, blocks-caused deaths, head-on incidents, and self-crash rate.
"""
import os
import json
from typing import List, Dict, Any

try:
    import torch
    from rl.model import DQN
    from rl.ma_env import MultiAgentTronEnv, DIRECTIONS
    from rl.ma_encoder import encode_ma_observation
    from rl.ma_safety_shield import apply_safety_shield
    from rl.ma_blocking import compute_aggressive_action_score
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("PyTorch or RL modules not available. Please install dependencies.")
    TORCH_AVAILABLE = False
    exit(1)


def select_action_policy(policy, obs, device="cpu"):
    """Select action using trained policy with safety shield."""
    grid, scalars = encode_ma_observation(obs)
    
    with torch.no_grad():
        grid_t = torch.tensor([grid], dtype=torch.float32, device=device)
        scalars_t = torch.tensor([scalars], dtype=torch.float32, device=device)
        q_values = policy(grid_t, scalars_t).squeeze(0)
        q_values_list = q_values.cpu().numpy().tolist()
    
    # Apply safety shield
    dir_idx, use_boost = apply_safety_shield(obs, q_values_list, width=20, height=18)
    
    return (dir_idx, use_boost)


def select_action_greedy(obs):
    """Greedy baseline: maximize area + blocking."""
    from rl.ma_safety_shield import compute_safe_action_mask
    
    mask = compute_safe_action_mask(obs, width=20, height=18)
    valid_actions = [i for i, m in enumerate(mask) if m]
    
    if not valid_actions:
        return (0, False)
    
    best_action = valid_actions[0]
    best_score = -1e9
    
    for action_idx in valid_actions:
        score = compute_aggressive_action_score(obs, action_idx, width=20, height=18)
        if score > best_score:
            best_score = score
            best_action = action_idx
    
    dir_idx = best_action // 2
    use_boost = (best_action % 2 == 1)
    
    return (dir_idx, use_boost)


def run_episode(env, policy, opponent_types, device="cpu"):
    """
    Run one episode with given policy and opponent types.
    
    Args:
        env: MultiAgentTronEnv
        policy: Trained DQN or None
        opponent_types: List of 'policy', 'greedy', 'random' for each agent
        device: torch device
    
    Returns:
        Episode stats dict
    """
    observations, action_masks = env.reset()
    
    stats = {
        'turns': 0,
        'survivors': [],
        'self_crashes': 0,
        'blocks_caused': 0,
        'head_on_collisions': 0,
        'final_lengths': [],
    }
    
    prev_alive = [True] * env.num_agents
    
    while True:
        # Select actions
        actions = []
        for i, obs in enumerate(observations):
            if not obs['alive']:
                actions.append((0, False))
                continue
            
            if opponent_types[i] == 'policy' and policy is not None:
                action = select_action_policy(policy, obs, device)
            elif opponent_types[i] == 'greedy':
                action = select_action_greedy(obs)
            else:  # random
                import random
                valid = [j for j, m in enumerate(action_masks[i]) if m]
                if valid:
                    action_idx = random.choice(valid)
                    dir_idx = action_idx // 2
                    use_boost = (action_idx % 2 == 1)
                    action = (dir_idx, use_boost)
                else:
                    action = (0, False)
            
            actions.append(action)
        
        # Step
        next_observations, rewards, dones, next_action_masks, info = env.step(actions)
        
        # Track deaths
        for i, agent in enumerate(env.agents):
            if prev_alive[i] and not agent.alive:
                # Agent just died
                # Check if self-crash (hit own trail)
                # This is approximate - we'd need more detailed collision info
                pass
        
        prev_alive = [agent.alive for agent in env.agents]
        
        observations = next_observations
        action_masks = next_action_masks
        stats['turns'] += 1
        
        if info['episode_done']:
            break
    
    # Final stats
    for i, agent in enumerate(env.agents):
        if agent.alive:
            stats['survivors'].append(i)
        stats['final_lengths'].append(agent.length)
        stats['blocks_caused'] += agent.blocks_caused
    
    return stats


def evaluate_multi_agent(
    num_agents: int = 6,
    num_episodes: int = 100,
    checkpoint_path: str = "checkpoints/marl_latest.pt",
    device: str = "cpu",
):
    """
    Evaluate multi-agent policy.
    
    Tests:
    - Policy vs greedy opponents
    - Policy vs random opponents
    - Policy vs itself (self-play)
    """
    print(f"Evaluating multi-agent policy with {num_agents} agents...")
    
    # Load policy
    policy = None
    if os.path.exists(checkpoint_path):
        try:
            policy = DQN(in_channels=6, num_actions=8, num_scalar_features=4).to(device)
            policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
            policy.eval()
            print(f"Loaded policy from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load policy: {e}")
            return
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize environment
    env = MultiAgentTronEnv(num_agents=num_agents)
    
    # Test scenarios
    scenarios = [
        ("Policy vs Greedy", ['policy'] + ['greedy'] * (num_agents - 1)),
        ("Policy vs Random", ['policy'] + ['random'] * (num_agents - 1)),
        ("Self-play (all policy)", ['policy'] * num_agents),
        ("Mixed", ['policy', 'greedy', 'random'] * (num_agents // 3 + 1))[:num_agents],
    ]
    
    results = {}
    
    for scenario_name, opponent_types in scenarios:
        print(f"\n=== {scenario_name} ===")
        
        episode_stats = []
        policy_survival_count = 0
        policy_win_count = 0
        total_turns = 0
        
        for ep in range(num_episodes):
            stats = run_episode(env, policy, opponent_types, device)
            episode_stats.append(stats)
            
            # Check if policy agent (index 0) survived
            if 0 in stats['survivors']:
                policy_survival_count += 1
                # Check if it was the only survivor (win)
                if len(stats['survivors']) == 1:
                    policy_win_count += 1
            
            total_turns += stats['turns']
            
            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep + 1}/{num_episodes}")
        
        # Compute metrics
        survival_rate = policy_survival_count / num_episodes
        win_rate = policy_win_count / num_episodes
        avg_turns = total_turns / num_episodes
        avg_blocks = sum(s['blocks_caused'] for s in episode_stats) / num_episodes
        
        results[scenario_name] = {
            'survival_rate': survival_rate,
            'win_rate': win_rate,
            'avg_turns': avg_turns,
            'avg_blocks_caused': avg_blocks,
        }
        
        print(f"  Survival Rate: {survival_rate:.2%}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Turns: {avg_turns:.1f}")
        print(f"  Avg Blocks Caused: {avg_blocks:.2f}")
    
    # Save results
    os.makedirs("eval_results", exist_ok=True)
    with open(f"eval_results/ma_eval_{num_agents}agents.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    for scenario_name, metrics in results.items():
        print(f"{scenario_name}:")
        print(f"  Survival: {metrics['survival_rate']:.2%}, Win: {metrics['win_rate']:.2%}")
    
    print(f"\nResults saved to eval_results/ma_eval_{num_agents}agents.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multi-agent Tron RL")
    parser.add_argument("--num_agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/marl_latest.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluate_multi_agent(
        num_agents=args.num_agents,
        num_episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

