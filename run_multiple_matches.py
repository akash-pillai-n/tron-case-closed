#!/usr/bin/env python3
"""
Run multiple self-play matches and track statistics.
Requires agent.py and test_agent.py to be running.
"""

import subprocess
import time
import requests
import json
from collections import defaultdict

def check_agents_running():
    """Check if both agents are running."""
    try:
        r1 = requests.get("http://localhost:5008/", timeout=2)
        r2 = requests.get("http://localhost:5009/", timeout=2)
        return r1.status_code == 200 and r2.status_code == 200
    except:
        return False

def run_match():
    """Run a single match via judge_engine.py and capture result."""
    try:
        result = subprocess.run(
            ["python", "judge_engine.py"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per match
        )
        
        output = result.stdout
        
        # Parse result from output (check actual judge_engine.py output format)
        if "Winner: Agent 1" in output:
            return "AGENT1_WIN"
        elif "Winner: Agent 2" in output:
            return "AGENT2_WIN"
        elif "Game ended in a draw" in output or "DRAW" in output:
            return "DRAW"
        else:
            # Debug: print last few lines if unknown
            lines = output.strip().split('\n')
            if lines:
                print(f"(Last line: {lines[-1][:80]})")
            return "UNKNOWN"
    except subprocess.TimeoutExpired:
        print("  Match timed out!")
        return "TIMEOUT"
    except Exception as e:
        print(f"  Error running match: {e}")
        return "ERROR"

def main():
    print("=== Self-Play Match Runner ===")
    print()
    
    # Check if agents are running
    print("Checking if agents are running...")
    if not check_agents_running():
        print()
        print("Error: Agents not running!")
        print()
        print("Please start both agents first:")
        print("  Terminal 1: python agent.py")
        print("  Terminal 2: python test_agent.py")
        print()
        return
    
    print("✓ Both agents are running")
    print()
    
    # Get number of matches
    try:
        num_matches = int(input("How many matches to run? (default: 10): ") or "10")
    except ValueError:
        num_matches = 10
    
    print()
    print(f"Running {num_matches} self-play matches...")
    print()
    
    results = defaultdict(int)
    
    for i in range(num_matches):
        print(f"Match {i+1}/{num_matches}...", end=" ", flush=True)
        result = run_match()
        results[result] += 1
        print(f"{result}")
        
        # Small delay between matches
        time.sleep(0.5)
    
    print()
    print("=== Results ===")
    print(f"Agent 1 (port 5008) wins: {results['AGENT1_WIN']}")
    print(f"Agent 2 (port 5009) wins: {results['AGENT2_WIN']}")
    print(f"Draws: {results['DRAW']}")
    
    if results['TIMEOUT'] > 0:
        print(f"Timeouts: {results['TIMEOUT']}")
    if results['ERROR'] > 0:
        print(f"Errors: {results['ERROR']}")
    if results['UNKNOWN'] > 0:
        print(f"Unknown: {results['UNKNOWN']}")
    
    print()
    
    # Calculate win rate
    total_valid = results['AGENT1_WIN'] + results['AGENT2_WIN'] + results['DRAW']
    if total_valid > 0:
        agent1_winrate = (results['AGENT1_WIN'] / total_valid) * 100
        agent2_winrate = (results['AGENT2_WIN'] / total_valid) * 100
        draw_rate = (results['DRAW'] / total_valid) * 100
        
        print(f"Agent 1 win rate: {agent1_winrate:.1f}%")
        print(f"Agent 2 win rate: {agent2_winrate:.1f}%")
        print(f"Draw rate: {draw_rate:.1f}%")
        print()
        
        if abs(agent1_winrate - agent2_winrate) < 10:
            print("✓ Balanced performance (similar win rates)")
        else:
            print("⚠ Unbalanced performance (check for bugs or randomness)")
    
    # Save results
    results_file = "selfplay_results.json"
    with open(results_file, 'w') as f:
        json.dump(dict(results), f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()

