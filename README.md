# Case Closed Agent Template

### Explanation of Files

This template provides a few key files to get you started. Here's what each one does:

#### `agent.py`
**This is the most important file. This is your starter code, where you will write your agent's logic.**

*   DO NOT RENAME THIS FILE! Our pipeline will only recognize your agent as `agent.py`.
*   It contains a fully functional, Flask-based web server that is already compatible with the Judge Engine's API.
*   It has all the required endpoints (`/`, `/send-state`, `/send-move`, `/end`). You do not need to change the structure of these.
*   Look for the `send_move` function. Inside, you will find a section marked with comments: `# --- YOUR CODE GOES HERE ---`. This is where you should add your code to decide which move to make based on the current game state.
*   Your agent can return moves in the format `"DIRECTION"` (e.g., `"UP"`, `"DOWN"`, `"LEFT"`, `"RIGHT"`) or `"DIRECTION:BOOST"` (e.g., `"UP:BOOST"`) to use a speed boost.

#### `requirements.txt`
**This file lists your agent's Python dependencies.**

*   Don't rename this file either.
*   It comes pre-populated with `Flask` and `requests`.
*   If your agent's logic requires other libraries (like `numpy`, `scipy`, or any other package from PyPI), you **must** add them to this file.
*   When you submit, our build pipeline will run `pip install -r requirements.txt` to install these libraries for your agent.

#### `judge_engine.py`
**A copy of the runner of matches.**

*   The judge engine is the heart of a match in Case Closed. It can be used to simulate a match.
*   The judge engine can be run only when two agents are running on ports `5008` and `5009`.
*   We provide a sample agent that can be used to train your agent and evaluate its performance.

#### `case_closed_game.py`
**A copy of the official game state logic.**

*   Don't rename this file either.
*   This file contains the complete state of the match played, including the `Game`, `GameBoard`, and `Agent` classes.
*   While your agent will receive the game state as a JSON object, you can read this file to understand the exact mechanics of the game: how collisions are detected, how trails work, how boosts function, and what ends a match. This is the "source of truth" for the game rules.
*   Key mechanics:
    - Agents leave permanent trails behind them
    - Hitting any trail (including your own) causes death
    - Head-on collisions: the agent with the longer trail survives
    - Each agent has 3 speed boosts (moves twice instead of once)
    - The board has torus (wraparound) topology
    - Game ends after 500 turns or when one/both agents die

#### `sample_agent.py`
**A simple agent that you can play against.**

*   The sample agent is provided to help you evaluate your own agent's performance. 
*   In conjunction with `judge_engine.py`, you should be able to simulate a match against this agent.

#### `local-tester.py`
**A local tester to verify your agent's API compliance.**

*   This script tests whether your agent correctly implements all required endpoints.
*   Run this to ensure your agent can communicate with the judge engine before submitting.

#### `Dockerfile`
**A copy of the Dockerfile your agent will be containerized with.**

*   This is a copy of a Dockerfile. This same Dockerfile will be used to containerize your agent so we can run it on our evaluation platform.
*   It is **HIGHLY** recommended that you try Dockerizing your agent once you're done. We can't run your agent if it can't be containerized.
*   There are a lot of resources at your disposal to help you with this. We recommend you recruit a teammate that doesn't run Windows for this. 

#### `.dockerignore`
**A .dockerignore file doesn't include its contents into the Docker image**

*   This `.dockerignore` file will be useful for ensuring unwanted files do not get bundled in your Docker image.
*   You have a 5GB image size restriction, so you are given this file to help reduce image size and avoid unnecessary files in the image.

#### `.gitignore`
*   A standard configuration file that tells Git which files and folders (like the `venv` virtual environment directory) to ignore. You shouldn't need to change this.


### Testing your agent:
**Both `agent.py` and `sample_agent.py` come ready to run out of the box!**

*   To test your agent, you will likely need to create a `venv`. Look up how to do this. 
*   Next, you'll need to `pip install` any required libraries. `Flask` is one of these.
*   Finally, in separate terminals, run both `agent.py` and `sample_agent.py`, and only then can you run `judge_engine.py`.
*   You can also run `local-tester.py` to verify your agent's API compliance before testing against another agent.


### Disclaimers:
* There is a 5GB limit on Docker image size, to keep competition fair and timely.
* Due to platform and build-time constraints, participants are limited to **CPU-only PyTorch**; GPU-enabled versions, including CUDA builds, are disallowed. Any other heavy-duty GPU or large ML frameworks (like Tensorflow, JAX) will not be allowed.
* Ensure your agent's `requirements.txt` is complete before pushing changes.
* If you run into any issues, take a look at your own agent first before asking for help.


### RL Training and Evaluation (optional, local-only)
We provide a minimal RL scaffold that does NOT add PyTorch to `requirements.txt`. Train and evaluate locally with a CPU-only PyTorch you install on your machine. The runtime agent will fall back to a heuristic policy if PyTorch or a checkpoint is unavailable.

#### Standard 2-Agent Training
Files:
* `rl/env_wrapper.py`: Training environment honoring README rules (max 500 turns, head-on = draw).
* `rl/model.py`: Tiny DQN for discrete actions (UP/DOWN/LEFT/RIGHT with optional BOOST).
* `rl/replay_buffer.py`: Simple experience replay.
* `train.py`: Self-play loop; saves checkpoints under `checkpoints/`.
* `eval.py`: Quick evaluation against a simple scripted opponent.
* `inference_config.json`: Optional knobs for inference behavior (e.g., conservative boost).

Usage:
1. Install CPU-only PyTorch locally (do not add to requirements.txt).
2. Train:
   - `python train.py`
   - Checkpoints are stored under `checkpoints/latest.pt`.
3. Evaluate:
   - `python eval.py`
4. Run a match via judge (separate terminals):
   - `python agent.py` (port 5008 by default)
   - `python sample_agent.py` (port 5009 by default)
   - `python judge_engine.py`

#### Multi-Agent Training (6+ agents)
For advanced training with 6+ simultaneous agents using safety shields and blocking strategies:

Files:
* `rl/ma_env.py`: Multi-agent environment with N agents, simultaneous moves, 500-turn cap.
* `rl/ma_encoder.py`: Observation encoder with occupancy, heads, danger maps, Voronoi partitioning.
* `rl/ma_safety_shield.py`: Action masking, 2-step collision checks, deadlock avoidance, flood-fill area scoring.
* `rl/ma_blocking.py`: Voronoi territory analysis, choke detection, aggressive blocking strategies.
* `train_ma.py`: Multi-agent self-play with opponent pool (random, greedy, frozen policy).
* `eval_ma.py`: Evaluation with survival rate, win rate, blocks-caused, head-on metrics.
* `agent_logic_ma.py`: Multi-agent inference logic with safety shield and aggressive blocking.

Usage:
1. Install CPU-only PyTorch locally.
2. Train multi-agent:
   ```bash
   python train_ma.py --num_agents 6 --episodes 5000
   ```
   - Checkpoints saved to `checkpoints/marl_latest.pt` and `checkpoints/marl_config.json`.
3. Evaluate multi-agent:
   ```bash
   python eval_ma.py --num_agents 6 --episodes 100
   ```
   - Results saved to `eval_results/ma_eval_6agents.json`.
4. Run match (agent.py automatically uses multi-agent logic if available):
   - `python agent.py` (loads `marl_latest.pt` if present, else `latest.pt`, else heuristic)

Multi-Agent Features:
* **Safety Shield**: Never crashes into walls or trails; avoids deadlocks via flood-fill area checks.
* **Blocking Strategy**: Uses Voronoi partitioning to identify opponent territories and prioritizes moves that reduce opponent space.
* **Boost Intelligence**: Only uses boosts when they increase reachable area or enable escapes.
* **Deadlock Avoidance**: Detects narrow corridors and low-area traps; prefers actions that maximize future options.
* **Backward Compatible**: Works with standard 2-agent judge payloads.

Runtime behavior:
* On startup of `/send-move`, the agent attempts to load `checkpoints/marl_latest.pt` (multi-agent), then `checkpoints/latest.pt` (2-agent), and `inference_config.json`. If unavailable or PyTorch is not present, it uses a safe heuristic with flood-fill area scoring and conservative BOOST usage.
