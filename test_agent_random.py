"""
Test agent with added randomness (epsilon-greedy) to break determinism.
Use this for self-play testing to get more varied results.
"""
import os
import uuid
import random
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Try multi-agent logic first, fallback to original
try:
    from agent_logic_ma import choose_move_ma_aggressive as choose_move_base
    USE_MA_LOGIC = True
except ImportError:
    from agent_logic import choose_move as choose_move_base
    USE_MA_LOGIC = False

try:
    import torch
    from rl.model import DQN
except Exception:
    torch = None
    DQN = None

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "TestAgentRandom"
AGENT_NAME = "SelfPlayAgentWithExploration"

MODEL_LOADED = False
MODEL_DEVICE = "cpu"
QNET = None
INFERENCE_CONFIG = {}

# Exploration parameter
EPSILON = 0.15  # 15% random moves to add variety

def _load_model_once():
    global MODEL_LOADED, QNET, INFERENCE_CONFIG
    if MODEL_LOADED:
        return
    MODEL_LOADED = True
    if torch is None or DQN is None:
        return
    
    import json
    try:
        cfg_path = os.path.join(os.getcwd(), "inference_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                INFERENCE_CONFIG = json.load(f)
    except Exception:
        INFERENCE_CONFIG = {}

    ckpt_candidates = []
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    if os.path.isdir(checkpoints_dir):
        for name in os.listdir(checkpoints_dir):
            if name.endswith(".pt"):
                ckpt_candidates.append(os.path.join(checkpoints_dir, name))
    if not ckpt_candidates:
        return
    
    # Prefer 'marl_latest.pt' for multi-agent, then 'latest.pt'
    ckpt_path = None
    for p in ckpt_candidates:
        basename = os.path.basename(p)
        if basename == "marl_latest.pt":
            ckpt_path = p
            break
    if ckpt_path is None:
        for p in ckpt_candidates:
            if os.path.basename(p) == "latest.pt":
                ckpt_path = p
                break
    if ckpt_path is None:
        ckpt_candidates.sort()
        ckpt_path = ckpt_candidates[-1]
    
    try:
        data = torch.load(ckpt_path, map_location="cpu")
        if isinstance(data, dict) and "model_state_dict" in data:
            in_channels = int(data.get("in_channels", 6))
            num_actions = int(data.get("num_actions", 8))
            num_scalar_features = int(data.get("num_scalar_features", 4))
            model = DQN(in_channels=in_channels, num_actions=num_actions, num_scalar_features=num_scalar_features)
            model.load_state_dict(data["model_state_dict"])
        else:
            if "marl" in os.path.basename(ckpt_path):
                model = DQN(in_channels=6, num_actions=8, num_scalar_features=4)
            else:
                model = DQN(in_channels=4, num_actions=8, num_scalar_features=3)
            model.load_state_dict(data)
        model.eval()
        QNET = model
        print(f"Test Agent (Random): Loaded model from {os.path.basename(ckpt_path)}")
        print(f"Test Agent (Random): Using epsilon={EPSILON} for exploration")
    except Exception as e:
        print(f"Test Agent (Random): Failed to load model: {e}")
        QNET = None


def choose_move_with_exploration(state, player_number, qnet, device, config):
    """Choose move with epsilon-greedy exploration."""
    # With probability epsilon, choose a random valid move
    if random.random() < EPSILON:
        # Get valid actions from state
        board = state.get("board", [])
        if board:
            # Simple random move (UP, DOWN, LEFT, RIGHT)
            directions = ["UP", "DOWN", "LEFT", "RIGHT"]
            return random.choice(directions)
    
    # Otherwise use the trained policy
    return choose_move_base(state, player_number, qnet=qnet, device=device, config=config)


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "participant": PARTICIPANT,
        "agent_name": AGENT_NAME,
        "epsilon": EPSILON,
        "status": "ready"
    }), 200


def _update_local_game_from_post(data):
    """Update local game state from judge's POST data."""
    with game_lock:
        global LAST_POSTED_STATE
        LAST_POSTED_STATE = data
        if "board" in data:
            GLOBAL_GAME.board.grid = data["board"]
        if "agent1_trail" in data:
            trail = data["agent1_trail"]
            GLOBAL_GAME.agent1.trail = deque(trail, maxlen=len(trail))
        if "agent2_trail" in data:
            trail = data["agent2_trail"]
            GLOBAL_GAME.agent2.trail = deque(trail, maxlen=len(trail))
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Receive game state from judge."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Return agent's move with epsilon-greedy exploration."""
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
   
    _load_model_once()
    move = choose_move_with_exploration(state, player_number, qnet=QNET, device=MODEL_DEVICE, config=INFERENCE_CONFIG)

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Game ended notification."""
    data = request.get_json()
    result = data.get("result", "UNKNOWN") if data else "UNKNOWN"
    print(f"Test Agent (Random): Game ended with result: {result}")
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    import sys
    port = 5009
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 5009")
    
    print(f"Test Agent (Random) starting on port {port}...")
    print(f"Using {'multi-agent' if USE_MA_LOGIC else '2-agent'} logic with epsilon={EPSILON}")
    app.run(host="0.0.0.0", port=port, debug=False)

