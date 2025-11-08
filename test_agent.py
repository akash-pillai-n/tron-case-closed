import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Try multi-agent logic first, fallback to original
try:
    from agent_logic_ma import choose_move_ma_aggressive as choose_move
    USE_MA_LOGIC = True
except ImportError:
    from agent_logic import choose_move
    USE_MA_LOGIC = False

try:
    import torch
    from rl.model import DQN
except Exception:
    torch = None
    DQN = None  # type: ignore

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "TestAgent"
AGENT_NAME = "SelfPlayAgent"

MODEL_LOADED = False
MODEL_DEVICE = "cpu"
QNET = None
INFERENCE_CONFIG = {}

def _load_model_once():
    global MODEL_LOADED, QNET, INFERENCE_CONFIG
    if MODEL_LOADED:
        return
    MODEL_LOADED = True
    # Try to load latest checkpoint if torch is available
    if torch is None or DQN is None:
        pass
    import os
    # Load optional inference config
    try:
        import json
        cfg_path = os.path.join(os.getcwd(), "inference_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                INFERENCE_CONFIG = json.load(f)
    except Exception:
        INFERENCE_CONFIG = {}

    if torch is None or DQN is None:
        return
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
        # Handle both dict and direct state_dict formats
        if isinstance(data, dict) and "model_state_dict" in data:
            in_channels = int(data.get("in_channels", 6))
            num_actions = int(data.get("num_actions", 8))
            num_scalar_features = int(data.get("num_scalar_features", 4))
            model = DQN(in_channels=in_channels, num_actions=num_actions, num_scalar_features=num_scalar_features)
            model.load_state_dict(data["model_state_dict"])
        else:
            # Direct state dict - infer from checkpoint name
            if "marl" in os.path.basename(ckpt_path):
                model = DQN(in_channels=6, num_actions=8, num_scalar_features=4)
            else:
                model = DQN(in_channels=4, num_actions=8, num_scalar_features=3)
            model.load_state_dict(data)
        model.eval()
        QNET = model  # type: ignore
        print(f"Test Agent: Loaded model from {os.path.basename(ckpt_path)}")
    except Exception as e:
        print(f"Test Agent: Failed to load model: {e}")
        QNET = None


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint. Returns participant and agent name."""
    return jsonify({
        "participant": PARTICIPANT,
        "agent_name": AGENT_NAME,
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
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    _load_model_once()
    move = choose_move(state, player_number, qnet=QNET, device=MODEL_DEVICE, config=INFERENCE_CONFIG)
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge calls this to notify the agent that the game has ended.
    
    The agent can use this to clean up resources or log final stats.
    """
    data = request.get_json()
    result = data.get("result", "UNKNOWN") if data else "UNKNOWN"
    print(f"Test Agent: Game ended with result: {result}")
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    import sys
    port = 5009  # Default port for player 2 (same as sample_agent)
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 5009")
    
    print(f"Test Agent starting on port {port}...")
    print(f"Using {'multi-agent' if USE_MA_LOGIC else '2-agent'} logic")
    app.run(host="0.0.0.0", port=port, debug=False)

