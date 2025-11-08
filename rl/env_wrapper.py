from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Import game primitives from the provided implementation
from case_closed_game import Game, Agent, Direction, GameResult


@dataclass
class EnvStepResult:
    next_state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class CaseClosedEnv:
    """
    A lightweight training environment that wraps the in-repo Case Closed game logic.

    Differences vs case_closed_game.Game.step():
      - Does NOT enforce the built-in 200-turn cap; instead uses max_turns (default 500)
      - Treats head-on collisions as draw (which the provided Agent.move + simultaneous step implies)
      - Keeps its own turn counter separate from Game.turns
      - Allows explicit boost flags per agent
    """

    def __init__(self, max_turns: int = 500):
        self.max_turns = max_turns
        self.game: Game = Game()
        self.turn_count: int = 0

    def reset(self) -> Dict[str, Any]:
        self.game.reset()
        self.turn_count = 0
        return self._export_state()

    def _export_state(self) -> Dict[str, Any]:
        """Export a judge-like state dictionary for training."""
        return {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.turn_count,
        }

    @staticmethod
    def str_to_direction(direction_str: str) -> Direction:
        mapping = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT,
        }
        return mapping[direction_str]

    def step(
        self,
        p1_direction: Direction,
        p2_direction: Direction,
        p1_boost: bool = False,
        p2_boost: bool = False,
    ) -> EnvStepResult:
        """
        Execute one simultaneous move for both agents, mirroring judge timing.
        Returns EnvStepResult with reward shaped from terminal outcome:
          - +1 win, -1 loss, 0 draw/ongoing by default (can be customized by trainer).
        """
        # Perform moves (simultaneous in spirit; both use same pre-state)
        agent1_alive = self.game.agent1.move(p1_direction, other_agent=self.game.agent2, use_boost=p1_boost)
        agent2_alive = self.game.agent2.move(p2_direction, other_agent=self.game.agent1, use_boost=p2_boost)

        self.turn_count += 1

        result: Optional[GameResult] = None
        if not agent1_alive and not agent2_alive:
            result = GameResult.DRAW
        elif not agent1_alive:
            result = GameResult.AGENT2_WIN
        elif not agent2_alive:
            result = GameResult.AGENT1_WIN
        elif self.turn_count >= self.max_turns:
            # On max turns, determine result by trail length (consistent with README language;
            # you may re-tune reward shaping externally if desired)
            if self.game.agent1.length > self.game.agent2.length:
                result = GameResult.AGENT1_WIN
            elif self.game.agent2.length > self.game.agent1.length:
                result = GameResult.AGENT2_WIN
            else:
                result = GameResult.DRAW

        done = result is not None
        reward = 0.0
        if done:
            if result == GameResult.AGENT1_WIN:
                reward = 1.0
            elif result == GameResult.AGENT2_WIN:
                reward = -1.0
            else:
                reward = 0.0

        return EnvStepResult(
            next_state=self._export_state(),
            reward=reward,
            done=done,
            info={
                "result": None if result is None else result.name,
                "turn_count": self.turn_count,
            },
        )


