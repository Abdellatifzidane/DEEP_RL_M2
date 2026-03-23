from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from environnements.base_env import BaseEnv
from typing import List, Optional, Tuple
import random

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8), # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8), # cols
    (0, 4, 8), (2, 4, 6) # diagonals
)

def check_winner(board: List[int]) -> int:
    """
    returns:
      +1 if X wins
      -1 if O wins
       0 otherwise
    board values: +1 (X), -1 (O), 0 empty
    """
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return +1
        if s == -3:
            return -1
    return 0


class TicTacToe(BaseEnv):
    """
    TicTacToe environment:
    - Agent plays X (+1), opponent plays O (-1) randomly
    - State: tuple of 9 integers (-1, 0, 1)
    - Actions: 0-8 (board positions)
    - Reward: +1 for win, -1 for loss, 0 for draw or ongoing
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = None
        self.done = False
        self.score = 0.0

    def reset(self):
        self.board = [0] * 9
        self.done = False
        self.score = 0.0
        return self.get_state()

    def get_state(self):
        # Return tuple for hashability
        return tuple(self.board)

    def get_available_actions(self):
        if self.done:
            return []
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0

        # Validate action
        if action < 0 or action > 8 or self.board[action] != 0:
            # Invalid move: immediate loss
            self.done = True
            self.score = -1.0
            return self.get_state(), -1.0

        # Agent move (X = +1)
        self.board[action] = 1

        # Check if agent wins
        if check_winner(self.board) == 1:
            self.done = True
            self.score = 1.0
            return self.get_state(), 1.0

        # Check for draw
        if self._is_draw():
            self.done = True
            self.score = 0.0
            return self.get_state(), 0.0

        # Opponent random move (O = -1)
        opp_action = self._random_legal_action()
        if opp_action is not None:
            self.board[opp_action] = -1

        # Check if opponent wins
        if check_winner(self.board) == -1:
            self.done = True
            self.score = -1.0
            return self.get_state(), -1.0

        # Check for draw after opponent move
        if self._is_draw():
            self.done = True
            self.score = 0.0
            return self.get_state(), 0.0

        # Game continues
        return self.get_state(), 0.0

    def is_terminal(self):
        return self.done

    def get_score(self):
        return self.score

    def _is_draw(self):
        return all(v != 0 for v in self.board) and check_winner(self.board) == 0

    def _random_legal_action(self):
        legal = [i for i, v in enumerate(self.board) if v == 0]
        if not legal:
            return None
        return self.rng.choice(legal)

    def render(self):
        symbols = {0: " ", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            rows.append(" | ".join(symbols[self.board[3*r + c]] for c in range(3)))
        print("\n---------\n".join(rows))


# Mini demo
if __name__ == "__main__":
    env = TicTacToe(seed=0)
    state = env.reset()
    print("Initial state:", state)
    print("Available actions:", env.get_available_actions())

    # Play a few moves
    actions = [0, 4, 8]  # Example moves
    for action in actions:
        if not env.is_terminal():
            next_state, reward = env.step(action)
            print(f"Action {action}: state={next_state}, reward={reward}, terminal={env.is_terminal()}")
            env.render()
        else:
            break

    print("Final score:", env.get_score())
