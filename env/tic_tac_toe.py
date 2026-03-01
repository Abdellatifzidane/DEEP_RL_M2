from __future__ import annotations
from dataclasses import dataclass, field
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


@dataclass
class TicTacToeVsRandom:
    """
    Interactive environment:
      - agent always plays X (+1)
      - opponent plays O (-1) with random move
    """

    seed: Optional[int] = None
    board: List[int] = field(default_factory=lambda: [0] * 9)

    current_state: int = 0  # encoded integer state (optional convenience)
    done: bool = False
    last_reward: float = 0.0
    score: float = 0.0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self._sync_state()

    def reset(self) -> None:
        self.board = [0] * 9
        self.done = False
        self.last_reward = 0.0
        self.score = 0.0
        self._sync_state()

    def step(self, action: int) -> None:
        """
        action: int in [0..8]
        if agent plays invalid action => immediate loss (-1) and terminal
        sinon: agent plays, check end; if not end, random opponent plays, check end
        """
        if self.done:
            self.last_reward = 0.0
            return

        #validate action
        if action < 0 or action > 8 or self.board[action] != 0:
            self.last_reward = -1.0
            self.score += self.last_reward
            self.done = True
            self._sync_state()
            return

        #agent move (X= +1)
        self.board[action] = +1

        w = check_winner(self.board)
        if w == +1:
            self.last_reward = +1.0
            self.score += self.last_reward
            self.done = True
            self._sync_state()
            return

        if self._is_draw():
            self.last_reward = 0.0
            self.score += self.last_reward
            self.done = True
            self._sync_state()
            return

        #opponent random move (O=-1)
        opp_action = self._random_legal_action()
        if opp_action is not None:
            self.board[opp_action] = -1

        w = check_winner(self.board)
        if w == -1:
            self.last_reward = -1.0
            self.score += self.last_reward
            self.done = True
            self._sync_state()
            return

        if self._is_draw():
            self.last_reward = 0.0
            self.score += self.last_reward
            self.done = True
            self._sync_state()
            return

        #continues
        self.last_reward = 0.0
        self.score += self.last_reward
        self._sync_state()

    def get_state(self) -> int:
        """returns an integer encoding of the board (base-3)"""
        return self.current_state

    def get_board(self) -> List[int]:
        """returns raw board list of length 9."""
        return list(self.board)

    def get_available_actions(self) -> List[int]:
        return [] if self.done else [i for i, v in enumerate(self.board) if v == 0]

    def is_terminal(self) -> bool:
        return self.done

    def get_score(self) -> float:
        return self.score

    def get_last_reward(self) -> float:
        return self.last_reward

    def get_num_states(self) -> int:
        return 3 ** 9

    def _is_draw(self) -> bool:
        return all(v != 0 for v in self.board) and check_winner(self.board) == 0

    def _random_legal_action(self) -> Optional[int]:
        legal = [i for i, v in enumerate(self.board) if v == 0]
        if not legal:
            return None
        return self.rng.choice(legal)

    def _sync_state(self) -> None:
        """
        Encode board into base-3 integer:
          empty=0 -> digit 0
          X=+1    -> digit 1
          O=-1    -> digit 2
        """
        mapping = {0: 0, +1: 1, -1: 2}
        s = 0
        power = 1
        for cell in self.board:
            s += mapping[cell] * power
            power *= 3
        self.current_state = s

    def render(self) -> None:
        symbols = {0: " ", +1: "X", -1: "O"}
        rows = []
        for r in range(3):
            rows.append(" | ".join(symbols[self.board[3*r + c]] for c in range(3)))
        print("\n---------\n".join(rows))


if __name__ == "__main__":
    env = TicTacToeVsRandom(seed=0)
    env.reset()

    while not env.is_terminal():
        a = random.choice(env.get_available_actions())
        env.step(a)

    env.render()
    print("done:", env.is_terminal(), "last_reward:", env.get_last_reward(), "score:", env.get_score())
