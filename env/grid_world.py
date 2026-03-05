from __future__ import annotations
from typing import List, Tuple
import math
from dataclasses import dataclass, field
from typing import List, Protocol
import math


# Utils

def compute_grid_size(num_states: int) -> int:
    """Equivalent of compute_grid_size(states.len()) in Rust."""
    g = int(math.isqrt(num_states))
    if g * g != num_states:
        raise ValueError(f"num_states must be a perfect square, got {num_states}")
    return g


def pos_to_index(row: int, col: int, grid_size: int) -> int:
    """Convert (row, col) -> unique state index."""
    return row * grid_size + col


# Interfaces

class RLStateEnv(Protocol):
    def get_states(self) -> List[int]: ...
    def get_actions(self) -> List[int]: ...
    def get_rewards(self) -> List[float]: ...
    def get_transitions(self) -> List[List[List[List[float]]]]: ...
    def get_terminal_states(self) -> List[int]: ...


class RLInteractiveEnv(Protocol):
    def reset(self) -> None: ...
    def step(self, action: int) -> None: ...
    def get_state(self) -> int: ...
    def get_available_actions(self) -> List[int]: ...
    def is_terminal(self) -> bool: ...
    def get_score(self) -> float: ...
    def get_num_states(self) -> int: ...


# =========================
# GRIDWORLD (Model-Based)
# =========================

@dataclass
class GridWorld(RLStateEnv):
    states: List[int]
    actions: List[int]
    rewards: List[float]
    terminal_states: List[int]
    transitions: List[List[List[List[float]]]]  # [s][a][s’][r]

    current_state: int
    score: float = 0.0
    done: bool = False
    last_reward: float = 0.0

    # RLStateEnv methods
    def get_states(self) -> List[int]:
        return self.states

    def get_actions(self) -> List[int]:
        return self.actions

    def get_rewards(self) -> List[float]:
        return self.rewards

    def get_transitions(self) -> List[List[List[List[float]]]]:
        return self.transitions

    def get_terminal_states(self) -> List[int]:
        return self.terminal_states

def value_iteration(env, gamma: float = 0.99, theta: float = 1e-10) -> Tuple[List[float], List[int]]:
    states = env.get_states()
    actions = env.get_actions()
    rewards = env.get_rewards()
    T = env.get_transitions()
    terminal = set(env.get_terminal_states())

    V = [0.0 for _ in states]
    pi = [0 for _ in states]

    while True:
        delta = 0.0
        for s in states:
            if s in terminal:
                continue

            best_q = -1e18
            best_a = 0

            for a in actions:
                q = 0.0
                # q(s,a)= sum_{s',r} P(s',r|s,a) * (r + gamma V[s'])
                for sp in states:
                    for r_idx, r_val in enumerate(rewards):
                        p = T[s][a][sp][r_idx]
                        if p:
                            q += p * (r_val + gamma * V[sp])

                if q > best_q:
                    best_q = q
                    best_a = a

            delta = max(delta, abs(best_q - V[s]))
            V[s] = best_q
            pi[s] = best_a

        if delta < theta:
            break

    return V, pi

def initialize_gridworld() -> GridWorld:
    grid_size = 5
    num_states = grid_size * grid_size

    states = list(range(num_states))
    actions = [0, 1, 2, 3] #0=left, 1=right, 2=up, 3=down
    rewards = [0.0, -3.0, 1.0] #neutral, negative, positive

    neg_reward_state = pos_to_index(0, grid_size - 1, grid_size) # (0,4) -> 4
    pos_reward_state = pos_to_index(grid_size - 1, grid_size - 1, grid_size) # (4,4) -> 24
    terminal_states = [neg_reward_state, pos_reward_state]

    # transitions[s][a][s_prime][reward_idx]
    transitions = [
        [
            [
                [0.0 for _ in range(len(rewards))]
                for _ in range(num_states)
            ]
            for _ in range(len(actions))
        ]
        for _ in range(num_states)
    ]

    for row in range(grid_size):
        for col in range(grid_size):
            s = pos_to_index(row, col, grid_size)

            if s in terminal_states:
                continue

            #action 0: left, 1: right, 2: up, 3: down
            for a in actions:
                if a == 0: # left
                    new_row, new_col = row, col - 1 if col > 0 else col
                elif a == 1: # right
                    new_row, new_col = row, col + 1 if col < grid_size - 1 else col
                elif a == 2: # up
                    new_row, new_col = row - 1 if row > 0 else row, col
                elif a == 3: # down
                    new_row, new_col = row + 1 if row < grid_size - 1 else row, col
                else:
                    new_row, new_col = row, col

                s_prime = pos_to_index(new_row, new_col, grid_size)
                destination = s if (new_row == row and new_col == col) else s_prime

                if destination == pos_reward_state:
                    reward_idx = 2
                elif destination == neg_reward_state:
                    reward_idx = 1
                else:
                    reward_idx = 0

                transitions[s][a][destination][reward_idx] = 1.0

    return GridWorld(
        states=states,
        actions=actions,
        rewards=rewards,
        terminal_states=terminal_states,
        transitions=transitions,
        current_state=pos_to_index(2, 2, grid_size),
        score=0.0,
        done=False,
        last_reward=0.0,
    )


def display_gridworld_mb(v: List[float], pi: List[int], env: RLStateEnv) -> None:
    grid_size = compute_grid_size(len(env.get_states()))

    print(" Valeurs optimales V* :")
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            print(f"{v[idx]:.5f} ", end="")
        print()

    print("\n Politique optimale π* (0=←, 1=→, 2=↑, 3=↓) :")
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            if idx in env.get_terminal_states():
                print("  XX  ", end="")
            else:
                symbol = {0: "←", 1: "→", 2: "↑", 3: "↓"}.get(pi[idx], "?")
                print(f"  {symbol:^2}  ", end="")
        print()

    print("\n=========================================")


# =========================
# GRIDWORLD (Interactive / Model-Free)
# =========================

@dataclass
class GridWorldInteractive(RLInteractiveEnv):
    grid_size: int = 5
    terminal_states: List[int] = field(default_factory=list)
    pos_reward_state: int = 0
    neg_reward_state: int = 0
    current_state: int = 0
    score: float = 0.0
    last_reward: float = 0.0
    done: bool = False

    def __post_init__(self) -> None:
        self.pos_reward_state = pos_to_index(self.grid_size - 1, self.grid_size - 1, self.grid_size)
        self.neg_reward_state = pos_to_index(0, self.grid_size - 1, self.grid_size)
        self.terminal_states = [self.neg_reward_state, self.pos_reward_state]
        self.current_state = pos_to_index(0, 0, self.grid_size)

    @staticmethod
    def pos_to_index(row: int, col: int, grid_size: int) -> int:
        return row * grid_size + col

    def move_agent(self, state: int, action: int) -> int:
        row = state // self.grid_size
        col = state % self.grid_size

        if action == 0 and col > 0:              # left
            new_row, new_col = row, col - 1
        elif action == 1 and col < self.grid_size - 1:  # right
            new_row, new_col = row, col + 1
        elif action == 2 and row > 0:            # up
            new_row, new_col = row - 1, col
        elif action == 3 and row < self.grid_size - 1:  # down
            new_row, new_col = row + 1, col
        else:
            new_row, new_col = row, col

        return self.pos_to_index(new_row, new_col, self.grid_size)

    # RLInteractiveEnv methods
    def reset(self) -> None:
        self.current_state = self.pos_to_index(0, 0, self.grid_size)
        self.score = 0.0
        self.last_reward = 0.0
        self.done = False

    def step(self, action: int) -> None:
        if self.done:
            self.last_reward = 0.0
            return

        next_state = self.move_agent(self.current_state, action)

        if next_state == self.pos_reward_state:
            self.last_reward = 1.0
        elif next_state == self.neg_reward_state:
            self.last_reward = -3.0
        else:
            self.last_reward = 0.0

        self.current_state = next_state
        self.score += self.last_reward
        self.done = self.current_state in (self.pos_reward_state, self.neg_reward_state)

    def get_state(self) -> int:
        return self.current_state

    def get_available_actions(self) -> List[int]:
        return [] if self.done else [0, 1, 2, 3]

    def is_terminal(self) -> bool:
        return self.done

    def get_score(self) -> float:
        return self.score

    def get_num_states(self) -> int:
        return self.grid_size * self.grid_size


def display_interactive_gridworld(v: List[float], pi: List[int], env: RLInteractiveEnv) -> None:
    grid_size = 5
    pos_reward_state = grid_size * grid_size - 1 # 24
    neg_reward_state = grid_size - 1 # 4

    print("=== Résultat pour l'environnement : Gridworld ===")

    for row in range(grid_size):
        for col in range(grid_size):
            s = row * grid_size + col
            val = v[s]
            action = pi[s]

            if s == pos_reward_state:
                symbol = "V"
            elif s == neg_reward_state:
                symbol = "X"
            elif env.is_terminal() and env.get_state() == s:
                symbol = "T"
            else:
                symbol = {0: "←", 1: "→", 2: "↑", 3: "↓"}.get(action, " ")

            print(f"[{val:>5.2f} {symbol}] ", end="")
        print()

    print("\n--- Policy  ---")
    for row in range(grid_size):
        for col in range(grid_size):
            s = row * grid_size + col
            action = pi[s]

            if s == pos_reward_state:
                symbol = "V"
            elif s == neg_reward_state:
                symbol = "X"
            else:
                symbol = {0: "←", 1: "→", 2: "↑", 3: "↓"}.get(action, " ")

            print(f" {symbol} ", end="")
        print()

    print("===========================================")


# Mini demo
# if __name__ == "__main__":
#     env_mb = initialize_gridworld()
#     # Dummy v/pi
#     v = [0.0] * (env_mb.states[-1] + 1)
#     pi = [1] * (env_mb.states[-1] + 1)
#     display_gridworld_mb(v, pi, env_mb)

#     env = GridWorldInteractive()
#     env.reset()
#     env.step(1) #right
#     print("state:", env.get_state(), "score:", env.get_score(), "done:", env.is_terminal())

if __name__ == "__main__":
    env_mb = initialize_gridworld()
    V, pi = value_iteration(env_mb, gamma=0.95)
    display_gridworld_mb(V, pi, env_mb)
