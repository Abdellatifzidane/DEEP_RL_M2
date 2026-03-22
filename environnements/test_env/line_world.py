from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol
from typing import List, Tuple
import math

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
# LINEWORLD (Model-Based)
# =========================

@dataclass
class LineWorld(RLStateEnv):
    states: List[int]                          # ex: [0,1,2,3,4]
    actions: List[int]                         # ex: [0(left), 1(right)]
    rewards: List[float]                       # ex: [-1, 0, +1]
    terminal_states: List[int]                 # ex: [0,4]
    transitions: List[List[List[List[float]]]] # [s][a][s’][r_idx] -> prob

    current_state: int = 1
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

def initialize_lineworld() -> LineWorld:
    states = [0, 1, 2, 3, 4]
    actions = [0, 1]                 # 0=left, 1=right
    rewards = [-1.0, 0.0, 1.0]       # reward indices: 0 -> -1, 1 -> 0, 2 -> +1
    terminal_states = [0, 4]

    # transitions[s][a][s_prime][reward_idx]
    transitions = [
        [
            [
                [0.0 for _ in range(len(rewards))]
                for _ in range(len(states))
            ]
            for _ in range(len(actions))
        ]
        for _ in range(len(states))
    ]

    # left (a=0)
    transitions[3][0][2][1] = 1.0  # 3 --left--> 2 with reward 0.0 (idx 1)
    transitions[2][0][1][1] = 1.0  # 2 --left--> 1 with reward 0.0
    transitions[1][0][0][0] = 1.0  # 1 --left--> 0 with reward -1.0 (idx 0)

    # right (a=1)
    transitions[1][1][2][1] = 1.0  # 1 --right--> 2 with reward 0.0
    transitions[2][1][3][1] = 1.0  # 2 --right--> 3 with reward 0.0
    transitions[3][1][4][2] = 1.0  # 3 --right--> 4 with reward +1.0 (idx 2)

    return LineWorld(
        states=states,
        actions=actions,
        rewards=rewards,
        terminal_states=terminal_states,
        transitions=transitions,
        current_state=1,
        score=0.0,
        done=False,
        last_reward=0.0,
    )


def display_lineworld_mb(v: List[float], pi: List[int], env: RLStateEnv) -> None:
    print("=== Résultat pour l'environnement : LineWorld MB ===")

    print(" Valeurs optimales V* :")
    for value in v:
        print(f"{value:.2f}  ", end="")
    print()

    print("\n Politique optimale π* (0=←, 1=→) :")
    terminals = set(env.get_terminal_states())
    for idx, action in enumerate(pi):
        if idx in terminals:
            print("XX  ", end="")
        else:
            symbol = "←" if action == 0 else ("→" if action == 1 else "?")
            print(f"{symbol}   ", end="")
    print("\n=========================================")


# =========================
# LINEWORLD (Interactive)
# =========================

@dataclass
class LineWorldInteractive(RLInteractiveEnv):
    current_state: int = 1
    score: float = 0.0
    last_reward: float = 0.0
    done: bool = False

    def reset(self) -> None:
        self.current_state = 1
        self.score = 0.0
        self.last_reward = 0.0
        self.done = False

    def step(self, action: int) -> None:
        if self.done:
            self.last_reward = 0.0
            return

        if action == 0:  # left
            next_state = max(0, self.current_state - 1)
        elif action == 1:  # right
            next_state = min(4, self.current_state + 1)
        else:
            next_state = self.current_state

        if next_state == 0:
            self.last_reward = -1.0
        elif next_state == 4:
            self.last_reward = 1.0
        else:
            self.last_reward = 0.0

        self.score += self.last_reward
        self.current_state = next_state
        self.done = self.current_state in (0, 4)

    def get_state(self) -> int:
        return self.current_state

    def get_available_actions(self) -> List[int]:
        return [] if self.done else [0, 1]

    def is_terminal(self) -> bool:
        return self.done

    def get_score(self) -> float:
        return self.score

    def get_num_states(self) -> int:
        return 5  # states 0..4


# Mini demo
if __name__ == "__main__":
    env_lw = initialize_lineworld()
    V2, pi2 = value_iteration(env_lw, gamma=0.95)
    display_lineworld_mb(V2, pi2, env_lw)
