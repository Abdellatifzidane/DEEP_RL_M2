import random
from collections import defaultdict


class TabularQLearningAgent:
    def __init__(
        self,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(float)

    def q_value(self, state, action):
        return self.q_table[(state, action)]

    def best_action(self, state, actions):
        if not actions:
            return None

        max_q = max(self.q_value(state, action) for action in actions)
        best_actions = [
            action for action in actions
            if self.q_value(state, action) == max_q
        ]
        return random.choice(best_actions)

    def select_action(self, state, actions, training=True):
        if not actions:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(actions)

        return self.best_action(state, actions)

    def update(self, state, action, reward, next_state, next_actions, done):
        current_q = self.q_value(state, action)

        if done or not next_actions:
            target = reward
        else:
            best_next_q = max(self.q_value(next_state, a) for a in next_actions)
            target = reward + self.gamma * best_next_q

        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=1000):
        scores = []

        for _ in range(num_episodes):
            env.reset()

            while not env.is_terminal():
                state = env.get_state()
                actions = env.get_available_actions()
                action = self.select_action(state, actions, training=True)

                _, reward = env.step(action)

                next_state = env.get_state()
                next_actions = env.get_available_actions()
                done = env.is_terminal()

                self.update(state, action, reward, next_state, next_actions, done)

            if hasattr(env, "get_score"):
                scores.append(env.get_score())

            self.decay_epsilon()

        return scores

    def evaluate(self, env, num_episodes=100):
        scores = []

        for _ in range(num_episodes):
            env.reset()

            while not env.is_terminal():
                state = env.get_state()
                actions = env.get_available_actions()
                action = self.select_action(state, actions, training=False)
                env.step(action)

            if hasattr(env, "get_score"):
                scores.append(env.get_score())

        if not scores:
            return None

        return sum(scores) / len(scores)