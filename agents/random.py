import random


class RandomAgent:
    def choose_action(self, env):
        actions = env.get_available_actions()
        if not actions:
            return None
        return random.choice(actions)