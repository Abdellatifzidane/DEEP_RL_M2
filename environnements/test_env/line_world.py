from environnements.base_env import BaseEnv


class LineWorld(BaseEnv):
    """
    LineWorld simple :
    - ligne de 5 états (0 à 4)
    - départ au milieu (1)
    - objectif à droite (4) avec récompense +1
    - piège à gauche (0) avec récompense -1
    - actions : 0=gauche, 1=droite
    """

    def __init__(self):
        self.start = 1
        self.goal = 4
        self.trap = 0

        self.state = None
        self.steps = 0
        self.max_steps = 25  # pour éviter les boucles infinies

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.get_state()

    def get_state(self):
        # état sous forme d'entier (hashable + compatible tabulaire)
        return self.state

    def get_available_actions(self):
        actions = []
        # 0: gauche, 1: droite
        if self.state > 0:
            actions.append(0)
        if self.state < 4:
            actions.append(1)
        return actions

    def step(self, action):
        if action == 0:  # gauche
            next_state = max(0, self.state - 1)
        elif action == 1:  # droite
            next_state = min(4, self.state + 1)
        else:
            next_state = self.state

        self.state = next_state
        self.steps += 1

        # récompense
        if self.state == self.goal:
            reward = 1.0
        elif self.state == self.trap:
            reward = -1.0
        else:
            reward = 0.0

        return self.get_state(), reward

    def is_terminal(self):
        return self.state in (self.trap, self.goal) or self.steps >= self.max_steps

    def get_score(self):
        # score simple : 1 si réussi, -1 si piège, 0 sinon
        if self.state == self.goal:
            return 1
        elif self.state == self.trap:
            return -1
        else:
            return 0
