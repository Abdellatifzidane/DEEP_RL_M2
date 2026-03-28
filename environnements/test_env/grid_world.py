from environnements.base_env import BaseEnv


class GridWorld(BaseEnv):
    """
    GridWorld simple :
    - grille NxN
    - départ en haut à gauche (0,0)
    - objectif en bas à droite (size-1, size-1)
    """

    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        self.state = None
        self.steps = 0
        self.max_steps = size * size  # pour éviter les boucles infinies

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.get_state()

    def get_state(self):
        # état sous forme de tuple (hashable + compatible tabulaire)
        return self.state

    def get_available_actions(self):
        actions = []
        x, y = self.state

        # 0: haut, 1: bas, 2: gauche, 3: droite
        if x > 0:
            actions.append(0)
        if x < self.size - 1:
            actions.append(1)
        if y > 0:
            actions.append(2)
        if y < self.size - 1:
            actions.append(3)

        return actions

    def step(self, action):
        x, y = self.state

        if action == 0:    # haut
            x -= 1
        elif action == 1:  # bas
            x += 1
        elif action == 2:  # gauche
            y -= 1
        elif action == 3:  # droite
            y += 1

        self.state = (x, y)
        self.steps += 1

        # récompense
        if self.state == self.goal:
            reward = 1.0
        else:
            reward = -0.01  # petite pénalité pour encourager le plus court chemin

        return self.get_state(), reward

    def is_terminal(self):
        return self.state == self.goal or self.steps >= self.max_steps

    def get_score(self):
        # score simple : 1 si réussi, 0 sinon
        return 1 if self.state == self.goal else 0