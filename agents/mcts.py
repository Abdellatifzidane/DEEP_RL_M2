import math
import random


class MCTSNode:
    __slots__ = ['parent', 'action', 'player', 'children',
                 'wins', 'visits', 'untried_actions']

    def __init__(self, parent=None, action=None, player=None):
        self.parent = parent
        self.action = action
        self.player = player  # joueur qui a joué l'action menant à ce noeud
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = None

    def uct(self, c):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child_uct(self, c):
        return max(self.children, key=lambda n: n.uct(c))

    def most_visited_child(self):
        return max(self.children, key=lambda n: n.visits)


class MCTSAgent:
    def __init__(self, num_simulations=1000, c=1.41):
        self.num_simulations = num_simulations
        self.c = c

    def choose_action(self, env):
        root = MCTSNode()
        root.untried_actions = list(env.get_available_actions())
        root.visits = 1  # éviter log(0) dans UCT des enfants

        for _ in range(self.num_simulations):
            node = root
            sim_env = env.clone()

            # 1. Sélection : descendre avec UCT jusqu'à un noeud non fully expanded
            while not node.untried_actions and node.children:
                node = node.best_child_uct(self.c)
                sim_env.step(node.action)

            # 2. Expansion : ajouter un enfant
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                player = sim_env.current_player
                sim_env.step(action)
                child = MCTSNode(parent=node, action=action, player=player)
                if not sim_env.is_terminal():
                    child.untried_actions = list(sim_env.get_available_actions())
                else:
                    child.untried_actions = []
                node.children.append(child)
                node = child

            # 3. Simulation (rollout random)
            while not sim_env.is_terminal():
                actions = sim_env.get_available_actions()
                if not actions:
                    break
                sim_env.step(random.choice(actions))

            # 4. Backpropagation
            score = sim_env.get_score()
            winner = sim_env.current_player if score > 0 else None
            while node is not None:
                node.visits += 1
                if winner is not None and node.player == winner:
                    node.wins += 1
                node = node.parent

        return root.most_visited_child().action
