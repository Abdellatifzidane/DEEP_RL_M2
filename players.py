
from enum import Enum
import random

class PlayerType(Enum):
    HUMAN = 0
    RANDOM = 1
    AGENT = 2

class Player:
    def __init__(self, player_type):
        self.player_type = player_type  

    def choose_action(self, action_type, env):
        # à implémenter dans les classes dérivées
        pass      

class HumanPlayer(Player):
    def __init__(self):
        super().__init__(PlayerType.HUMAN)    

    def choose_action(self, action_type, env):
        # demander à l'utilisateur de choisir une action
        if action_type == 0: # choisir une pièce
            available_pieces = [i for i, available in enumerate(env.ramaining_pieces) if available]
            print("Pièces disponibles: {}".format(available_pieces))
            piece_index = int(input("Choisissez une pièce : "))
            return piece_index
        elif action_type == 1: # poser la pièce
            available_pieces = [i for i, available in enumerate(env.remaining_cells) if available]
            print("Cellules disponibles: {}".format(available_pieces))
            cell_index = int(input("Choisissez une cellule : "))
            return cell_index
        
            
class RandomPlayer(Player):
    def __init__(self):
        super().__init__(PlayerType.RANDOM)

    def choose_action(self, action_type,env):
        # choisir une action aléatoire parmi les actions autorisées
        if action_type == 0: # choisir une pièce
            available_pieces = [i for i, available in enumerate(env.ramaining_pieces) if available]
            return random.choice(available_pieces)
        elif action_type == 1: # poser la pièce
            available_cells = [i for i, available in enumerate(env.remaining_cells) if available]
            return random.choice(available_cells)    


class AgentPlayer(Player):
    def __init__(self, model):
        super().__init__(PlayerType.AGENT)
        pass     
