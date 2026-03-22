import itertools
import numpy as np
import random
from environnements.quarto.players import RandomPlayer, HumanPlayer, AgentPlayer

BOARD_SIZE = 4
NB_CELLS = 16
NB_PIECES = 16
NB_ATTRS = 4  # taille, couleur, forme, remplissage
NB_ACTION_TYPES = 2
from enum import Enum

PIECES = list(itertools.product([-1,1], repeat=NB_ATTRS))

for i, piece in enumerate(PIECES):
    print("Pièce {}: {}".format(i, piece))


class Quatro:
    def __init__(self):
        self.board = [None] * NB_CELLS # None pour les cellules vides, sinon le vecteur de caractéristiques de la pièce 
        self.ramaining_pieces = [1] * NB_PIECES # mask sur les pions autoriés 
        self.remaining_cells = [1] * NB_CELLS # mask sur les cellules autorisées
        self.current_piece = None # la pièce choiste par l'adversaire, à poser par le joueur courant, à remplacer par un vecteur de caractéristiques de la pièce choisie 

    def reset(self):
        self.board = [None] * NB_CELLS
        self.ramaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None


    def encode_state(self):
        # encode l'état du jeu en un vecteur de caractéristiques
        state = []
        for cell in self.board:
            if cell is None:
                state.extend([0] * NB_ATTRS)
            else:
                state.extend(cell)
        if self.current_piece is None:
            state.extend([0] * NB_ATTRS)
        else:
            state.extend(self.current_piece)
        return np.array(state)
    
    def action_encode(self, action_type, index):
        # Encode une action en un vecteur de caractéristiques
        action = [0] * NB_ACTION_TYPES
        action[action_type] = 1  # One-hot encoding pour le type d'action
        action.append(index)
        return np.array(action)
    
    def apply_action(self, action_type, index):
        # appliquer une action au jeu
        if action_type == 1: # poser la pièce
            self.board[index] = self.current_piece
            self.remaining_cells[index] = 0
            self.current_piece = None
        elif action_type == 0: # choisir une pièce
            self.current_piece = PIECES[index]
            self.ramaining_pieces[index] = 0    
    
    def check_win(self):
        alignments = [
        # lignes
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        # colonnes
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
        # diagonales
        [0, 5, 10, 15],
        [3, 6, 9, 12],
        ]
        for alignment in alignments:
            pieces = []
            for piece in alignment:
                if self.board[piece] is None:
                    break
                pieces.append(self.board[piece])
            
            if len(pieces) < 4:
                continue

            for attr in range(NB_ATTRS):
                if pieces[0][attr] == pieces[1][attr] == pieces[2][attr] == pieces[3][attr]:
                    return True 
        return False    
    


