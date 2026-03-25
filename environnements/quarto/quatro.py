import itertools
import numpy as np
import random
from environnements.base_env import BaseEnv

BOARD_SIZE = 4
NB_CELLS = 16
NB_PIECES = 16
NB_ATTRS = 4  # taille, couleur, forme, remplissage
NB_ACTION_TYPES = 2
from enum import Enum

PIECES = list(itertools.product([-1,1], repeat=NB_ATTRS))

for i, piece in enumerate(PIECES):
    print("Pièce {}: {}".format(i, piece))


class Quatro(BaseEnv):
    """
    Quarto environment compatible avec BaseEnv.

    Jeu à deux joueurs, chaque joueur appelle step() à son tour.
      - reset() : état initial, current_piece = None
      - Premier step : le joueur 1 choisit une pièce → action = piece_index (0-15)
      - Steps suivants : poser la pièce courante + choisir la prochaine
        → action = cell_index * NB_PIECES + piece_index
      - Dernier step (plus de pièces à choisir) : juste poser
        → action = cell_index
      - reward : +1 victoire du joueur qui vient de poser, 0 sinon
    """

    def __init__(self, seed=None):
        self.board = [None] * NB_CELLS
        self.ramaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None
        self.done = False
        self._score = 0.0
        self.current_player = 0  # 0 ou 1

    # ── BaseEnv interface ────────────────────────────────────────────

    def reset(self):
        self.board = [None] * NB_CELLS
        self.ramaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None
        self.done = False
        self._score = 0.0
        self.current_player = 0
        return self.get_state()

    def get_state(self):
        return tuple(self._build_state_list())

    def get_available_actions(self):
        if self.done:
            return []

        # Phase 1 : choisir une pièce (pas de pièce courante à poser)
        if self.current_piece is None:
            return [i for i, v in enumerate(self.ramaining_pieces) if v == 1]

        # Phase 2 : poser la pièce + choisir la prochaine
        free_cells = [i for i, v in enumerate(self.remaining_cells) if v == 1]
        avail_pieces = [i for i, v in enumerate(self.ramaining_pieces) if v == 1]

        if avail_pieces:
            return [cell * NB_PIECES + piece
                    for cell in free_cells for piece in avail_pieces]
        else:
            # Dernier placement, plus de pièce à choisir
            return list(free_cells)

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0

        # Phase 1 : choisir une pièce pour l'adversaire
        if self.current_piece is None:
            self.current_piece = PIECES[action]
            self.ramaining_pieces[action] = 0
            self.current_player = 1 - self.current_player
            return self.get_state(), 0.0

        # Phase 2 : poser + choisir
        avail_pieces = [i for i, v in enumerate(self.ramaining_pieces) if v == 1]

        if avail_pieces:
            cell_index = action // NB_PIECES
            piece_index = action % NB_PIECES
        else:
            cell_index = action
            piece_index = None

        # Poser la pièce courante
        self.board[cell_index] = self.current_piece
        self.remaining_cells[cell_index] = 0
        self.current_piece = None

        if self.check_win():
            self.done = True
            self._score = 1.0
            return self.get_state(), 1.0

        if self._is_board_full():
            self.done = True
            self._score = 0.0
            return self.get_state(), 0.0

        # Choisir la pièce pour l'adversaire
        if piece_index is not None:
            self.current_piece = PIECES[piece_index]
            self.ramaining_pieces[piece_index] = 0

        self.current_player = 1 - self.current_player
        return self.get_state(), 0.0

    def is_terminal(self):
        return self.done

    def get_score(self):
        return self._score

    # ── Méthodes internes ────────────────────────────────────────────

    def _build_state_list(self):
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
        return state

    def _is_board_full(self):
        return all(v == 0 for v in self.remaining_cells)

    # ── Legacy API (utilisée par game.py et Players) ─────────────────

    def encode_state(self):
        return np.array(self._build_state_list())

    def action_encode(self, action_type, index):
        action = [0] * NB_ACTION_TYPES
        action[action_type] = 1
        action.append(index)
        return np.array(action)

    def apply_action(self, action_type, index):
        if action_type == 1:  # poser la pièce
            self.board[index] = self.current_piece
            self.remaining_cells[index] = 0
            self.current_piece = None
        elif action_type == 0:  # choisir une pièce
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
