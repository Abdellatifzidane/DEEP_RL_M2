import itertools
import numpy as np
from environnements.base_env import BaseEnv

BOARD_SIZE = 4
NB_CELLS = 16
NB_PIECES = 16
NB_ATTRS = 4  # taille, couleur, forme, remplissage
NB_ACTION_TYPES = 2
ACTION_SIZE = NB_PIECES + NB_CELLS  # 32: [0-15] choisir pièce, [16-31] poser sur case
STATE_SIZE = NB_CELLS * NB_ATTRS + NB_ATTRS + 1  # 69: plateau(64) + pièce(4) + joueur(1)

PIECES = tuple(itertools.product((-1, 1), repeat=NB_ATTRS))

# Alignements pour vérification de victoire (constante module, plus recréé à chaque appel)
ALIGNMENTS = (
    (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15),
    (0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
    (0, 5, 10, 15), (3, 6, 9, 12),
)


class Quatro(BaseEnv):
    """
    Quarto environment compatible avec BaseEnv.

    Action = index plat (0-31) :
      - 0-15  : choisir la pièce i pour l'adversaire
      - 16-31 : poser la pièce courante sur la case (j - 16)
    State = vecteur de taille 69 :
      - [0:64]  16 cases × 4 attributs (plateau)
      - [64:68] 4 attributs pièce courante
      - [68]    joueur courant (0 ou 1)
    Reward : +1 victoire, 0 sinon
    """

    def __init__(self, seed=None):
        self._state = np.zeros(STATE_SIZE, dtype=np.float32)
        self.board = [None] * NB_CELLS
        self.remaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None
        self.done = False
        self._score = 0.0
        self.current_player = 0
        self.aa_buffer = []

    def reset(self):
        self._state[:] = 0.0
        self.board = [None] * NB_CELLS
        self.remaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None
        self.done = False
        self._score = 0.0
        self.current_player = 0
        self.aa_buffer.clear()
        return self.get_state()

    def get_state(self):
        self._state[68] = self.current_player
        return tuple(self._state.tolist())

    def encode_state(self):
        self._state[68] = self.current_player
        return self._state.copy()

    def get_available_actions(self):
        """Retourne les indices plats d'actions valides (0-31).
        0-15 = choisir pièce, 16-31 = poser sur case."""
        self.aa_buffer.clear()
        if self.done:
            return self.aa_buffer
        if self.current_piece is None:
            for i, v in enumerate(self.remaining_pieces):
                if v == 1:
                    self.aa_buffer.append(i)
        else:
            for j, v in enumerate(self.remaining_cells):
                if v == 1:
                    self.aa_buffer.append(NB_PIECES + j)
        return self.aa_buffer

    def get_action_mask(self):
        """Retourne un masque float (taille ACTION_SIZE=32) pour masquer
        les actions invalides dans les probabilités du réseau."""
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        if self.done:
            return mask
        if self.current_piece is None:
            for i, v in enumerate(self.remaining_pieces):
                if v == 1:
                    mask[i] = 1.0
        else:
            for j, v in enumerate(self.remaining_cells):
                if v == 1:
                    mask[NB_PIECES + j] = 1.0
        return mask

    @staticmethod
    def decode_action(flat_index):
        """Convertit un index plat (0-31) en (action_type, index).
        0-15 → (0, i) choisir pièce i
        16-31 → (1, j) poser sur case j"""
        if flat_index < NB_PIECES:
            return 0, flat_index
        return 1, flat_index - NB_PIECES

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0

        # Accepte tuple (ancien format) ou int (nouveau format plat)
        if isinstance(action, (tuple, list)):
            action_type, index = action
        else:
            action_type, index = self.decode_action(action)

        if action_type == 0:
            # Choisir une pièce pour l'adversaire
            self.current_piece = PIECES[index]
            self.remaining_pieces[index] = 0
            self.current_player = 1 - self.current_player
            self._state[64:68] = self.current_piece
            self._state[68] = self.current_player
            return self.get_state(), 0.0

        # Poser la pièce courante
        self.board[index] = self.current_piece
        self.remaining_cells[index] = 0
        offset = index * NB_ATTRS
        self._state[offset:offset + NB_ATTRS] = self.current_piece
        self.current_piece = None
        self._state[64:68] = 0

        if self.check_win():
            self.done = True
            self._score = 1.0
            return self.get_state(), 1.0

        if all(v == 0 for v in self.remaining_cells):
            self.done = True
            self._score = 0.0
            return self.get_state(), 0.0

        return self.get_state(), 0.0

    def is_terminal(self):
        return self.done

    def get_score(self):
        return self._score

    def apply_action(self, action_type, index):
        """Applique une action sans retour d'état (compatibilité game.py)."""
        if action_type == 1:
            offset = index * NB_ATTRS
            self._state[offset:offset + NB_ATTRS] = self.current_piece
            self.board[index] = self.current_piece
            self.remaining_cells[index] = 0
            self.current_piece = None
            self._state[64:68] = 0
        elif action_type == 0:
            self.current_piece = PIECES[index]
            self.remaining_pieces[index] = 0
            self._state[64:68] = self.current_piece

    def check_win(self):
        for a, b, c, d in ALIGNMENTS:
            pa, pb, pc, pd = self.board[a], self.board[b], self.board[c], self.board[d]
            if pa is None or pb is None or pc is None or pd is None:
                continue
            for attr in range(NB_ATTRS):
                if pa[attr] == pb[attr] == pc[attr] == pd[attr]:
                    return True
        return False
