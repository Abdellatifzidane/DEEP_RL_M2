import itertools
import numpy as np
from environnements.base_env import BaseEnv

BOARD_SIZE = 4
NB_CELLS = 16
NB_PIECES = 16
NB_ATTRS = 4
NB_ACTION_TYPES = 2

PIECES = list(itertools.product([-1, 1], repeat=NB_ATTRS))


class Quatro(BaseEnv):
    """
    Quarto environment compatible avec BaseEnv.

    Action = (type, index) :
      - (0, i) : choisir la pièce i pour l'adversaire
      - (1, j) : poser la pièce courante sur la case j

    State = tuple(68) :
      - 16 cases × 4 attributs
      - + 4 attributs de la pièce courante

    Reward :
      - +1 si victoire
      - 0 sinon
    """

    def __init__(self, seed=None):
        self.board = [None] * NB_CELLS
        self.ramaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None
        self.done = False
        self._score = 0.0
        self.current_player = 0
        self.aa_buffer = []

        # espace d'action fixe pour les agents Deep RL :
        # 16 actions "choisir une pièce" + 16 actions "poser sur une case"
        self.action_size = NB_PIECES + NB_CELLS

    def reset(self):
        self.board = [None] * NB_CELLS
        self.ramaining_pieces = [1] * NB_PIECES
        self.remaining_cells = [1] * NB_CELLS
        self.current_piece = None
        self.done = False
        self._score = 0.0
        self.current_player = 0
        self.aa_buffer.clear()
        return self.get_state()

    def get_state(self):
        return tuple(self._build_state_list())

    def get_available_actions(self):
        self.aa_buffer.clear()

        if self.done:
            return []

        if self.current_piece is None:
            # choisir une pièce pour l'adversaire
            for i, v in enumerate(self.ramaining_pieces):
                if v == 1:
                    self.aa_buffer.append((0, i))
        else:
            # poser la pièce courante sur une case libre
            for j, v in enumerate(self.remaining_cells):
                if v == 1:
                    self.aa_buffer.append((1, j))

        return self.aa_buffer

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0

        action_type, index = action

        if action_type == 0:
            # choisir une pièce pour l'adversaire
            self.current_piece = PIECES[index]
            self.ramaining_pieces[index] = 0
            self.current_player = 1 - self.current_player
            return self.get_state(), 0.0

        # action_type == 1 : poser la pièce courante
        self.board[index] = self.current_piece
        self.remaining_cells[index] = 0
        self.current_piece = None

        if self.check_win():
            self.done = True
            self._score = 1.0
            return self.get_state(), 1.0

        if self._is_board_full():
            self.done = True
            self._score = 0.0
            return self.get_state(), 0.0

        return self.get_state(), 0.0

    def is_terminal(self):
        return self.done

    def get_score(self):
        return self._score

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def determinize(self):
        return self.clone()

    def action_to_index(self, action):
        action_type, index = action
        if action_type == 0:
            return index
        return NB_PIECES + index

    def index_to_action(self, action_index):
        if action_index < NB_PIECES:
            return (0, action_index)
        return (1, action_index - NB_PIECES)

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

    def encode_state(self):
        return np.array(self._build_state_list())

    def action_encode(self, action_type, index):
        action = [0] * NB_ACTION_TYPES
        action[action_type] = 1
        action.append(index)
        return np.array(action)

    def apply_action(self, action_type, index):
        if action_type == 1:
            self.board[index] = self.current_piece
            self.remaining_cells[index] = 0
            self.current_piece = None
        elif action_type == 0:
            self.current_piece = PIECES[index]
            self.ramaining_pieces[index] = 0

    def check_win(self):
        alignments = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
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
                if (
                    pieces[0][attr]
                    == pieces[1][attr]
                    == pieces[2][attr]
                    == pieces[3][attr]
                ):
                    return True

        return False