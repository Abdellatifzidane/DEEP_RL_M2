import pygame
import sys
import os
import threading
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from environnements.quarto.quatro import Quatro, PIECES, BOARD_SIZE, NB_PIECES, ACTION_SIZE, STATE_SIZE
from agents.alpha_zero import AlphaZeroAgent
from agents.ppo_a2c import PPO_A2C
from agents.expert_apprentice import ExpertApprentice
from agents.mcts import MCTSAgent

# --- Chemins modeles ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
AZ_MODEL_PATH = os.path.join(MODEL_DIR, "az_quarto_100000g.pt")
PPO_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_quarto_100000ep.pt")
EXIT_MODEL_PATH = os.path.join(MODEL_DIR, "exit_quarto_100000g.pt")
REINFORCE_MODEL_PATH = os.path.join(MODEL_DIR, "reinforce_300000.pt")
REINFORCE_MEAN_MODEL_PATH = os.path.join(MODEL_DIR, "mean_300000.pt")
REINFORCE_CRITIC_MODEL_PATH = os.path.join(MODEL_DIR, "critic_300000.pt")

# --- Couleurs ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_GRAY = (230, 230, 230)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GREEN = (50, 180, 50)
YELLOW = (220, 200, 50)
HIGHLIGHT = (255, 255, 100)
HOVER_COLOR = (220, 240, 255)
BG_COLOR = (245, 245, 240)
WIN_HIGHLIGHT = (100, 255, 100, 100)
WIN_LINE_COLOR = (0, 200, 0)

ATTR_NAMES = ["Taille", "Couleur", "Forme", "Remplissage"]
ATTR_VALUES = {
    0: {1: "Grand", -1: "Petit"},
    1: {1: "Rouge", -1: "Bleu"},
    2: {1: "Rond", -1: "Carre"},
    3: {1: "Plein", -1: "Creux"},
}

# --- Types d'action (pour l'affichage) ---
CHOOSE_PIECE = 0
PLACE_PIECE = 1

# --- Dimensions ---
CELL_SIZE = 100
BOARD_PX = BOARD_SIZE * CELL_SIZE
PANEL_W = 280
WINDOW_W = BOARD_PX + PANEL_W
WINDOW_H = BOARD_PX + 60

# --- Etats ---
STATE_MENU = 0
STATE_PLAYING = 1
STATE_GAME_OVER = 2

# --- Types de joueur ---
PLAYER_HUMAN = "Humain"
PLAYER_RANDOM = "Random"
PLAYER_AZ = "AlphaZero"
PLAYER_PPO = "PPO 100k"
PLAYER_MCTS = "MCTS"
PLAYER_EXIT = "ExIt 100k"
PLAYER_REINFORCE = "REINFORCE"
PLAYER_REINFORCE_MEAN = "R. Mean"
PLAYER_REINFORCE_CRITIC = "R. Critic"

ALL_PLAYER_TYPES = [
    PLAYER_HUMAN, PLAYER_RANDOM, PLAYER_AZ, PLAYER_PPO,
    PLAYER_MCTS, PLAYER_EXIT, PLAYER_REINFORCE,
    PLAYER_REINFORCE_MEAN, PLAYER_REINFORCE_CRITIC,
]


# ── Reseau REINFORCE (68 → 256 → 256 → 32) ────────────────────────────────

class ReinforcePolicy(nn.Module):
    def __init__(self, state_dim=68, action_dim=32, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


# ── Cache d'agents (chargement paresseux) ───────────────────────────────────

_agents_cache = {}


def _get_agent(name):
    if name in _agents_cache:
        return _agents_cache[name]

    agent = None

    if name == PLAYER_AZ:
        agent = AlphaZeroAgent(
            state_dim=STATE_SIZE, action_dim=ACTION_SIZE,
            hidden_dim=256, num_simulations=200,
        )
        agent.load(AZ_MODEL_PATH)
        agent.network.eval()
        print(f"AlphaZero charge depuis {AZ_MODEL_PATH}")

    elif name == PLAYER_PPO:
        agent = PPO_A2C(
            state_dim=STATE_SIZE, action_dim=ACTION_SIZE,
            hidden_dim=256,
        )
        agent.load(PPO_MODEL_PATH)
        agent.policy.eval()
        print(f"PPO charge depuis {PPO_MODEL_PATH}")

    elif name == PLAYER_EXIT:
        agent = ExpertApprentice(
            state_dim=STATE_SIZE, action_dim=ACTION_SIZE,
            hidden_dim=256,
        )
        agent.load(EXIT_MODEL_PATH)
        agent.network.eval()
        print(f"ExIt charge depuis {EXIT_MODEL_PATH}")

    elif name == PLAYER_MCTS:
        agent = MCTSAgent(num_simulations=500)
        print("MCTS initialise (500 simulations)")

    elif name in (PLAYER_REINFORCE, PLAYER_REINFORCE_MEAN, PLAYER_REINFORCE_CRITIC):
        paths = {
            PLAYER_REINFORCE: REINFORCE_MODEL_PATH,
            PLAYER_REINFORCE_MEAN: REINFORCE_MEAN_MODEL_PATH,
            PLAYER_REINFORCE_CRITIC: REINFORCE_CRITIC_MODEL_PATH,
        }
        path = paths[name]
        net = ReinforcePolicy(state_dim=68, action_dim=32, hidden_dim=256)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["policy_state_dict"])
        net.eval()
        agent = net
        print(f"{name} charge depuis {path}")

    _agents_cache[name] = agent
    return agent


def _agent_choose_action(name, agent, env):
    """Choisit une action selon le type d'agent."""
    available = env.get_available_actions()

    if name == PLAYER_RANDOM:
        return random.choice(list(available))

    if name == PLAYER_AZ:
        return agent.select_action(env)

    if name == PLAYER_MCTS:
        return agent.choose_action(env)

    if name == PLAYER_PPO:
        state = env.get_state()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = agent.policy(state_t)
            mask = torch.zeros(ACTION_SIZE)
            mask[available] = 1
            action_probs = action_probs[0] * mask
            total = action_probs.sum()
            if total > 0:
                action_probs = action_probs / total
            else:
                action_probs = mask / mask.sum()
            return available[torch.argmax(action_probs[available]).item()]

    if name == PLAYER_EXIT:
        state = env.encode_state()
        return agent.select_action(state, available)

    if name in (PLAYER_REINFORCE, PLAYER_REINFORCE_MEAN, PLAYER_REINFORCE_CRITIC):
        # REINFORCE utilise 68 dimensions (pas le joueur courant)
        state = env.encode_state()[:68]
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = agent(state_t)[0].numpy()
            mask = np.zeros(ACTION_SIZE, dtype=np.float32)
            mask[available] = 1.0
            probs = probs * mask
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs[available] = 1.0 / len(available)
            return int(np.argmax(probs))

    return random.choice(list(available))


# ── GUIPlayer (humain via clics) ──────────────────────────────────────────────

class GUIPlayer:
    """Joueur humain. choose_action() bloque jusqu'a ce que la GUI envoie un clic."""

    def __init__(self):
        self._event = threading.Event()
        self._choice = None
        self.waiting_for = None  # CHOOSE_PIECE ou PLACE_PIECE

    def choose_flat_action(self, env):
        """Bloque jusqu'au clic, retourne une action plate (0-31)."""
        if env.current_piece is None:
            self.waiting_for = CHOOSE_PIECE
        else:
            self.waiting_for = PLACE_PIECE

        self._event.clear()
        self._choice = None
        self._event.wait()
        self.waiting_for = None
        return self._choice

    def set_choice(self, flat_action):
        """Appelee par la GUI quand l'utilisateur clique."""
        self._choice = flat_action
        self._event.set()


# ── Dessin de piece ───────────────────────────────────────────────────────────

def draw_piece(screen, piece, cx, cy, size, selected=False):
    """Dessine une piece avec ses 4 attributs."""
    p_size, p_color, p_shape, p_fill = piece
    r = int(size * 0.42) if p_size == 1 else int(size * 0.28)
    color = RED if p_color == 1 else BLUE
    width = 0 if p_fill == 1 else 3

    if selected:
        pygame.draw.circle(screen, YELLOW, (cx, cy), r + 5, 3)

    if p_shape == 1:  # cercle
        pygame.draw.circle(screen, color, (cx, cy), r, width)
    else:  # carre
        rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
        pygame.draw.rect(screen, color, rect, width)


# ── GUI principale ────────────────────────────────────────────────────────────

class QuartoGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Quarto")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.font_big = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 13)

        self.state = STATE_MENU
        self.env = None
        self.players = [None, None]
        self.player_type_names = [PLAYER_HUMAN, PLAYER_HUMAN]
        self.gui_players = []
        self.game_thread = None
        self.winner = None
        self.game_finished = False

        self.hover_cell = None
        self.hover_piece = None
        self.winning_line = None
        self.winning_attr = None

        self.menu_choices = [0, 2]  # indices dans ALL_PLAYER_TYPES

        # Mode pas-a-pas pour les coups IA
        self._step_event = threading.Event()
        self._waiting_step = False  # True quand un AI attend le clic "Suivant"
        self._last_action_desc = ""  # description du dernier coup joue
        self._next_btn = None  # rect du bouton Suivant

    # ========== MENU ==========
    def draw_menu(self):
        self.screen.fill(BG_COLOR)
        title = self.font_big.render("QUARTO", True, BLACK)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 30))

        subtitle = self.font.render("Selectionnez les joueurs", True, DARK_GRAY)
        self.screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 68))

        for p in range(2):
            y = 110 + p * 180
            label = self.font.render(f"Joueur {p + 1}:", True, BLACK)
            self.screen.blit(label, (30, y))

            # Grille de boutons : 3 colonnes x 3 lignes
            cols = 3
            btn_w = 170
            btn_h = 30
            gap_x = 8
            gap_y = 6
            start_x = 30
            start_y = y + 28

            for i, name in enumerate(ALL_PLAYER_TYPES):
                row = i // cols
                col = i % cols
                bx = start_x + col * (btn_w + gap_x)
                by = start_y + row * (btn_h + gap_y)
                rect = pygame.Rect(bx, by, btn_w, btn_h)

                if self.menu_choices[p] == i:
                    pygame.draw.rect(self.screen, GREEN, rect, border_radius=5)
                    tc = WHITE
                else:
                    pygame.draw.rect(self.screen, GRAY, rect, border_radius=5)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1, border_radius=5)
                    tc = BLACK
                txt = self.font_small.render(name, True, tc)
                self.screen.blit(txt, (bx + btn_w // 2 - txt.get_width() // 2,
                                       by + btn_h // 2 - txt.get_height() // 2))

        btn = pygame.Rect(WINDOW_W // 2 - 80, WINDOW_H - 55, 160, 45)
        pygame.draw.rect(self.screen, GREEN, btn, border_radius=8)
        txt = self.font_big.render("JOUER", True, WHITE)
        self.screen.blit(txt, (btn.centerx - txt.get_width() // 2, btn.centery - txt.get_height() // 2))

    def handle_menu_click(self, pos):
        cols = 3
        btn_w = 170
        btn_h = 30
        gap_x = 8
        gap_y = 6
        start_x = 30

        for p in range(2):
            y = 110 + p * 180
            start_y = y + 28
            for i in range(len(ALL_PLAYER_TYPES)):
                row = i // cols
                col = i % cols
                bx = start_x + col * (btn_w + gap_x)
                by = start_y + row * (btn_h + gap_y)
                if pygame.Rect(bx, by, btn_w, btn_h).collidepoint(pos):
                    self.menu_choices[p] = i
                    return

        if pygame.Rect(WINDOW_W // 2 - 80, WINDOW_H - 55, 160, 45).collidepoint(pos):
            self.start_new_game()

    # ========== LANCEMENT ==========
    def start_new_game(self):
        self.gui_players = []
        self.env = Quatro()
        self.env.reset()

        for i in range(2):
            name = ALL_PLAYER_TYPES[self.menu_choices[i]]
            self.player_type_names[i] = name
            if name == PLAYER_HUMAN:
                gp = GUIPlayer()
                self.players[i] = gp
                self.gui_players.append(gp)
            else:
                self.players[i] = _get_agent(name)

        self.winner = None
        self.game_finished = False
        self.winning_line = None
        self.winning_attr = None
        self.state = STATE_PLAYING

        self.game_thread = threading.Thread(target=self._run_game, daemon=True)
        self.game_thread.start()

    def _describe_action(self, action, name):
        """Retourne une description lisible de l'action."""
        action_type, index = Quatro.decode_action(action)
        if action_type == 0:
            piece = PIECES[index]
            attrs = []
            attrs.append(ATTR_VALUES[0][piece[0]])
            attrs.append(ATTR_VALUES[1][piece[1]])
            attrs.append(ATTR_VALUES[2][piece[2]])
            attrs.append(ATTR_VALUES[3][piece[3]])
            return f"{name} choisit piece {index} ({', '.join(attrs)})"
        else:
            row = index // BOARD_SIZE
            col = index % BOARD_SIZE
            return f"{name} pose en ({row},{col})"

    def _run_game(self):
        """Boucle de jeu utilisant env.step() directement."""
        env = self.env
        has_human = PLAYER_HUMAN in self.player_type_names

        while not env.is_terminal():
            available = env.get_available_actions()
            if not available:
                break

            cp = env.current_player
            name = self.player_type_names[cp]
            player = self.players[cp]

            if name == PLAYER_HUMAN:
                action = player.choose_flat_action(env)
            else:
                action = _agent_choose_action(name, player, env)
                # Mode pas-a-pas : attendre le clic "Suivant"
                self._last_action_desc = self._describe_action(action, name)
                self._step_event.clear()
                self._waiting_step = True
                self._step_event.wait()
                self._waiting_step = False

            env.step(action)

        if env.get_score() > 0:
            self.winner = env.current_player
        else:
            self.winner = -1
        self.game_finished = True

    # ========== DESSIN PLATEAU ==========
    def draw_board(self):
        env = self.env
        board_rect = pygame.Rect(0, 0, BOARD_PX, BOARD_PX)
        pygame.draw.rect(self.screen, WHITE, board_rect)

        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(self.screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, BOARD_PX), 2)
            pygame.draw.line(self.screen, BLACK, (0, i * CELL_SIZE), (BOARD_PX, i * CELL_SIZE), 2)

        gui_waiting_place = self._get_waiting_action() == PLACE_PIECE
        if self.hover_cell is not None and gui_waiting_place:
            row, col = self.hover_cell
            idx = row * BOARD_SIZE + col
            if env.remaining_cells[idx]:
                r = pygame.Rect(col * CELL_SIZE + 1, row * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(self.screen, HIGHLIGHT, r)

        if self.winning_line is not None:
            surf = pygame.Surface((CELL_SIZE - 4, CELL_SIZE - 4), pygame.SRCALPHA)
            surf.fill(WIN_HIGHLIGHT)
            for idx in self.winning_line:
                r = idx // BOARD_SIZE
                c = idx % BOARD_SIZE
                self.screen.blit(surf, (c * CELL_SIZE + 2, r * CELL_SIZE + 2))

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                idx = i * BOARD_SIZE + j
                piece = env.board[idx]
                if piece is not None:
                    cx = j * CELL_SIZE + CELL_SIZE // 2
                    cy = i * CELL_SIZE + CELL_SIZE // 2
                    draw_piece(self.screen, piece, cx, cy, CELL_SIZE)

        if self.winning_line is not None:
            cells = self.winning_line
            r0, c0 = cells[0] // BOARD_SIZE, cells[0] % BOARD_SIZE
            r3, c3 = cells[3] // BOARD_SIZE, cells[3] % BOARD_SIZE
            start = (c0 * CELL_SIZE + CELL_SIZE // 2, r0 * CELL_SIZE + CELL_SIZE // 2)
            end = (c3 * CELL_SIZE + CELL_SIZE // 2, r3 * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.line(self.screen, WIN_LINE_COLOR, start, end, 5)

    # ========== PANNEAU LATERAL ==========
    def draw_panel(self):
        env = self.env
        panel_x = BOARD_PX
        pygame.draw.rect(self.screen, BG_COLOR, (panel_x, 0, PANEL_W, BOARD_PX))
        pygame.draw.line(self.screen, BLACK, (panel_x, 0), (panel_x, BOARD_PX), 2)

        title = self.font.render("Pieces disponibles", True, BLACK)
        self.screen.blit(title, (panel_x + 10, 10))

        area_x = panel_x + 15
        area_y = 40
        spacing = 62

        gui_waiting_choose = self._get_waiting_action() == CHOOSE_PIECE

        for i in range(NB_PIECES):
            if not env.remaining_pieces[i]:
                continue
            row = i // 4
            col = i % 4
            cx = area_x + col * spacing + spacing // 2
            cy = area_y + row * spacing + spacing // 2

            is_hovered = (self.hover_piece == i and gui_waiting_choose)
            if is_hovered:
                bg = pygame.Rect(cx - 28, cy - 28, 56, 56)
                pygame.draw.rect(self.screen, HOVER_COLOR, bg, border_radius=5)

            draw_piece(self.screen, PIECES[i], cx, cy, spacing)
            num = self.font_small.render(str(i), True, DARK_GRAY)
            self.screen.blit(num, (cx - num.get_width() // 2, cy + 22))

        if env.current_piece is not None:
            info_y = 310
            txt = self.font.render("Piece a poser:", True, BLACK)
            self.screen.blit(txt, (panel_x + 10, info_y))
            draw_piece(self.screen, env.current_piece, panel_x + PANEL_W // 2, info_y + 55, 80)

    # ========== BARRE STATUT ==========
    def draw_status_bar(self):
        bar = pygame.Rect(0, BOARD_PX, WINDOW_W, 60)
        pygame.draw.rect(self.screen, (60, 60, 60), bar)

        env = self.env

        if self._waiting_step:
            # Mode pas-a-pas : afficher le coup et le bouton Suivant
            txt = self.font.render(self._last_action_desc, True, WHITE)
            self.screen.blit(txt, (10, BOARD_PX + 5))

            hint = self.font_small.render("Cliquez Suivant ou appuyez Espace", True, GRAY)
            self.screen.blit(hint, (10, BOARD_PX + 28))

            self._next_btn = pygame.Rect(WINDOW_W - 140, BOARD_PX + 10, 120, 40)
            pygame.draw.rect(self.screen, GREEN, self._next_btn, border_radius=8)
            btn_txt = self.font.render("Suivant >", True, WHITE)
            self.screen.blit(btn_txt, (self._next_btn.centerx - btn_txt.get_width() // 2,
                                       self._next_btn.centery - btn_txt.get_height() // 2))
        else:
            self._next_btn = None
            cp = env.current_player
            player_type = self.player_type_names[cp]
            waiting = self._get_waiting_action()

            if env.current_piece is None:
                action_txt = "choisit une piece pour l'adversaire"
            else:
                action_txt = "place la piece sur le plateau"

            if waiting is not None:
                pass
            elif not self.game_finished:
                action_txt = "reflechit..."

            msg = f"Joueur {cp + 1} ({player_type}) {action_txt}"
            txt = self.font.render(msg, True, WHITE)
            self.screen.blit(txt, (10, BOARD_PX + 20))

        turn_num = 16 - sum(env.remaining_cells)
        t = self.font_small.render(f"Tour: {turn_num}/16", True, GRAY)
        self.screen.blit(t, (WINDOW_W - 80, BOARD_PX + 48))

    # ========== ECRAN FIN ==========
    def draw_game_over(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        if self.winner >= 0:
            pt = self.player_type_names[self.winner]
            msg = f"Joueur {self.winner + 1} ({pt}) a gagne!"
            color = GREEN
        else:
            msg = "Match nul!"
            color = YELLOW

        txt = self.font_big.render(msg, True, color)
        self.screen.blit(txt, (WINDOW_W // 2 - txt.get_width() // 2, WINDOW_H // 2 - 50))

        if self.winning_attr is not None:
            pieces = [self.env.board[i] for i in self.winning_line]
            val = pieces[0][self.winning_attr]
            attr_name = ATTR_NAMES[self.winning_attr]
            attr_val = ATTR_VALUES[self.winning_attr][val]
            reason = f"Attribut commun: {attr_name} = {attr_val}"
            rtxt = self.font.render(reason, True, WHITE)
            self.screen.blit(rtxt, (WINDOW_W // 2 - rtxt.get_width() // 2, WINDOW_H // 2 - 15))

        btn1 = pygame.Rect(WINDOW_W // 2 - 80, WINDOW_H // 2 + 20, 160, 40)
        pygame.draw.rect(self.screen, GREEN, btn1, border_radius=8)
        t1 = self.font.render("Rejouer", True, WHITE)
        self.screen.blit(t1, (btn1.centerx - t1.get_width() // 2, btn1.centery - t1.get_height() // 2))

        btn2 = pygame.Rect(WINDOW_W // 2 - 80, WINDOW_H // 2 + 70, 160, 40)
        pygame.draw.rect(self.screen, DARK_GRAY, btn2, border_radius=8)
        t2 = self.font.render("Menu", True, WHITE)
        self.screen.blit(t2, (btn2.centerx - t2.get_width() // 2, btn2.centery - t2.get_height() // 2))

        return btn1, btn2

    # ========== DETECTION LIGNE GAGNANTE ==========
    def find_winning_line(self):
        board = self.env.board
        alignments = [
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
            [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
            [0, 5, 10, 15], [3, 6, 9, 12],
        ]
        for alignment in alignments:
            pieces = [board[i] for i in alignment]
            if any(p is None for p in pieces):
                continue
            for attr in range(4):
                if pieces[0][attr] == pieces[1][attr] == pieces[2][attr] == pieces[3][attr]:
                    self.winning_line = alignment
                    self.winning_attr = attr
                    return
        self.winning_line = None
        self.winning_attr = None

    # ========== UTILITAIRES ==========
    def _get_waiting_action(self):
        for gp in self.gui_players:
            if gp.waiting_for is not None:
                return gp.waiting_for
        return None

    def _get_waiting_gui_player(self):
        for gp in self.gui_players:
            if gp.waiting_for is not None:
                return gp
        return None

    def handle_game_click(self, pos):
        gp = self._get_waiting_gui_player()
        if gp is None:
            return

        env = self.env
        x, y = pos

        if gp.waiting_for == PLACE_PIECE and x < BOARD_PX and y < BOARD_PX:
            col = x // CELL_SIZE
            row = y // CELL_SIZE
            idx = row * BOARD_SIZE + col
            if env.remaining_cells[idx]:
                gp.set_choice(NB_PIECES + idx)

        elif gp.waiting_for == CHOOSE_PIECE and x >= BOARD_PX:
            area_x = BOARD_PX + 15
            area_y = 40
            spacing = 62
            x_rel = x - area_x
            y_rel = y - area_y
            if x_rel >= 0 and y_rel >= 0:
                col = x_rel // spacing
                row = y_rel // spacing
                if 0 <= col < 4 and 0 <= row < 4:
                    idx = row * 4 + col
                    if idx < 16 and env.remaining_pieces[idx]:
                        gp.set_choice(idx)

    def update_hover(self, pos):
        x, y = pos
        self.hover_cell = None
        self.hover_piece = None
        if x < BOARD_PX and y < BOARD_PX:
            self.hover_cell = (y // CELL_SIZE, x // CELL_SIZE)
        elif x >= BOARD_PX:
            area_x = BOARD_PX + 15
            area_y = 40
            spacing = 62
            xr = x - area_x
            yr = y - area_y
            if xr >= 0 and yr >= 0:
                col = xr // spacing
                row = yr // spacing
                if 0 <= col < 4 and 0 <= row < 4:
                    self.hover_piece = row * 4 + col

    # ========== BOUCLE PRINCIPALE ==========
    def run(self):
        running = True
        go_buttons = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Espace ou Entree pour avancer en mode pas-a-pas
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        if self._waiting_step:
                            self._step_event.set()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == STATE_MENU:
                        self.handle_menu_click(event.pos)
                    elif self.state == STATE_PLAYING:
                        # Verifier clic sur bouton "Suivant"
                        if self._waiting_step and self._next_btn and self._next_btn.collidepoint(event.pos):
                            self._step_event.set()
                        else:
                            self.handle_game_click(event.pos)
                    elif self.state == STATE_GAME_OVER and go_buttons:
                        btn1, btn2 = go_buttons
                        if btn1.collidepoint(event.pos):
                            self.start_new_game()
                        elif btn2.collidepoint(event.pos):
                            self.state = STATE_MENU
                elif event.type == pygame.MOUSEMOTION:
                    if self.state == STATE_PLAYING:
                        self.update_hover(event.pos)

            self.screen.fill(BG_COLOR)

            if self.state == STATE_MENU:
                self.draw_menu()
                go_buttons = None

            elif self.state == STATE_PLAYING:
                self.draw_board()
                self.draw_panel()
                self.draw_status_bar()

                if self.game_finished:
                    self.find_winning_line()
                    self.state = STATE_GAME_OVER
                    go_buttons = self.draw_game_over()

            elif self.state == STATE_GAME_OVER:
                self.draw_board()
                self.draw_panel()
                self.draw_status_bar()
                go_buttons = self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    gui = QuartoGUI()
    gui.run()
