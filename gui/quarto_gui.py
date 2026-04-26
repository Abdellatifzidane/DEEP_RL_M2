import pygame
import sys
import os
import threading
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environnements.quarto.quatro import Quatro, PIECES, BOARD_SIZE, NB_PIECES, ACTION_SIZE, STATE_SIZE
from agents.alpha_zero import AlphaZeroAgent

# --- Chemins modeles ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
AZ_MODEL_PATH = os.path.join(MODEL_DIR, "az_quarto_100000g.pt")

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
PLAYER_HUMAN = 0
PLAYER_RANDOM = 1
PLAYER_AGENT = 2


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


# ── Chargement AlphaZero ──────────────────────────────────────────────────────

_az_agent = None

def get_alphazero_agent():
    global _az_agent
    if _az_agent is None:
        _az_agent = AlphaZeroAgent(
            state_dim=STATE_SIZE, action_dim=ACTION_SIZE,
            hidden_dim=256, num_simulations=200,
        )
        _az_agent.load(AZ_MODEL_PATH)
        _az_agent.network.eval()
        print(f"AlphaZero charge depuis {AZ_MODEL_PATH}")
    return _az_agent


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
        self.font_small = pygame.font.SysFont("Arial", 14)

        self.state = STATE_MENU
        self.env = None
        self.players = [None, None]       # objets joueur (GUIPlayer ou None pour AI)
        self.player_types = [0, 0]        # PLAYER_HUMAN / RANDOM / AGENT
        self.current_player_index = 0     # 0 ou 1
        self.gui_players = []
        self.game_thread = None
        self.winner = None                # 0, 1, ou -1 (nul)
        self.game_finished = False

        self.hover_cell = None
        self.hover_piece = None
        self.winning_line = None
        self.winning_attr = None

        # Menu: 0=Humain, 1=Random, 2=Agent (AlphaZero)
        self.menu_choices = [0, 2]
        self.player_labels = ["Humain", "Random", "AlphaZero"]

    # ========== MENU ==========
    def draw_menu(self):
        self.screen.fill(BG_COLOR)
        title = self.font_big.render("QUARTO", True, BLACK)
        self.screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 40))

        subtitle = self.font.render("Selectionnez les joueurs", True, DARK_GRAY)
        self.screen.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 80))

        for p in range(2):
            y = 140 + p * 120
            label = self.font.render(f"Joueur {p + 1}:", True, BLACK)
            self.screen.blit(label, (WINDOW_W // 2 - 140, y))

            for i, name in enumerate(self.player_labels):
                bx = WINDOW_W // 2 - 140 + i * 100
                by = y + 30
                bw, bh = 90, 35
                rect = pygame.Rect(bx, by, bw, bh)
                if self.menu_choices[p] == i:
                    pygame.draw.rect(self.screen, GREEN, rect, border_radius=5)
                    tc = WHITE
                else:
                    pygame.draw.rect(self.screen, GRAY, rect, border_radius=5)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1, border_radius=5)
                    tc = BLACK
                txt = self.font_small.render(name, True, tc)
                self.screen.blit(txt, (bx + bw // 2 - txt.get_width() // 2, by + bh // 2 - txt.get_height() // 2))

        btn = pygame.Rect(WINDOW_W // 2 - 80, 400, 160, 45)
        pygame.draw.rect(self.screen, GREEN, btn, border_radius=8)
        txt = self.font_big.render("JOUER", True, WHITE)
        self.screen.blit(txt, (btn.centerx - txt.get_width() // 2, btn.centery - txt.get_height() // 2))

    def handle_menu_click(self, pos):
        for p in range(2):
            base_y = 140 + p * 120 + 30
            for i in range(len(self.player_labels)):
                bx = WINDOW_W // 2 - 140 + i * 100
                if pygame.Rect(bx, base_y, 90, 35).collidepoint(pos):
                    self.menu_choices[p] = i
                    return
        if pygame.Rect(WINDOW_W // 2 - 80, 400, 160, 45).collidepoint(pos):
            self.start_new_game()

    # ========== LANCEMENT ==========
    def start_new_game(self):
        self.gui_players = []
        self.env = Quatro()
        self.env.reset()
        self.current_player_index = 0

        for i in range(2):
            choice = self.menu_choices[i]
            self.player_types[i] = choice
            if choice == PLAYER_HUMAN:
                gp = GUIPlayer()
                self.players[i] = gp
                self.gui_players.append(gp)
            elif choice == PLAYER_AGENT:
                self.players[i] = get_alphazero_agent()
            else:
                self.players[i] = None  # Random

        self.winner = None
        self.game_finished = False
        self.winning_line = None
        self.winning_attr = None
        self.state = STATE_PLAYING

        self.game_thread = threading.Thread(target=self._run_game, daemon=True)
        self.game_thread.start()

    def _run_game(self):
        """Boucle de jeu utilisant env.step() directement."""
        env = self.env

        while not env.is_terminal():
            available = env.get_available_actions()
            if not available:
                break

            cp = env.current_player
            player = self.players[cp]
            ptype = self.player_types[cp]

            if ptype == PLAYER_HUMAN:
                # Bloque jusqu'au clic
                action = player.choose_flat_action(env)
            elif ptype == PLAYER_AGENT:
                # AlphaZero : MCTS + reseau
                time.sleep(0.3)  # delai visuel
                action = player.select_action(env)
            else:
                # Random
                time.sleep(0.3)
                action = random.choice(list(available))

            env.step(action)

        # Determiner le gagnant
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

        # Surbrillance cellule survolee (si on attend un placement)
        gui_waiting_place = self._get_waiting_action() == PLACE_PIECE
        if self.hover_cell is not None and gui_waiting_place:
            row, col = self.hover_cell
            idx = row * BOARD_SIZE + col
            if env.remaining_cells[idx]:
                r = pygame.Rect(col * CELL_SIZE + 1, row * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(self.screen, HIGHLIGHT, r)

        # Surligner les cellules gagnantes
        if self.winning_line is not None:
            surf = pygame.Surface((CELL_SIZE - 4, CELL_SIZE - 4), pygame.SRCALPHA)
            surf.fill(WIN_HIGHLIGHT)
            for idx in self.winning_line:
                r = idx // BOARD_SIZE
                c = idx % BOARD_SIZE
                self.screen.blit(surf, (c * CELL_SIZE + 2, r * CELL_SIZE + 2))

        # Pieces posees
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                idx = i * BOARD_SIZE + j
                piece = env.board[idx]
                if piece is not None:
                    cx = j * CELL_SIZE + CELL_SIZE // 2
                    cy = i * CELL_SIZE + CELL_SIZE // 2
                    draw_piece(self.screen, piece, cx, cy, CELL_SIZE)

        # Tracer la ligne gagnante par-dessus les pieces
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

        # Piece courante a poser
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
        cp = env.current_player
        player_type = self.player_labels[self.player_types[cp]]
        waiting = self._get_waiting_action()

        if env.current_piece is None:
            action_txt = "choisit une piece pour l'adversaire"
        else:
            action_txt = "place la piece sur le plateau"

        if waiting is not None:
            # Humain en attente → afficher le message d'action
            pass
        elif not self.game_finished:
            action_txt = "reflechit..."

        msg = f"Joueur {cp + 1} ({player_type}) {action_txt}"
        txt = self.font.render(msg, True, WHITE)
        self.screen.blit(txt, (10, BOARD_PX + 20))

        turn_num = 16 - sum(env.remaining_cells)
        t = self.font_small.render(f"Tour: {turn_num}/16", True, GRAY)
        self.screen.blit(t, (WINDOW_W - 80, BOARD_PX + 22))

    # ========== ECRAN FIN ==========
    def draw_game_over(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        if self.winner >= 0:
            pt = self.player_labels[self.player_types[self.winner]]
            msg = f"Joueur {self.winner + 1} ({pt}) a gagne!"
            color = GREEN
        else:
            msg = "Match nul!"
            color = YELLOW

        txt = self.font_big.render(msg, True, color)
        self.screen.blit(txt, (WINDOW_W // 2 - txt.get_width() // 2, WINDOW_H // 2 - 50))

        # Afficher l'attribut partage
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
        """Retrouve l'alignement gagnant et l'attribut partage."""
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
        """Retourne le type d'action attendu par un GUIPlayer, ou None."""
        for gp in self.gui_players:
            if gp.waiting_for is not None:
                return gp.waiting_for
        return None

    def _get_waiting_gui_player(self):
        """Retourne le GUIPlayer qui attend un choix, ou None."""
        for gp in self.gui_players:
            if gp.waiting_for is not None:
                return gp
        return None

    def handle_game_click(self, pos):
        """Transmet le clic au GUIPlayer qui attend, en action plate (0-31)."""
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
                gp.set_choice(NB_PIECES + idx)  # action plate 16-31

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
                        gp.set_choice(idx)  # action plate 0-15

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
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == STATE_MENU:
                        self.handle_menu_click(event.pos)
                    elif self.state == STATE_PLAYING:
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

            # Dessin
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
