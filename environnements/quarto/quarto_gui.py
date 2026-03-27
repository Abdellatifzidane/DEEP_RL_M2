"""
Quarto GUI - Interface graphique simple pour jouer au Quarto.

Modes supportes :
    Random vs Random, Random vs Human, Human vs Human,
    Human vs Agent, Agent vs Agent, etc.

Utilise l'environnement Quarto existant comme moteur de jeu.

Usage :
    python quarto_gui.py

Pour brancher un agent entraine, modifier la methode _load_agent().
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from environnements.quarto.quatro import Quatro, PIECES, NB_PIECES, NB_CELLS
from agents.random import RandomAgent


# =====================================================================
#  CONSTANTES VISUELLES
# =====================================================================
CELL_SIZE = 100          # taille d'une case du plateau (px)
PIECE_PANEL_SIZE = 65    # taille d'une case du panneau de pieces (px)
AI_DELAY_MS = 400        # delai entre coups IA (ms)

# Couleurs des pieces (attribut "couleur" : -1 = bleu, +1 = rouge)
PIECE_COLORS = {-1: "#2980b9", 1: "#c0392b"}


# =====================================================================
#  DESSIN D'UNE PIECE
# =====================================================================
# Chaque piece a 4 attributs binaires (-1 ou +1) :
#   0: taille   -> petit (-1) / grand (+1)
#   1: couleur  -> bleu  (-1) / rouge (+1)
#   2: forme    -> rond  (-1) / carre (+1)
#   3: rempli   -> creux (-1) / plein (+1)

def draw_piece(canvas, cx, cy, piece, max_radius=20):
    """Dessine une piece sur le canvas au centre (cx, cy)."""
    attr_size, attr_color, attr_shape, attr_fill = piece

    # --- taille ---
    r = max_radius * 0.95 if attr_size == 1 else max_radius * 0.55

    # --- couleur ---
    color = PIECE_COLORS[attr_color]

    # --- remplissage ---
    fill = color if attr_fill == 1 else ""
    outline = color
    width = 3

    # --- forme ---
    if attr_shape == -1:   # rond
        canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=fill, outline=outline, width=width,
        )
    else:                  # carre
        canvas.create_rectangle(
            cx - r, cy - r, cx + r, cy + r,
            fill=fill, outline=outline, width=width,
        )


# =====================================================================
#  CLASSE PRINCIPALE
# =====================================================================
class QuartoGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Quarto")
        self.root.resizable(False, False)

        self.env = Quatro()
        self.game_running = False

        # agents[i] = None  -> joueur humain
        # agents[i] = objet -> IA (doit avoir choose_action ou select_action)
        self.agents = [None, None]

        # Phase 2 humaine : on selectionne d'abord la case, puis la piece
        self.selected_cell = None

        self._build_ui()

    # -----------------------------------------------------------------
    #  CONSTRUCTION DE L'INTERFACE
    # -----------------------------------------------------------------
    def _build_ui(self):
        # ---- barre de controle ----
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Joueur 1 :").grid(row=0, column=0, padx=(0, 4))
        self.p1_var = ttk.Combobox(
            ctrl, values=["Random", "Human", "Agent"],
            state="readonly", width=8,
        )
        self.p1_var.set("Random")
        self.p1_var.grid(row=0, column=1, padx=(0, 15))

        ttk.Label(ctrl, text="Joueur 2 :").grid(row=0, column=2, padx=(0, 4))
        self.p2_var = ttk.Combobox(
            ctrl, values=["Random", "Human", "Agent"],
            state="readonly", width=8,
        )
        self.p2_var.set("Human")
        self.p2_var.grid(row=0, column=3, padx=(0, 15))

        self.start_btn = ttk.Button(ctrl, text="Demarrer", command=self.start_game)
        self.start_btn.grid(row=0, column=4, padx=5)

        self.reset_btn = ttk.Button(ctrl, text="Reset", command=self.reset_game)
        self.reset_btn.grid(row=0, column=5, padx=5)

        # ---- zone de jeu (plateau + pieces) ----
        middle = ttk.Frame(self.root, padding=10)
        middle.pack()

        # plateau 4x4
        board_frame = ttk.LabelFrame(middle, text="Plateau (4x4)", padding=5)
        board_frame.grid(row=0, column=0, padx=(0, 15))

        bw = CELL_SIZE * 4 + 10
        self.board_canvas = tk.Canvas(board_frame, width=bw, height=bw, bg="#f5f0e1")
        self.board_canvas.pack()
        self.board_canvas.bind("<Button-1>", self._on_board_click)

        # panneau des pieces disponibles
        right = ttk.Frame(middle)
        right.grid(row=0, column=1, sticky="n")

        pieces_frame = ttk.LabelFrame(right, text="Pieces disponibles", padding=5)
        pieces_frame.pack()

        pw = PIECE_PANEL_SIZE * 4 + 10
        self.pieces_canvas = tk.Canvas(pieces_frame, width=pw, height=pw, bg="#fafafa")
        self.pieces_canvas.pack()
        self.pieces_canvas.bind("<Button-1>", self._on_piece_click)

        # piece courante a poser
        curr_frame = ttk.LabelFrame(right, text="Piece a poser", padding=5)
        curr_frame.pack(pady=(10, 0), fill="x")
        self.curr_canvas = tk.Canvas(curr_frame, width=pw, height=70, bg="#fafafa")
        self.curr_canvas.pack()

        # legende
        legend_frame = ttk.LabelFrame(right, text="Legende", padding=5)
        legend_frame.pack(pady=(10, 0), fill="x")
        legend_text = (
            "Taille : petit / grand\n"
            "Couleur : bleu / rouge\n"
            "Forme : rond / carre\n"
            "Rempli : creux / plein"
        )
        ttk.Label(legend_frame, text=legend_text, font=("Arial", 9)).pack()

        # ---- barre de statut ----
        self.status_var = tk.StringVar(value="Cliquez 'Demarrer' pour jouer.")
        ttk.Label(
            self.root, textvariable=self.status_var,
            font=("Arial", 12, "bold"), padding=10, anchor="center",
        ).pack(fill="x")

        self._refresh()

    # -----------------------------------------------------------------
    #  DESSIN
    # -----------------------------------------------------------------
    def _draw_board(self):
        c = self.board_canvas
        c.delete("all")
        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                x0 = col * CELL_SIZE + 5
                y0 = row * CELL_SIZE + 5
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                # fond vert clair si la case est selectionnee par le joueur
                bg = "#b8e6b8" if self.selected_cell == idx else "#f5f0e1"
                c.create_rectangle(x0, y0, x1, y1, fill=bg, outline="#333", width=2)

                # dessiner la piece posee
                piece = self.env.board[idx]
                if piece is not None:
                    draw_piece(c, x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2,
                               piece, max_radius=CELL_SIZE * 0.38)

                # numero de la case (discret)
                c.create_text(x0 + 12, y0 + 12, text=str(idx),
                              fill="#aaa", font=("Arial", 8))

    def _draw_pieces(self):
        c = self.pieces_canvas
        c.delete("all")
        s = PIECE_PANEL_SIZE
        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                x0 = col * s + 5
                y0 = row * s + 5
                x1 = x0 + s
                y1 = y0 + s

                available = self.env.ramaining_pieces[idx] == 1
                bg = "#fafafa" if available else "#d5d5d5"
                c.create_rectangle(x0, y0, x1, y1, fill=bg, outline="#bbb")

                if available:
                    draw_piece(c, x0 + s // 2, y0 + s // 2,
                               PIECES[idx], max_radius=s * 0.35)

                c.create_text(x0 + 9, y0 + 9, text=str(idx),
                              fill="#bbb", font=("Arial", 7))

    def _draw_current_piece(self):
        c = self.curr_canvas
        c.delete("all")
        if self.env.current_piece is not None:
            w = int(c["width"])
            draw_piece(c, w // 2, 35, self.env.current_piece, max_radius=25)

    def _refresh(self):
        self._draw_board()
        self._draw_pieces()
        self._draw_current_piece()

    # -----------------------------------------------------------------
    #  CONTROLE DU JEU
    # -----------------------------------------------------------------
    def start_game(self):
        self.env.reset()
        self.selected_cell = None
        self.game_running = True

        # creer les agents selon la selection
        for i, combo in enumerate([self.p1_var, self.p2_var]):
            ptype = combo.get()
            if ptype == "Random":
                self.agents[i] = RandomAgent()
            elif ptype == "Human":
                self.agents[i] = None
            elif ptype == "Agent":
                self.agents[i] = self._load_agent(i)

        # verrouiller les controles
        self.start_btn.config(state="disabled")
        self.p1_var.config(state="disabled")
        self.p2_var.config(state="disabled")

        self._refresh()
        self._next_turn()

    def reset_game(self):
        self.env.reset()
        self.selected_cell = None
        self.game_running = False
        self.agents = [None, None]

        self.start_btn.config(state="normal")
        self.p1_var.config(state="readonly")
        self.p2_var.config(state="readonly")
        self.status_var.set("Cliquez 'Demarrer' pour jouer.")
        self._refresh()

    def _load_agent(self, player_index):
        """
        Charge un agent entraine pour le joueur donne.

        ---------------------------------------------------------------
        PAR DEFAUT : utilise RandomAgent (placeholder).
        POUR UTILISER VOTRE AGENT ENTRAINE, remplacez le contenu de
        cette methode. Exemple :

            import pickle
            with open("mon_agent.pkl", "rb") as f:
                agent = pickle.load(f)
            return agent

        L'agent doit avoir :
          - choose_action(env)  -> action
          OU
          - select_action(state, actions, training=False) -> action
        ---------------------------------------------------------------
        """
        return RandomAgent()

    # -----------------------------------------------------------------
    #  BOUCLE DE JEU
    # -----------------------------------------------------------------
    def _next_turn(self):
        """Determine si c'est un tour IA ou humain et agit."""
        if not self.game_running or self.env.is_terminal():
            return

        p = self.env.current_player
        agent = self.agents[p]

        if agent is not None:
            # ---- tour IA ----
            phase = "choisir piece" if self.env.current_piece is None else "poser"
            self.status_var.set(f"Joueur {p + 1} (IA) joue...  [{phase}]")
            self.root.update()
            self.root.after(AI_DELAY_MS, lambda: self._ai_play(agent))
        else:
            # ---- tour humain ----
            self._update_human_status()

    def _update_human_status(self):
        p = self.env.current_player + 1
        if self.env.current_piece is None:
            self.status_var.set(
                f"Joueur {p} : choisissez une piece a donner a l'adversaire.")
        elif self.selected_cell is None:
            self.status_var.set(
                f"Joueur {p} : placez votre piece sur le plateau.")
        else:
            self.status_var.set(
                f"Joueur {p} : choisissez une piece pour l'adversaire.")

    def _ai_play(self, agent):
        if not self.game_running:
            return
        action = self._get_agent_action(agent)
        if action is None:
            return
        _, reward = self.env.step(action)
        self.selected_cell = None
        self._refresh()

        if self.env.is_terminal():
            self._game_over(reward)
        else:
            self._next_turn()

    def _get_agent_action(self, agent):
        """Compatible avec RandomAgent, TabularQ et DQN."""
        if hasattr(agent, "choose_action"):
            return agent.choose_action(self.env)
        if hasattr(agent, "select_action"):
            state = self.env.get_state()
            actions = self.env.get_available_actions()
            return agent.select_action(state, actions, training=False)
        return None

    # -----------------------------------------------------------------
    #  INTERACTIONS HUMAINES
    # -----------------------------------------------------------------
    def _on_board_click(self, event):
        """Clic sur le plateau : le joueur humain pose sa piece."""
        if not self.game_running or self.env.is_terminal():
            return
        if self.agents[self.env.current_player] is not None:
            return   # pas le tour d'un humain
        if self.env.current_piece is None:
            self.status_var.set("Choisissez d'abord une piece (panneau de droite).")
            return

        col = (event.x - 5) // CELL_SIZE
        row = (event.y - 5) // CELL_SIZE
        if not (0 <= row < 4 and 0 <= col < 4):
            return
        cell = row * 4 + col

        if self.env.remaining_cells[cell] != 1:
            self.status_var.set("Case occupee ! Choisissez une case libre.")
            return

        avail_pieces = [i for i, v in enumerate(self.env.ramaining_pieces) if v == 1]

        if not avail_pieces:
            # derniere piece a poser, pas de choix supplementaire
            _, reward = self.env.step(cell)
            self.selected_cell = None
            self._refresh()
            if self.env.is_terminal():
                self._game_over(reward)
            else:
                self._next_turn()
        else:
            # stocker la case, attendre le choix de piece
            self.selected_cell = cell
            self._refresh()
            self._update_human_status()

    def _on_piece_click(self, event):
        """Clic sur le panneau de pieces : le joueur choisit une piece."""
        if not self.game_running or self.env.is_terminal():
            return
        if self.agents[self.env.current_player] is not None:
            return

        col = (event.x - 5) // PIECE_PANEL_SIZE
        row = (event.y - 5) // PIECE_PANEL_SIZE
        if not (0 <= row < 4 and 0 <= col < 4):
            return
        piece_idx = row * 4 + col

        if self.env.ramaining_pieces[piece_idx] != 1:
            self.status_var.set("Piece indisponible !")
            return

        # ---- Phase 1 : juste choisir une piece (pas de piece a poser) ----
        if self.env.current_piece is None:
            _, reward = self.env.step(piece_idx)
            self.selected_cell = None
            self._refresh()
            if self.env.is_terminal():
                self._game_over(reward)
            else:
                self._next_turn()
            return

        # ---- Phase 2 : il faut d'abord avoir selectionne une case ----
        if self.selected_cell is None:
            self.status_var.set("Placez d'abord votre piece sur le plateau !")
            return

        # combiner case + piece en une seule action
        action = self.selected_cell * NB_PIECES + piece_idx
        _, reward = self.env.step(action)
        self.selected_cell = None
        self._refresh()

        if self.env.is_terminal():
            self._game_over(reward)
        else:
            self._next_turn()

    # -----------------------------------------------------------------
    #  FIN DE PARTIE
    # -----------------------------------------------------------------
    def _game_over(self, reward):
        self.game_running = False
        self.start_btn.config(state="normal")
        self.p1_var.config(state="readonly")
        self.p2_var.config(state="readonly")

        if reward == 1.0:
            # le gagnant est le joueur qui vient de poser
            # (current_player n'a PAS bascule lors d'une victoire)
            winner = self.env.current_player + 1
            self.status_var.set(f"Joueur {winner} a gagne !")
        else:
            self.status_var.set("Match nul !")


# =====================================================================
#  POINT D'ENTREE
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    QuartoGUI(root)
    root.mainloop()
