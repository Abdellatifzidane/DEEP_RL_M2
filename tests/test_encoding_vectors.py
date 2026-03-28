import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from environnements.quarto.quatro import Quatro, PIECES


def test_quarto_encoding():
    print("=== QUARTO — State Encoding ===")
    env = Quatro()
    state = env.reset()

    print(f"State : tuple de {len(state)} valeurs")
    print(f"  [0:64]  = 16 cases x 4 attributs (plateau)")
    print(f"  [64:68] = piece courante (4 attributs)")
    print(f"  Attributs : taille, couleur, forme, remplissage  (-1 ou +1)")
    print(f"  Case vide = [0,0,0,0]")
    print(f"DQN : {np.array(state, dtype=np.float32)}")

    print(f"\nLes 16 pieces (taille, couleur, forme, remplissage) :")
    for i, p in enumerate(PIECES):
        print(f"  {i:2d}: {p}")

    print("\n=== QUARTO — Action Encoding ===")
    print("Action = (type, index), vecteur de taille 2")
    print("  (0, i) = choisir piece i pour l'adversaire")
    print("  (1, j) = poser la piece courante en case j")

    # choisir une piece
    actions = env.get_available_actions()
    print(f"\nAvant pose — actions dispo : {actions}")
    env.step((0, 3))  # choisir piece 3

    # poser la piece
    actions = env.get_available_actions()
    print(f"Apres choix piece 3 — actions dispo : {actions}")

    state2 = env.get_state()
    print(f"  piece courante [64:68] = {state2[64:68]}")


if __name__ == "__main__":
    test_quarto_encoding()
