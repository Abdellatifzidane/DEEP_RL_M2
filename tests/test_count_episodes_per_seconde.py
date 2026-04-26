"""
Benchmark : combien d'épisodes random vs random en 1 seconde sur Quarto.
"""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.quarto.quatro import Quatro
from agents.random import RandomAgent


player1 = RandomAgent()
player2 = RandomAgent()
players = [player1, player2]

count = 0
start = time.time()
env = Quatro()

while time.time() - start < 1.0:
    env.reset()

    while not env.is_terminal():
        current = players[env.current_player]
        action = current.choose_action(env)
        env.step(action)

    count += 1 

elapsed = time.time() - start
print(f"{count} épisodes en {elapsed:.2f}s ({count / elapsed:.0f} épisodes/s)")
