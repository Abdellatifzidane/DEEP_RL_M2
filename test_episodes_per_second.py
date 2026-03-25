
from game import game
from quatro import Quatro

from players import RandomPlayer, HumanPlayer, AgentPlayer
import time

def test_episodes_per_second():
    start_time = time.time()
    nb_episodes = 0
    time_limit = 2.0
    start_time = time.time()

    while time.time() - start_time < time_limit:
        env = Quatro()
        player_01 = RandomPlayer()
        player_02 = RandomPlayer()
        game_test = game(env, player_01, player_02)
        game_test.start_game_without_print()
        nb_episodes += 1

    print("Nombre d'épisodes joués en 1 seconde: {}".format(nb_episodes))


test_episodes_per_second()