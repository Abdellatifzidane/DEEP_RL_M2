import sys
import os

# Ajuster le path pour importer l'environnement local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from environnements.test_env.tic_tac_toe import TicTacToe


def print_board(state):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    for r in range(3):
        row = [symbols[state[3*r + c]] for c in range(3)]
        print(' ' + ' | '.join(row))
        if r < 2:
            print('-----------')


def main():
    print("=== TicTacToe Humain vs Random ===")
    env = TicTacToe(seed=42)
    state = env.reset()

    while not env.is_terminal():
        print('\nBoard actuel:')
        print_board(state)
        print('Actions disponibles:', env.get_available_actions())

        try:
            action = int(input('Choisissez une action (0-8, position sur la grille) : ').strip())
        except ValueError:
            print('Entrée invalide. Veuillez choisir un entier entre 0 et 8.')
            continue

        if action not in env.get_available_actions():
            print('Action impossible. Choisissez une case vide valide.')
            continue

        state, reward = env.step(action)
        print(f'Vous avez joué en {action}, reward: {reward}')

        if env.is_terminal():
            break

        print('\nAprès l"action adversaire (random) :')
        print_board(state)

    print('\n=== Partie Terminée ===')
    print_board(env.get_state())
    score = env.get_score()
    if score == 1:
        print('Résultat: Vous avez gagné !')
    elif score == -1:
        print('Résultat: Vous avez perdu (adversaire O a gagné).')
    else:
        print('Résultat: Match nul.')


if __name__ == '__main__':
    main()
