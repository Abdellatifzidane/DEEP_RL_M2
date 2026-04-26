# Deep RL sur Quarto — Guide de reproductibilite

## Structure du projet

```
DEEP_RL_M2/
├── agents/                  # Algorithmes
│   ├── ppo_a2c.py           # PPO + Advantage Actor-Critic
│   ├── mcts.py              # Monte Carlo Tree Search (pur)
│   ├── expert_apprentice.py # Expert Iteration (ExIt)
│   └── alpha_zero.py        # AlphaZero (MCTS + reseau)
├── environnements/
│   └── quarto/quatro.py     # Environnement Quarto (state 69, actions 32)
├── evaluate/
│   └── tracker.py           # RLTracker (TensorBoard + CSV)
├── gui/
│   └── quarto_gui.py        # Interface graphique (demo interactive)
├── tests/                   # Scripts d'entrainement et d'evaluation
│   ├── test_quarto_ppo.py
│   ├── test_quarto_mcts.py
│   ├── test_quarto_exit.py
│   ├── test_quarto_alphazero.py
│   └── test_quarto_tournament.py
├── models/                  # Checkpoints sauvegardes (.pt)
├── results/                 # CSV des evaluations
└── runs/                    # Logs TensorBoard
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install pygame              # pour la GUI
```

## Lancer les entrainements

Chaque agent a son propre script de test. Les entrainements produisent des checkpoints dans `models/`, des CSV dans `results/` et des logs TensorBoard dans `runs/`.

```bash
# PPO-A2C (100k episodes, ~20 min)
python tests/test_quarto_ppo.py

# MCTS (pas d'entrainement, evaluation directe, ~5 min)
python tests/test_quarto_mcts.py

# Expert Iteration (100k parties MCTS, ~2h30)
python tests/test_quarto_exit.py

# AlphaZero (100k parties self-play, ~7h)
python tests/test_quarto_alphazero.py
```

Pour reprendre un entrainement depuis un checkpoint :

```bash
python tests/test_quarto_ppo.py --resume models/ppo_quarto_50000ep.pt
python tests/test_quarto_exit.py --resume models/exit_quarto_50000g.pt
python tests/test_quarto_alphazero.py --resume models/az_quarto_50000g.pt
```

## Lancer le tournoi entre agents

Compare tous les agents entre eux sur 200 parties par confrontation :

```bash
python tests/test_quarto_tournament.py
```

## Lancer la demo (interface graphique)

```bash
python gui/quarto_gui.py
```

L'interface permet de faire jouer n'importe quel agent contre n'importe quel autre (ou contre un humain). Agents disponibles dans le menu :

- **Humain** — clics souris
- **Random** — joueur aleatoire
- **AlphaZero** — MCTS guide par reseau (200 sims)
- **PPO 100k** — policy gradient (Actor-Critic)
- **MCTS** — recherche arborescente pure (500 sims)
- **ExIt 100k** — reseau apprenti (imitation MCTS)
- **REINFORCE / R. Mean / R. Critic** — policy gradient basique (3 variantes)

Quand deux agents jouent entre eux, le jeu avance en **mode pas-a-pas** : chaque coup est affiche dans la barre de statut et il faut cliquer **Suivant** ou appuyer sur **Espace** pour avancer.

## Visualiser les courbes d'entrainement

```bash
tensorboard --logdir runs/
```

Puis ouvrir http://localhost:6006 dans le navigateur.

## Resultats precalcules

Les CSV dans `results/` contiennent les evaluations aux paliers (1k, 10k, 50k, 100k episodes). Le fichier `results/quarto_tournament_results.csv` contient les resultats du tournoi entre agents.
