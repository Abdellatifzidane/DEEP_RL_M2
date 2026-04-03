L’objectif est de tester plusieurs types d’agents intelligents, les entraîner, puis comparer leurs performances.

Concrètement :

- choisir un jeu (environnement)

- implémenter plusieurs types d’agents RL

- les entraîner sur ce jeu

- mesurer leurs performances avec des métriques

- comparer les résultats

- présenter les résultats dans un rapport + une soutenance

*Le but pédagogique est de* :

comprendre comment fonctionnent différents algorithmes RL et voir dans quels cas ils sont efficaces ou non

**Algorithems (agents) a implementer**
- Random

- Q-Learning

- Deep Q-Learning

- Double Deep Q-Learning

- PPO / A2C

- Monte Carlo Tree Search

- AlphaZero

- MuZero

- etc

*Tester plusieurs de ces agents et comparer leur performances*
**Les métriques**
Pour chaque agent vous devrez mesurer par exemple :

- score moyen après 1000 parties

- score moyen après 10 000 parties

- score moyen après 100 000 parties

- score moyen après 1 000 000 parties (si possible)

- temps moyen pour jouer un coup

- longueur moyenne des parties

Ces métriques servent à voir quel agent apprend le mieux

**ETAPE INTERMEDIAIRE**
- Creer une interface graphique ( voir jouer les agents + permettre a un humain de jouer)
- Faire fonctionner le jeu avec un agent random ( mesurer le nombre de parties par seconde)
- Définir la representation du jeu ( vous devez proposer:
    - encoding de l’état du jeu : comment comment transformer le jeu en vecteur numérique
    - encoding des actions : comment représenter une action possible
)
*Livrable*
Vous devez rendre :

- le code

- une petite démo

- un document qui explique l’encoding des états et actions

---

## RLTracker — Logging unifié avec TensorBoard

`RLTracker` est le composant partagé pour logger les entraînements et évaluations de tous les agents. Il produit :
- Des logs **TensorBoard** (courbes en temps réel, comparaison entre agents)
- Un export **CSV** automatique (compatible avec `evaluate/results_plot.py`)

### Installation

```bash
pip install tensorboard
```

### Utilisation rapide

```python
from evaluate.tracker import RLTracker

# 1. Créer le tracker
tracker = RLTracker(
    agent_name="DQN",           # nom de l’agent
    env_name="GridWorld",       # nom de l’environnement
    config={                    # hyperparamètres (sauvegardés automatiquement)
        "lr": 1e-3,
        "gamma": 0.99,
        "hidden_dim": 128,
    },
    run_name="config_1",        # optionnel, auto-généré si omis
)

# 2. Logger chaque épisode pendant l’entraînement
for episode in range(num_episodes):
    # ... votre boucle d’entraînement ...
    tracker.log_episode(
        episode,
        score=score,            # score de fin de partie
        loss=loss,              # loss moyenne (agents deep uniquement)
        epsilon=agent.epsilon,  # taux d’exploration
        steps=steps,            # nombre de steps dans l’épisode
    )

# 3. Logger l’évaluation (aux paliers 1K, 10K, 100K, 1M)
avg_score = agent.evaluate(env, num_episodes=100)
tracker.log_evaluation(
    num_episodes_trained=10000,
    avg_score=avg_score,
    avg_steps=12.5,             # optionnel, pour jeux à durée variable
)

# 4. Logger le temps moyen par coup (en secondes)
tracker.log_move_time(0.003)

# 5. Finaliser (exporte le CSV + ferme TensorBoard)
tracker.finish()
```

### Utilisation avec context manager

```python
with RLTracker("TabularQL", "TicTacToe", config=my_config) as tracker:
    scores = agent.train(env, num_episodes=10000)
    for i, s in enumerate(scores):
        tracker.log_episode(i, score=s)
    tracker.log_evaluation(10000, avg_score=agent.evaluate(env))
# finish() est appelé automatiquement
```

### Exemple complet (copier-coller dans un test)

```python
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environnements.test_env.grid_world import GridWorld
from agents.deep_qlearning import DQNAgent
from evaluate.tracker import RLTracker

config = {
    "state_size": 2, "action_size": 4, "hidden_dim": 64,
    "lr": 5e-4, "gamma": 0.95, "epsilon": 1.0,
    "epsilon_min": 0.1, "epsilon_decay": 0.995,
}

env = GridWorld(size=5)
agent = DQNAgent(**config)
tracker = RLTracker("DQN", "GridWorld", config=config, run_name="exemple")

scores, losses = agent.train(env, num_episodes=1000)
for i, score in enumerate(scores):
    loss = losses[i] if i < len(losses) else None
    tracker.log_episode(i, score=score, loss=loss, epsilon=agent.epsilon)

avg_score = agent.evaluate(env, num_episodes=100)
tracker.log_evaluation(1000, avg_score=avg_score)
tracker.finish()
```

### Visualiser les résultats

```bash
tensorboard --logdir runs/
```

Puis ouvrir http://localhost:6006

### Structure des logs TensorBoard

```
runs/
├── GridWorld/
│   ├── DQN/
│   │   ├── config_1/
│   │   └── config_2/
│   ├── TabularQL/
│   │   └── config_1/
│   └── Random/
│       └── baseline/
├── TicTacToe/
│   └── REINFORCE/
│       └── config_1/
└── Quarto/
    └── AlphaZero/
        └── config_1/
```

### Comparer des agents dans TensorBoard

- **Tous les agents sur GridWorld** : dans le panneau de gauche, filtrer par `GridWorld/`
- **Toutes les configs de DQN** : filtrer par `GridWorld/DQN/`
- **DQN vs TabularQL** : sélectionner les deux runs dans le panneau

### Métriques loggées

| Tag TensorBoard | Description | Méthode |
|---|---|---|
| `train/score` | Score par épisode | `log_episode()` |
| `train/score_avg` | Moyenne glissante (100 ep.) | automatique |
| `train/loss` | Loss par épisode | `log_episode()` |
| `train/loss_avg` | Moyenne glissante loss | automatique |
| `train/epsilon` | Taux d’exploration | `log_episode()` |
| `train/steps` | Longueur de partie | `log_episode()` |
| `train/steps_avg` | Moyenne glissante steps | automatique |
| `eval/avg_score` | Score moyen d’évaluation | `log_evaluation()` |
| `eval/avg_steps` | Steps moyen d’évaluation | `log_evaluation()` |
| `perf/move_time_ms` | Temps par coup (ms) | `log_move_time()` |
| `perf/total_time_sec` | Temps total d’entraînement | `finish()` |

### Paramètres du constructeur

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `agent_name` | str | requis | Nom de l’agent (`"DQN"`, `"TabularQL"`, `"REINFORCE"`, ...) |
| `env_name` | str | requis | Nom de l’environnement (`"GridWorld"`, `"Quarto"`, ...) |
| `config` | dict | `None` | Hyperparamètres (sauvegardés dans TensorBoard + CSV) |
| `run_name` | str | `None` | Nom du run (auto-généré avec timestamp si omis) |
| `log_dir` | str | `"runs"` | Dossier racine des logs TensorBoard |
| `csv_dir` | str | `"results"` | Dossier d’export CSV |
| `smoothing_window` | int | `100` | Fenêtre pour les moyennes glissantes |