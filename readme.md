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