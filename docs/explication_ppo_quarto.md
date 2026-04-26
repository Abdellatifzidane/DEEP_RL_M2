# PPO-A2C sur Quarto : Explication du Code et de la Logique

## Table des matieres

1. [Vue d'ensemble](#vue-densemble)
2. [Rappel : le jeu Quarto](#rappel--le-jeu-quarto)
3. [Architecture du reseau : ActorCriticNetwork](#architecture-du-reseau--actorcriticnetwork)
4. [L'agent PPO_A2C](#lagent-ppo_a2c)
5. [La boucle d'entrainement (test_quarto_ppo.py)](#la-boucle-dentrainement)
6. [L'evaluation](#levaluation)
7. [Resume du flux complet](#resume-du-flux-complet)
---

## Vue d'ensemble

Cette implementation combine deux idees du reinforcement learning :

- **A2C (Advantage Actor-Critic)** : un reseau de neurones avec deux "tetes" — l'*Actor* qui decide quelle action jouer, et le *Critic* qui estime la valeur d'un etat.
- **PPO (Proximal Policy Optimization)** : un algorithme qui stabilise l'apprentissage en empechant la policy de changer trop brutalement d'une mise a jour a l'autre.

L'agent apprend a jouer au Quarto en s'entrainant contre un joueur aleatoire sur 100 000 parties.

---

## Rappel : le jeu Quarto

**Fichier** : `environnements/quarto/quatro.py`

Quarto est un jeu de plateau 4x4 avec 16 pieces, chacune ayant 4 attributs binaires (taille, couleur, forme, remplissage). Le but est d'aligner 4 pieces partageant au moins un attribut commun.

### Encodage de l'etat (STATE_SIZE = 69)

```
[0:64]  → 16 cases x 4 attributs = plateau aplati
[64:68] → 4 attributs de la piece courante a poser
[68]    → joueur courant (0 ou 1)
```

### Encodage des actions (ACTION_SIZE = 32)

```
[0-15]  → choisir une piece (pour l'adversaire)
[16-31] → poser la piece courante sur une case du plateau
```

Le jeu alterne entre deux phases : choisir une piece pour l'adversaire, puis l'adversaire pose cette piece. C'est ce qui rend Quarto unique — on choisit la piece que l'autre va jouer.

### Reward

- `+1.0` si la piece posee cree un alignement gagnant
- `0.0` sinon (y compris match nul)

---

## Architecture du reseau : ActorCriticNetwork

**Fichier** : `agents/ppo_a2c.py`, lignes 10-29

```
Etat (69) → [Linear 69→128] → ReLU → [Linear 128→128] → ReLU
                                                              ↓
                                                    ┌─────────┴─────────┐
                                                    ↓                   ↓
                                            Actor Head            Critic Head
                                         [Linear 128→32]       [Linear 128→1]
                                              ↓                      ↓
                                          Softmax                  Valeur V(s)
                                      (probabilites              (scalaire)
                                       sur 32 actions)
```

### Pourquoi deux tetes ?

- **Actor** (`actor_head`) : produit une distribution de probabilites sur les 32 actions possibles. C'est la **policy** π(a|s) — "quelle action jouer dans cet etat ?".
- **Critic** (`critic_head`) : estime la **valeur** V(s) — "a quel point cet etat est bon pour moi ?". Cette estimation sert a calculer l'*advantage* pendant l'apprentissage.

Le tronc commun (`shared_fc1`, `shared_fc2`) est partage : les deux tetes apprennent des representations communes du jeu, ce qui est plus efficace que deux reseaux separes.

---

## L'agent PPO_A2C

**Fichier** : `agents/ppo_a2c.py`, lignes 32-175

### Initialisation (lignes 33-63)

```python
self.policy      = ActorCriticNetwork(...)   # reseau qu'on optimise
self.policy_old  = ActorCriticNetwork(...)   # copie figee pour collecter les donnees
```

**Pourquoi deux reseaux ?** PPO compare les probabilites de la nouvelle policy avec celles de l'ancienne. `policy_old` est la version "figee" utilisee pendant la collecte des donnees. Apres chaque `update()`, on synchronise les deux.

La **memoire** stocke toutes les transitions collectees entre deux mises a jour :

| Champ         | Contenu                                            |
|---------------|----------------------------------------------------|
| `states`      | Etat observe avant chaque action                   |
| `actions`     | Action choisie                                     |
| `logprobs`    | log π_old(a|s) — log-probabilite sous l'ancienne policy |
| `masks`       | Masque des actions legales (pour Quarto)            |
| `rewards`     | Reward recu apres l'action                         |
| `is_terminals`| True si la partie est terminee apres cette action  |
| `values`      | V(s) estime par le critic au moment de la collecte |

### Hyperparametres cles

| Parametre     | Valeur | Role                                                       |
|---------------|--------|-------------------------------------------------------------|
| `lr`          | 3e-4   | Learning rate d'Adam                                        |
| `gamma`       | 0.99   | Facteur de discount — importance des rewards futurs         |
| `eps_clip`    | 0.2    | Limite du ratio PPO (empeche les mises a jour trop grandes) |
| `k_epochs`    | 4      | Nombre de passes d'optimisation sur le meme batch           |
| `batch_size`  | 64     | Non utilise directement ici (update sur tout le buffer)     |
| `hidden_dim`  | 128    | Taille des couches cachees                                  |

---

### Selection d'action : `select_action()` (lignes 65-92)

```
1. Convertir l'etat en tenseur
2. Passer dans policy_old → obtenir les probabilites et la valeur
3. Masquer les actions illegales (actions indisponibles dans Quarto)
4. Re-normaliser les probabilites
5. Echantillonner une action selon cette distribution
6. Stocker (state, action, logprob, mask, value) en memoire
7. Retourner l'action choisie
```

**Le masquage des actions** est crucial pour Quarto : a chaque instant, seules certaines actions sont valides (soit choisir une piece parmi les restantes, soit poser sur une case libre). On met a zero les probabilites des actions invalides et on re-normalise.

**Pourquoi echantillonner** au lieu de prendre l'action avec la plus haute probabilite ? C'est l'**exploration**. Pendant l'entrainement, l'agent doit essayer differentes actions pour decouvrir lesquelles sont bonnes. La distribution de probabilites guide naturellement l'exploration — les actions prometteuses sont plus souvent choisies, mais les autres ne sont pas ignorees.

---

### Stockage du reward : `store_reward()` (lignes 94-96)

Apres que l'environnement renvoie un reward, on l'associe a la derniere action de l'agent en memoire. On stocke aussi si la partie est terminee.

---

### Mise a jour PPO : `update()` (lignes 98-152)

C'est le coeur de l'algorithme. Voici le processus etape par etape :

#### Etape 1 : Calcul des rewards discountes (lignes 102-112)

```python
for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
    if is_terminal:
        discounted_reward = 0    # on repart de zero a la fin d'une partie
    discounted_reward = reward + gamma * discounted_reward
    rewards.insert(0, discounted_reward)
```

On parcourt les rewards **a l'envers** (du plus recent au plus ancien). Pour chaque etape :
- Si c'est un etat terminal, on remet le cumul a 0
- Sinon : `G_t = r_t + γ * G_{t+1}`

Cela donne le **retour disconte** G_t : la somme des rewards futurs, ponderes par γ. Un γ de 0.99 signifie que les rewards dans 100 etapes comptent encore pour ~37% de leur valeur.

Ensuite, on **normalise** les rewards (moyenne 0, ecart-type 1) pour stabiliser l'apprentissage.

#### Etape 2 : Calcul de l'avantage (ligne 120)

```python
advantages = rewards - old_values
```

L'**advantage** A(s,a) = G_t - V(s) mesure "a quel point cette action etait meilleure que ce qu'on attendait". Si A > 0, l'action etait meilleure que prevu → on veut augmenter sa probabilite. Si A < 0, elle etait pire → on veut la diminuer.

#### Etape 3 : Optimisation PPO sur k_epochs passes (lignes 122-148)

Pour chaque epoque, on recalcule les probabilites avec le reseau **actuel** (pas `policy_old`), puis :

**3a. Ratio des probabilites (ligne 134)**
```python
ratios = exp(log π_new(a|s) - log π_old(a|s)) = π_new(a|s) / π_old(a|s)
```

Ce ratio mesure combien la nouvelle policy differe de l'ancienne pour chaque action. Si ratio = 1, rien n'a change. Si ratio = 2, la nouvelle policy est deux fois plus susceptible de choisir cette action.

**3b. Clipping PPO (lignes 136-139)**
```python
surr1 = ratio * advantage                           # objectif non-clippe
surr2 = clamp(ratio, 1-ε, 1+ε) * advantage          # objectif clippe
actor_loss = -min(surr1, surr2)                      # on prend le pire des deux
```

C'est **l'idee centrale de PPO**. On limite le ratio entre `[1-ε, 1+ε]` = `[0.8, 1.2]` :

- Si l'advantage est **positif** (bonne action) et que le ratio depasse 1.2, on clip → on ne laisse pas la policy changer trop vite dans cette direction.
- Si l'advantage est **negatif** (mauvaise action) et que le ratio descend sous 0.8, on clip aussi.

Le `min()` garantit qu'on prend toujours l'estimation la plus **pessimiste**, ce qui empeche l'over-optimization. C'est ce qui rend PPO stable par rapport aux methodes precedentes (comme TRPO ou vanilla policy gradient).

**3c. Loss du Critic (ligne 141)**
```python
critic_loss = 0.5 * MSE(V_predicted, G_t)
```

On entraine le critic a predire les retours discountes. C'est une regression classique.

**3d. Bonus d'entropie (ligne 143)**
```python
loss = actor_loss + critic_loss - 0.01 * entropy
```

On **soustrait** l'entropie de la loss (donc on la maximise). L'entropie mesure le "desordre" de la distribution : une distribution uniforme a une entropie maximale. En encourageant l'entropie, on empeche l'agent de devenir trop deterministe trop tot — c'est une forme d'**exploration**.

Le coefficient 0.01 controle l'equilibre exploration/exploitation.

**3e. Gradient clipping (ligne 147)**
```python
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
```

On limite la norme des gradients a 0.5 pour eviter les mises a jour explosives.

#### Etape 4 : Synchronisation (ligne 150)

```python
self.policy_old.load_state_dict(self.policy.state_dict())
```

Apres les k_epochs d'optimisation, on copie les poids du reseau optimise dans `policy_old`. La prochaine phase de collecte utilisera cette nouvelle policy.

---

## La boucle d'entrainement

**Fichier** : `tests/test_quarto_ppo.py`

### Configuration (lignes 21-37)

```python
"num_episodes": 100_000      # nombre total de parties
"update_interval": 10        # mise a jour PPO tous les 10 episodes
"eval_games": 100            # 100 parties pour chaque evaluation
"eval_every": 1_000          # evaluer tous les 1000 episodes
```

Avec le test config, on utilise aussi :
- `hidden_dim: 256` (plus grand que le defaut de 128)
- `gamma: 0.95` (moins de poids sur le futur que le defaut 0.99)
- `eps_clip: 0.1` (clipping plus strict que le defaut 0.2)
- `k_epochs: 8` (plus de passes d'optimisation que le defaut 4)

### Deroulement d'un episode d'entrainement (lignes 116-156)

```
Pour chaque episode :
    1. Reset de l'environnement
    2. L'agent joue alternativement joueur 0 ou joueur 1 (episode % 2)
    3. Boucle de jeu :
       - Si c'est le tour de l'agent → select_action() + store_reward()
       - Si c'est le tour du random  → action aleatoire
       - Astuce : si le random gagne, on modifie le dernier reward de
         l'agent a -1.0 (ligne 139) pour lui apprendre qu'il a perdu
    4. Tous les 10 episodes → agent.update() (mise a jour PPO)
    5. Log du score dans le tracker
```

### L'astuce du reward adverse (lignes 136-139)

```python
if is_agent_turn:
    agent.store_reward(reward, env.is_terminal())
elif env.is_terminal() and len(agent.memory['rewards']) > 0:
    agent.memory['rewards'][-1] = -1.0
    agent.memory['is_terminals'][-1] = True
```

C'est un point subtil. Quand c'est l'adversaire (random) qui joue le coup gagnant :
- L'environnement donne `reward = +1.0` a l'adversaire, pas a l'agent
- On ecrase le dernier reward de l'agent par `-1.0` pour lui signaler qu'il a perdu

Sans cette astuce, l'agent ne recevrait que des rewards de 0 quand il perd, et ne saurait pas distinguer une defaite d'un match nul.

### Alternance des roles (ligne 119)

```python
agent_is_p0 = episode % 2 == 0
```

L'agent joue en premier dans les episodes pairs, en second dans les impairs. C'est essentiel pour qu'il apprenne a bien jouer dans les deux positions.

---

## L'evaluation

**Fichier** : `tests/test_quarto_ppo.py`, lignes 40-79

### `evaluate_vs_random()`

Tous les 1000 episodes, on evalue la policy **sans exploration** (greedy) :

```python
action = available_actions[np.argmax(probs)]   # on prend le MEILLEUR coup
```

Contrairement a l'entrainement ou on echantillonne, ici on prend **toujours l'action avec la plus haute probabilite**. On veut mesurer la performance reelle de ce que l'agent a appris.

Le score retourne est :
- `+1` si l'agent gagne
- `-1` si l'agent perd
- `0` si match nul

Le score moyen sur 100 parties donne une estimation de la force de l'agent. Un score de 0.0 = niveau random, un score de +1.0 = victoire systematique.

### Mesure de performance (lignes 170-186)

On mesure aussi le **temps par coup** du reseau :

```python
for _ in range(200):      # 200 parties
    while not terminal:
        t = time.time()
        # forward pass du reseau + argmax
        move_times.append(time.time() - t)
```

Cela verifie que l'inference est assez rapide pour une utilisation interactive (typiquement < 1ms par coup).

### Checkpoints (lignes 189-191)

A chaque evaluation, on sauvegarde le modele :
```python
agent.save(f"models/ppo_quarto_{episode+1}ep.pt")
```

Cela permet de reprendre l'entrainement plus tard (`--resume`) ou d'utiliser le meilleur modele.

### Tracking (RLTracker)

Le tracker log tout dans TensorBoard (`runs/`) et exporte un CSV (`results/`). On peut visualiser :
- `train/score` : score brut par episode
- `train/score_avg` : moyenne glissante sur 100 episodes
- `eval/avg_score` : score moyen aux paliers d'evaluation
- `perf/move_time_ms` : temps par coup en millisecondes

---

## Resume du flux complet

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOUCLE D'ENTRAINEMENT                        │
│                                                                 │
│  Pour chaque episode (0 a 100 000) :                            │
│                                                                 │
│    1. Reset du jeu                                              │
│    2. Alterner agent/random comme J1/J2                         │
│    3. Jouer la partie :                                         │
│       ┌──────────────────────────────────────────┐              │
│       │ Tour de l'agent :                        │              │
│       │   state → policy_old → probs → masquage  │              │
│       │   → echantillonnage → action             │              │
│       │   → stocker (s, a, logp, mask, v)        │              │
│       │   → stocker reward                       │              │
│       ├──────────────────────────────────────────┤              │
│       │ Tour du random :                         │              │
│       │   action aleatoire                       │              │
│       │   Si random gagne → reward agent = -1    │              │
│       └──────────────────────────────────────────┘              │
│                                                                 │
│    4. Tous les 10 episodes → UPDATE PPO :                       │
│       ┌──────────────────────────────────────────┐              │
│       │ a) Calculer G_t (rewards discountes)     │              │
│       │ b) Normaliser les rewards                │              │
│       │ c) Advantage = G_t - V(s)                │              │
│       │ d) Pour k=8 epoques :                    │              │
│       │    - Recalculer probs avec policy actuel  │              │
│       │    - ratio = π_new / π_old               │              │
│       │    - Clipper le ratio dans [0.9, 1.1]    │              │
│       │    - Loss = actor + critic - entropie    │              │
│       │    - Backprop + gradient clipping         │              │
│       │ e) Copier policy → policy_old            │              │
│       │ f) Vider la memoire                      │              │
│       └──────────────────────────────────────────┘              │
│                                                                 │
│    5. Tous les 1000 episodes → EVALUATION :                     │
│       ┌──────────────────────────────────────────┐              │
│       │ 100 parties greedy vs random             │              │
│       │ → score moyen, longueur moyenne          │              │
│       │ → mesure du temps par coup               │              │
│       │ → sauvegarde checkpoint                  │              │
│       │ → log TensorBoard + CSV                  │              │
│       └──────────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pourquoi PPO fonctionne bien ici

1. **Stabilite** : le clipping empeche la policy de diverger, meme avec un jeu complexe comme Quarto.
2. **Efficacite des donnees** : les k_epochs permettent de reutiliser les memes donnees plusieurs fois au lieu de les jeter apres une seule passe.
3. **Action masking** : le masquage des actions illegales evite a l'agent de gaspiller de la capacite a apprendre qu'il ne peut pas poser sur une case deja occupee.
4. **Advantage** : en soustrayant V(s), on reduit la variance des gradients — l'agent apprend plus vite ce qui est *relativement* bon, pas juste ce qui donne un reward positif.
