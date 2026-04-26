import random


class RandomRolloutAgent:
    """
    Agent Random Rollout agnostique à l'environnement.

    Principe général :
    ------------------
    A chaque tour, l'agent :
    1. récupère toutes les actions possibles dans l'état courant
    2. teste chaque action plusieurs fois
    3. pour chaque test :
       - il fait une copie de l'environnement
       - il "déterminise" cette copie si nécessaire
       - il joue l'action à évaluer
       - il termine ensuite la partie de façon aléatoire
    4. il calcule la récompense moyenne obtenue pour chaque action
    5. il choisit l'action avec le meilleur score moyen

    Pourquoi "rollout" ?
    --------------------
    Parce qu'on déroule / simule la suite de la partie jusqu'à la fin.

    Pourquoi "random" ?
    -------------------
    Parce qu'après l'action testée, la suite est jouée aléatoirement.

    Pourquoi cet agent est agnostique ?
    -----------------------------------
    Parce qu'il ne dépend pas d'un jeu précis (GridWorld, Quarto, etc.).
    Il utilise seulement les méthodes communes définies dans l'interface
    de base des environnements.
    """

    def __init__(self, num_rollouts=10, max_rollout_steps=1000):
        """
        Constructeur de l'agent.

        Paramètres :
        ------------
        num_rollouts : int
            Nombre de simulations aléatoires effectuées pour évaluer
            UNE action possible.
            Exemple :
            si 4 actions sont disponibles et num_rollouts = 10,
            alors l'agent fera 40 simulations au total.

        max_rollout_steps : int
            Nombre maximal d'étapes autorisées pendant un rollout.
            Cela sert de sécurité pour éviter une boucle infinie si jamais
            l'environnement ne se termine pas correctement.
        """
        self.num_rollouts = num_rollouts
        self.max_rollout_steps = max_rollout_steps

    def act(self, env):
        """
        Méthode simple d'utilisation de l'agent.

        Rôle :
        ------
        Retourner directement l'action choisie par l'agent.

        Pourquoi cette méthode ?
        ------------------------
        Dans beaucoup de projets, on standardise les agents avec une méthode
        `act(env)` pour que tous aient la même interface d'appel.

        Ici, elle appelle simplement `choose_action(env)`.
        """
        return self.choose_action(env)

    def choose_action(self, env):
        """
        Méthode principale de décision.

        Rôle :
        ------
        Evaluer toutes les actions possibles dans l'état courant
        et retourner la meilleure.

        Fonctionnement détaillé :
        -------------------------
        1. On récupère les actions disponibles
        2. Pour chaque action :
           - on réalise plusieurs simulations
           - on calcule le score moyen obtenu par cette action
        3. On garde l'action avec le meilleur score moyen

        Paramètre :
        -----------
        env : BaseEnv
            Environnement courant.

        Retour :
        --------
        best_action :
            L'action jugée la meilleure.
            Retourne None si aucune action n'est disponible.
        """

        # On demande à l'environnement la liste des actions possibles
        # dans l'état courant.
        actions = env.get_available_actions()

        # S'il n'y a aucune action possible, l'agent ne peut rien jouer.
        if not actions:
            return None

        # Variables servant à mémoriser la meilleure action trouvée.
        best_action = None
        best_score = float("-inf")
        # float("-inf") = moins l'infini
        # On part volontairement d'une valeur très basse pour être sûr
        # que la première vraie action testée sera meilleure.

        # ------------------------------------------------------------
        # Boucle sur chaque action candidate
        # ------------------------------------------------------------
        for action in actions:
            # Cette variable accumule le score total obtenu par l'action
            # sur tous les rollouts effectués.
            total_score = 0.0

            # On répète plusieurs simulations pour estimer la qualité
            # de cette action.
            for _ in range(self.num_rollouts):

                # ----------------------------------------------------
                # ETAPE 1 : faire une copie de l'environnement
                # ----------------------------------------------------
                # Pourquoi ?
                # Parce qu'on ne veut surtout pas modifier le vrai état du jeu
                # pendant qu'on teste une action "dans notre tête".
                sim_env = env.clone()

                # ----------------------------------------------------
                # ETAPE 2 : déterminiser la copie
                # ----------------------------------------------------
                # Pourquoi ?
                # - Dans un env déterministe : cette méthode peut juste
                #   renvoyer la copie telle quelle.
                # - Dans un env non déterministe ou partiellement observable :
                #   elle complète les infos manquantes pour obtenir
                #   un état entièrement simulable.
                sim_env = sim_env.determinize()

                # ----------------------------------------------------
                # ETAPE 3 : jouer l'action qu'on veut évaluer
                # ----------------------------------------------------
                # La méthode step(action) applique l'action à l'environnement.
                # Elle retourne normalement :
                # - le prochain état
                # - la récompense obtenue
                #
                # Ici on ignore explicitement le prochain état avec "_"
                # car on n'en a pas besoin directement dans cette méthode :
                # l'environnement sim_env a déjà été modifié en interne.
                _, reward = sim_env.step(action)

                # On initialise la récompense totale du rollout avec la
                # récompense immédiate obtenue en jouant l'action testée.
                total_reward = reward

                # ----------------------------------------------------
                # ETAPE 4 : continuer la partie aléatoirement
                # ----------------------------------------------------
                # Une fois l'action candidate jouée, on simule la suite
                # jusqu'à la fin avec des choix aléatoires.
                total_reward += self._rollout(sim_env)

                # On ajoute le score obtenu pendant ce rollout au total.
                total_score += total_reward

            # --------------------------------------------------------
            # ETAPE 5 : calcul de la moyenne pour cette action
            # --------------------------------------------------------
            # On ne garde pas seulement un score d'un seul test,
            # car un seul test peut être trompeur.
            # On prend donc la moyenne sur plusieurs simulations.
            avg_score = total_score / self.num_rollouts

            # --------------------------------------------------------
            # ETAPE 6 : mise à jour de la meilleure action
            # --------------------------------------------------------
            if avg_score > best_score:
                best_score = avg_score
                best_action = action

        # A la fin, on renvoie l'action ayant obtenu la meilleure
        # récompense moyenne.
        return best_action

    def _rollout(self, env):
        """
        Simule aléatoirement la suite d'une partie jusqu'à sa fin.

        Rôle :
        ------
        A partir d'un environnement déjà modifié par l'action testée,
        on choisit ensuite des actions aléatoires jusqu'à la fin de l'épisode.

        Pourquoi cette méthode est séparée ?
        -----------------------------------
        Pour rendre le code plus clair :
        - choose_action() gère la comparaison entre les actions
        - _rollout() gère la simulation aléatoire de la suite

        Paramètre :
        -----------
        env : BaseEnv
            Une copie déjà simulée de l'environnement.

        Retour :
        --------
        total_reward : float
            Somme des récompenses obtenues pendant la simulation aléatoire.
        """

        # Somme des récompenses accumulées pendant le rollout
        total_reward = 0.0

        # Compteur d'étapes pour éviter une boucle infinie
        steps = 0

        # On continue tant que :
        # - l'épisode n'est pas terminé
        # - on n'a pas dépassé le nombre maximal d'étapes autorisées
        while not env.is_terminal() and steps < self.max_rollout_steps:

            # On récupère les actions encore possibles dans cet état.
            actions = env.get_available_actions()

            # S'il n'y a plus d'actions possibles, on arrête le rollout.
            if not actions:
                break

            # On choisit une action au hasard parmi les actions disponibles.
            action = random.choice(actions)

            # On applique cette action aléatoire.
            _, reward = env.step(action)

            # On ajoute la récompense obtenue à la somme totale.
            total_reward += reward

            # On incrémente le compteur de pas.
            steps += 1

        # On retourne le score total obtenu pendant la simulation.
        return total_reward