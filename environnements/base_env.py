from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Interface minimale pour tous les environnements RL discrets.
    """

    @abstractmethod
    def reset(self):
        """
        Réinitialise l'environnement au début d'un épisode.
        Retourne l'état initial (optionnel mais recommandé).
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Retourne l'état courant.
        - Tabulaire : hashable (tuple, int, str)
        - Deep RL : vecteur numérique (list, np.array)
        """
        pass

    @abstractmethod
    def get_available_actions(self):
        """
        Retourne la liste des actions possibles (indices ou valeurs discrètes).
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Applique une action.

        Retour:
            next_state, reward
        """
        pass

    @abstractmethod
    def is_terminal(self):
        """
        Retourne True si l'épisode est terminé.
        """
        pass

    def get_score(self):
        """
        Optionnel : score final de l'épisode.
        """
        return None

    @abstractmethod
    def action_to_index(self, action):
        pass