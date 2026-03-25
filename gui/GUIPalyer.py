
import threading
from players import Player, PlayerType


class GUIPlayer(Player):
    """Joueur humain via l'interface graphique.
    choose_action() bloque jusqu'a ce que la GUI fournisse le choix via set_choice().
    """
    def __init__(self):
        super().__init__(PlayerType.HUMAN)
        self._event = threading.Event()
        self._choice = None
        self.waiting_for = None  # (action_type) quand on attend un clic

    def choose_action(self, action_type, env):
        if action_type == 0:
            available = [i for i, a in enumerate(env.ramaining_pieces) if a]
        else:
            available = [i for i, a in enumerate(env.remaining_cells) if a]

        if not available:
            return None

        # Signaler a la GUI qu'on attend un choix
        self._event.clear()
        self._choice = None
        self.waiting_for = action_type

        # Bloquer jusqu'a ce que la GUI appelle set_choice()
        self._event.wait()
        self.waiting_for = None
        return self._choice

    def set_choice(self, index):
        """Appelee par la GUI quand l'utilisateur clique."""
        self._choice = index
        self._event.set()
