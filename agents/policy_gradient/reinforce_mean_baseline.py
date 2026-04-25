import torch
from agents.policy_gradient.base import BasePolicyGradientAgent


class ReinforceMeanBaselineAgent(BasePolicyGradientAgent):
    """
    Implémentation de REINFORCE avec mean baseline.

    Idée :
    ------
    Au lieu d'utiliser directement le return G_t,
    on lui soustrait une baseline égale à la moyenne des returns
    de l'épisode.

    Formule :
    ---------
    advantage_t = G_t - baseline
    loss = - somme( log_prob(action_t) * advantage_t )

    Intuition :
    -----------
    - si une action fait mieux que la moyenne, son avantage est positif
      → on augmente sa probabilité
    - si elle fait moins bien que la moyenne, son avantage est négatif
      → on diminue sa probabilité

    Cette baseline permet souvent de réduire la variance
    et de stabiliser l'apprentissage.
    """

    def mettre_a_jour_policy(self, liste_etats, liste_log_prob_actions, liste_rewards):
        """
        Met à jour la policy à la fin de l'épisode
        avec une baseline = moyenne des returns.

        Paramètres :
        ------------
        liste_etats : list
            Liste des états rencontrés pendant l'épisode.
            (Pas directement utilisée ici, mais gardée pour garder
            la même signature que les autres variantes.)
        liste_log_prob_actions : list
            Log-probabilités des actions choisies à chaque étape.
        liste_rewards : list
            Récompenses obtenues pendant l'épisode.

        Retour :
        --------
        float
            Valeur numérique de la loss totale.
        """

        # Sécurité : si aucune action n'a été jouée,
        # on ne fait aucune mise à jour
        if len(liste_log_prob_actions) == 0:
            return 0.0

        # ------------------------------------------------------------
        # ETAPE 1 : calculer les returns G_t
        # ------------------------------------------------------------
        liste_returns = self.calculer_returns(liste_rewards)

        # Conversion en tensor PyTorch
        tensor_returns = torch.tensor(liste_returns, dtype=torch.float32)

        # ------------------------------------------------------------
        # ETAPE 2 : calculer la baseline
        # ------------------------------------------------------------
        # Ici la baseline est simplement la moyenne des returns
        baseline = tensor_returns.mean()

        # ------------------------------------------------------------
        # ETAPE 3 : calculer les avantages
        # ------------------------------------------------------------
        # advantage_t = G_t - baseline
        avantages = tensor_returns - baseline

        # ------------------------------------------------------------
        # ETAPE 4 : calculer la loss
        # ------------------------------------------------------------
        # Pour chaque étape :
        # loss_t = - log_prob(action_t) * advantage_t
        liste_losses = [
            -log_prob * avantage
            for log_prob, avantage in zip(liste_log_prob_actions, avantages)
        ]

        # On additionne toutes les contributions de l'épisode
        loss_totale = torch.stack(liste_losses).sum()

        # ------------------------------------------------------------
        # ETAPE 5 : mettre à jour le réseau de policy
        # ------------------------------------------------------------
        self.optimizer.zero_grad()   # remet les gradients à zéro
        loss_totale.backward()       # calcule les gradients
        self.optimizer.step()        # met à jour les poids

        return loss_totale.item()