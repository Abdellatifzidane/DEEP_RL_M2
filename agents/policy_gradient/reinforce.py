import torch
from agents.policy_gradient.base import BasePolicyGradientAgent


class ReinforceAgent(BasePolicyGradientAgent):
    """
    Implémentation de REINFORCE classique.

    Idée :
    ------
    On joue un épisode complet, puis on met à jour la policy
    en utilisant les returns G_t.

    Formule :
    ---------
    loss = - somme( log_prob(action_t) * G_t )

    Intuition :
    -----------
    - si une action a conduit à un bon return, on augmente sa probabilité
    - si elle a conduit à un mauvais return, on la diminue
    """

    def mettre_a_jour_policy(self, liste_etats, liste_log_prob_actions, liste_rewards):
        """
        Met à jour la policy à la fin de l'épisode.

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
        # Chaque return mesure la qualité d'une action sur le long terme
        liste_returns = self.calculer_returns(liste_rewards)

        # Conversion en tensor PyTorch
        tensor_returns = torch.tensor(liste_returns, dtype=torch.float32)

        # ------------------------------------------------------------
        # ETAPE 2 : normaliser les returns
        # ------------------------------------------------------------
        # Cette normalisation aide souvent à stabiliser l'apprentissage
        if len(tensor_returns) > 1:
            moyenne = tensor_returns.mean()
            ecart_type = tensor_returns.std()

            if ecart_type.item() > 1e-8:
                tensor_returns = (tensor_returns - moyenne) / (ecart_type + 1e-8)

        # ------------------------------------------------------------
        # ETAPE 3 : calculer la loss REINFORCE
        # ------------------------------------------------------------
        # Pour chaque étape :
        # loss_t = - log_prob(action_t) * G_t
        liste_losses = [
            -log_prob * valeur_return
            for log_prob, valeur_return in zip(liste_log_prob_actions, tensor_returns)
        ]

        # On additionne toutes les contributions de l'épisode
        loss_totale = torch.stack(liste_losses).sum()

        # ------------------------------------------------------------
        # ETAPE 4 : mettre à jour le réseau de policy
        # ------------------------------------------------------------
        self.optimizer.zero_grad()   # remet les gradients à zéro
        loss_totale.backward()       # calcule les gradients
        self.optimizer.step()        # met à jour les poids

        return loss_totale.item()