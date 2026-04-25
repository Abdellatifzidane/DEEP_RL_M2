import torch
import torch.optim as optim
from agents.policy_gradient.base import BasePolicyGradientAgent, ValueNetwork


class ReinforceCriticAgent(BasePolicyGradientAgent):
    """
    Implémentation de REINFORCE avec baseline apprise par un critic.

    Idée :
    ------
    Au lieu de prendre comme baseline une simple moyenne,
    on apprend une baseline avec un deuxième réseau :
    le critic.

    Le critic approxime :
        V(s) = valeur attendue de l'état s

    On utilise ensuite :
        advantage_t = G_t - V(s_t)

    Formules :
    ----------
    Policy loss :
        loss_policy = - somme(log_prob(action_t) * advantage_t)

    Value loss :
        loss_value = moyenne((V(s_t) - G_t)^2)
    """

    def __init__(
        self,
        taille_etat,
        nombre_actions,
        gamma=0.99,
        learning_rate_policy=0.001,
        learning_rate_value=0.001
    ):
        super().__init__(
            taille_etat=taille_etat,
            nombre_actions=nombre_actions,
            gamma=gamma,
            learning_rate=learning_rate_policy
        )

        self.value_network = ValueNetwork(taille_etat)
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate_value
        )

    def mettre_a_jour_policy(self, liste_etats, liste_log_prob_actions, liste_rewards):
        """
        Met à jour la policy et le critic à la fin de l'épisode.
        """
        if len(liste_log_prob_actions) == 0:
            return 0.0

        # 1) Returns
        liste_returns = self.calculer_returns(liste_rewards)
        tensor_returns = torch.tensor(liste_returns, dtype=torch.float32)

        # 2) Valeurs V(s_t)
        liste_valeurs = []
        for etat in liste_etats:
            etat_tensor = self.convertir_etat_en_tensor(etat)
            valeur_etat = self.value_network(etat_tensor).squeeze()
            liste_valeurs.append(valeur_etat)

        tensor_valeurs = torch.stack(liste_valeurs)

        # 3) Avantages
        avantages = tensor_returns - tensor_valeurs.detach()

        # Optionnel mais souvent utile pour stabiliser
        if len(avantages) > 1:
            moyenne = avantages.mean()
            ecart_type = avantages.std()
            if ecart_type.item() > 1e-8:
                avantages = (avantages - moyenne) / (ecart_type + 1e-8)

        # 4) Policy loss
        liste_policy_losses = [
            -log_prob * avantage
            for log_prob, avantage in zip(liste_log_prob_actions, avantages)
        ]
        loss_policy = torch.stack(liste_policy_losses).sum()

        # 5) Value loss
        loss_value = torch.mean((tensor_valeurs - tensor_returns) ** 2)

        # 6) Update policy
        self.optimizer.zero_grad()
        loss_policy.backward()
        self.optimizer.step()

        # 7) Update critic
        self.value_optimizer.zero_grad()
        loss_value.backward()
        self.value_optimizer.step()

        loss_totale = loss_policy + loss_value
        return loss_totale.item()