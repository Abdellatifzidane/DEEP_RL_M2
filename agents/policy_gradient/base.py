import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod


class PolicyNetwork(nn.Module):
    """
    Réseau de policy :
    état -> 2 couches cachées -> scores des actions
    """

    def __init__(self, taille_etat, nombre_actions):
        super().__init__()

        self.fc1 = nn.Linear(taille_etat, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, nombre_actions)
        self.activation = nn.ReLU()

    def forward(self, etat_tensor):
        x = self.fc1(etat_tensor)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        scores_actions = self.fc3(x)
        return scores_actions


class ValueNetwork(nn.Module):
    """
    Réseau de valeur :
    état -> 2 couches cachées -> valeur scalaire V(s)
    Utilisé pour l'agent avec critic.
    """

    def __init__(self, taille_etat):
        super().__init__()

        self.fc1 = nn.Linear(taille_etat, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, etat_tensor):
        x = self.fc1(etat_tensor)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        valeur = self.fc3(x)
        return valeur


class BasePolicyGradientAgent(ABC):
    """
    Classe de base pour les agents de type Policy Gradient.

    Contient la logique commune :
    - réseau de policy
    - conversion état -> tensor
    - calcul des probabilités
    - choix d'action
    - calcul des returns
    - boucle d'un épisode complet

    Chaque sous-classe définit seulement sa propre
    méthode de mise à jour de la policy.
    """

    def __init__(self, taille_etat, nombre_actions, gamma=0.99, learning_rate=0.001):
        self.taille_etat = taille_etat
        self.nombre_actions = nombre_actions
        self.gamma = gamma

        self.policy_network = PolicyNetwork(taille_etat, nombre_actions)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def convertir_etat_en_tensor(self, etat):
        """
        Convertit un état en tenseur PyTorch de forme [1, taille_etat].
        """
        etat_numpy = np.array(etat, dtype=np.float32)
        etat_tensor = torch.tensor(etat_numpy, dtype=torch.float32)
        etat_tensor = etat_tensor.unsqueeze(0)
        return etat_tensor

    def calculer_probabilites_actions(self, etat, actions_disponibles, env, avec_gradient=True):
        """
        Calcule les probabilités des actions disponibles uniquement.

        Important :
        actions_disponibles peut contenir des actions complexes
        comme des tuples (type, index), donc on les convertit
        d'abord en indices fixes grâce à env.action_to_index().
        """
        etat_tensor = self.convertir_etat_en_tensor(etat)

        if avec_gradient:
            scores_toutes_actions = self.policy_network(etat_tensor)[0]
        else:
            with torch.no_grad():
                scores_toutes_actions = self.policy_network(etat_tensor)[0]

        indices_actions_disponibles = [
            env.action_to_index(action) for action in actions_disponibles
        ]

        scores_actions_disponibles = torch.stack(
            [scores_toutes_actions[idx] for idx in indices_actions_disponibles]
        )

        probabilites = torch.softmax(scores_actions_disponibles, dim=0)
        return probabilites

    def choisir_action(self, env, avec_gradient=True):
        """
        Choisit une action selon la policy courante.
        """
        etat_courant = env.get_state()
        actions_disponibles = env.get_available_actions()

        if len(actions_disponibles) == 0:
            return None, None

        probabilites_actions = self.calculer_probabilites_actions(
            etat_courant,
            actions_disponibles,
            env,
            avec_gradient=avec_gradient
        )

        distribution = torch.distributions.Categorical(probs=probabilites_actions)
        index_action_choisie = distribution.sample()

        action_choisie = actions_disponibles[index_action_choisie.item()]

        if avec_gradient:
            log_prob_action_choisie = distribution.log_prob(index_action_choisie)
        else:
            log_prob_action_choisie = None

        return action_choisie, log_prob_action_choisie

    def calculer_returns(self, liste_rewards):
        """
        Calcule les returns actualisés de la fin vers le début.
        """
        liste_returns = []
        return_courant = 0.0

        for reward in reversed(liste_rewards):
            return_courant = reward + self.gamma * return_courant
            liste_returns.insert(0, return_courant)

        return liste_returns

    @abstractmethod
    def mettre_a_jour_policy(self, liste_etats, liste_log_prob_actions, liste_rewards):
        """
        Méthode abstraite.
        Chaque sous-classe définit sa propre mise à jour.
        """
        pass

    def jouer_un_episode_et_apprendre(self, env):
        """
        Joue un épisode complet puis met à jour la policy.
        """
        env.reset()

        liste_etats = []
        liste_actions = []
        liste_rewards = []
        liste_log_prob_actions = []

        reward_total_episode = 0
        nombre_etapes_episode = 0

        while not env.is_terminal():
            etat_courant = env.get_state()

            action_choisie, log_prob_action = self.choisir_action(env, avec_gradient=True)

            if action_choisie is None:
                break

            _, reward = env.step(action_choisie)

            liste_etats.append(etat_courant)
            liste_actions.append(action_choisie)
            liste_rewards.append(reward)
            liste_log_prob_actions.append(log_prob_action)

            reward_total_episode += reward
            nombre_etapes_episode += 1

        valeur_loss = self.mettre_a_jour_policy(
            liste_etats,
            liste_log_prob_actions,
            liste_rewards
        )

        return {
            "reward_total_episode": reward_total_episode,
            "nombre_etapes_episode": nombre_etapes_episode,
            "loss": valeur_loss,
            "etats": liste_etats,
            "actions": liste_actions,
            "rewards": liste_rewards
        }

    def agir(self, env):
        """
        Utilisé en inférence pure : choisit une action sans calculer de gradient.
        """
        action_choisie, _ = self.choisir_action(env, avec_gradient=False)
        return action_choisie