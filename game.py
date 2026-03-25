
from quatro import Quatro
from players import RandomPlayer, HumanPlayer, AgentPlayer

# ce module est une encpasulation des jeux et des joueur
# grace à ce module on peux appliquer des mode de jeux différents (humain vs humain, humain vs agent, agent vs agent) sans changer le code de l'environnement et des joueurs
# le module game gère le déroulement du jeu, l'alternance des joueurs, et l'application des actions choisies par les joueurs sur l'environnement

CHOOSE_PIECE = 0
PLACE_PIECE = 1
class game:
    def __init__(self,env, player_01, player_02):
        self.env = env
        self.player_01 = player_01
        self.player_02 = player_02
        self.current_player_index = 0 # index du joueur courant
        self.first_action = True

    def get_current_player_obj(self):
        if self.current_player_index == 0:
            return self.player_01
        else:
            return self.player_02  
    
    def reset(self):
        self.env.reset()
        self.current_player_index = 0  

    def start_game(self):
            for i in range(16): # maximum NB_CELLS tours de jeu
                if (self.first_action):
                    index = self.player_01.choose_action(CHOOSE_PIECE,self.env)
                    # appliquer l'action et mettre à jour l'état du jeu
                    self.env.apply_action(CHOOSE_PIECE,index)
                    # changer de joueur
                    self.current_player_index = 1 - self.current_player_index
                    self.first_action = False
                    print("Etat après l'action: {}".format(self.env.encode_state()))
                    print("le board après l'action: {}".format(self.env.board))
                    print("encodage de  l'action: {}".format(self.env.action_encode(CHOOSE_PIECE,index)))

                    win =  self.env.check_win()
                    if win:
                        print("Le joueur {} a gagné!".format(self.current_player_index))
                        break
                else:
                    index = self.player_02.choose_action(PLACE_PIECE,self.env)
                    # appliquer l'action et mettre à jour l'état du jeu
                    self.env.apply_action(PLACE_PIECE,index)
                    # changer de joueur
                    print("Etat après l'action: {}".format(self.env.encode_state()))
                    print("le board après l'action: {}".format(self.env.board))
                    print("encodage de  l'action: {}".format(self.env.action_encode(PLACE_PIECE,index)))

                    win =  self.env.check_win()
                    if win:
                        print("Le joueur {} a gagné!".format(self.current_player_index))
                        break

                    index = self.player_02.choose_action(CHOOSE_PIECE,self.env)
                    self.env.apply_action(CHOOSE_PIECE,index)
                    self.current_player_index = 1 - self.current_player_index
                    print("Etat après l'action: {}".format(self.env.encode_state()))
                    print("le board après l'action: {}".format(self.env.board))
                    print("encodage de  l'action: {}".format(self.env.action_encode(CHOOSE_PIECE,index)))


                    index = self.player_01.choose_action(PLACE_PIECE,self.env)
                    # appliquer l'action et mettre à jour l'état du jeu
                    self.env.apply_action(PLACE_PIECE,index)
                    print("Etat après l'action: {}".format(self.env.encode_state()))
                    print("le board après l'action: {}".format(self.env.board))
                    print("encodage de  l'action: {}".format(self.env.action_encode(PLACE_PIECE,index)))

                    win =  self.env.check_win()
                    if win:
                        print("Le joueur {} a gagné!".format(self.current_player_index))
                        break

                    # changer de joueur
                    index = self.player_01.choose_action(CHOOSE_PIECE,self.env)
                    self.env.apply_action(CHOOSE_PIECE,index)
                    self.current_player_index = 1 - self.current_player_index  
                    print("Etat après l'action: {}".format(self.env.encode_state()))            
                    print("le board après l'action: {}".format(self.env.board))
                    print("encodage de  l'action: {}".format(self.env.action_encode(CHOOSE_PIECE,index)))

    def start_game_without_print(self):
            for i in range(16): # maximum NB_CELLS tours de jeu
                if (self.first_action):
                    index = self.player_01.choose_action(CHOOSE_PIECE,self.env)
                    if index is None:
                        print("Aucune pièce disponible à choisir pour le joueur {}. Match nul!".format(self.current_player_index))
                        break
                    # appliquer l'action et mettre à jour l'état du jeu
                    self.env.apply_action(CHOOSE_PIECE,index)
                    # changer de joueur
                    self.current_player_index = 1 - self.current_player_index
                    self.first_action = False
                    win =  self.env.check_win()
                    if win:
                        print("Le joueur {} a gagné!".format(self.current_player_index))
                        break
                else:
                    index = self.player_02.choose_action(PLACE_PIECE,self.env)
                    if index is None:
                        print("Aucune pièce disponible à choisir pour le joueur {}. Match nul!".format(self.current_player_index))
                        break
                    # appliquer l'action et mettre à jour l'état du jeu
                    self.env.apply_action(PLACE_PIECE,index)
                    # changer de joueur
                    win =  self.env.check_win()
                    if win:
                        print("Le joueur {} a gagné!".format(self.current_player_index))
                        break

                    index = self.player_02.choose_action(CHOOSE_PIECE,self.env)
                    if index is None:
                        print("Aucune pièce disponible à choisir pour le joueur {}. Match nul!".format(self.current_player_index))
                        break
                    self.env.apply_action(CHOOSE_PIECE,index)
                    self.current_player_index = 1 - self.current_player_index
                    index = self.player_01.choose_action(PLACE_PIECE,self.env)
                    if index is None:
                        print("Aucune pièce disponible à choisir pour le joueur {}. Match nul!".format(self.current_player_index))
                        break
                    # appliquer l'action et mettre à jour l'état du jeu
                    self.env.apply_action(PLACE_PIECE,index)

                    win =  self.env.check_win()
                    if win:
                        print("Le joueur {} a gagné!".format(self.current_player_index))
                        break

                    # changer de joueur
                    index = self.player_01.choose_action(CHOOSE_PIECE,self.env)
                    if index is None:
                        print("Aucune pièce disponible à choisir pour le joueur {}. Match nul!".format(self.current_player_index))
                        break
                    self.env.apply_action(CHOOSE_PIECE,index)
                    self.current_player_index = 1 - self.current_player_index  


if __name__ == "__main__":
    env = Quatro()

    player_01 = RandomPlayer()
    player_02 = RandomPlayer()

    game_test = game(env, player_01, player_02)

    state = game_test.env.encode_state()

    print("Etat initial du jeu: {}".format(state))

    game_test.start_game()
