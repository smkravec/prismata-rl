import prismataengine

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    """
    def __init__(self, game_state, cards=11, player1=None, player2=None, ai_json=None, one_hot=False):
        self.init_game_state = prismataengine.GameState(game_state, cards, player1, player2, ai_json, one_hot)
        self.cards = cards
        self.player1 = player1
        self.player2 = player2
        self.ai_json = ai_json
        self.one_hot = one_hot

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return self.init_game_state.toVector()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        raise NotImplementedError()

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return len(self.init_game_state.getAbstractActions())

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        game_state = prismataengine.GameState(board.json(), self.cards, self.player1, self.player2, self.ai_json, self.one_hot)
        game_state.doAction(action)
        return game_state, game_state.activePlayer

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return self.getCanonicalForm(board, player).getAbstractActions()

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        if not board.isGameOver():
            return 0
        if board.winner() not in [prismataengine.Players.One, prismataengine.Players.Two]:
            return 0.1
        if board.winner() == player:
            return 1
        else:
            return -1

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        board = prismataengine.GameState(board.json(), self.cards, self.player1, self.player2, self.ai_json, self.one_hot)
        if board.activePlayer != player:
            board.inactivePlayer = board.activePlayer
            board.activePlayer = player
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        raise NotImplementedError()

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board)
