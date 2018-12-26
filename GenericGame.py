import numpy as np


class GenericState:
    """
    This class specify a base class for 2d board game with 2 player
    use 1 for first player and 2 for second player
    It is used for the game class where each state is an implementation
    of this class.
    """
    def __init__(self, board, player_turn):
        self.board = board
        self.length, self.height = board.shape
        self.action_possible = self._get_actions()
        self.player_turn = player_turn
        self.corresp = self._get_corresp()
        self.id = self.__str__()

    def take_action(self, action):
        """
        Args:
            action: a valid action to take in this state

        Returns:
            next_state, value of the next_state and if the game is done in the next state
        """
        raise NotImplementedError

    def to_model(self):
        board = np.zeros((2, self.length, self.height))
        board[0] = self.board
        board[1] = self.player_turn
        return board

    def get_symmetries(self, pi):
        """

        Args:
            pi: policy vector
        Returns:
            all equivalent state of the current state with equivalent policy
        """
        # THINK LATER ON HOW TO MAKE THIS WORK (problem with allowed_moves)

        # board = np.zeros((2, self.length, self.height))
        # board[0] = np.flip(self.board, 1)
        # board[1] = self.player_turn
        # return [(self.to_model(), pi), (board, pi[::-1])]
        raise NotImplementedError

    def is_terminal(self, action):
        """
        For the minmaxAgent to know if it's a terminal state
        Returns:
            True if the game is over false otherwise
        """
        raise NotImplementedError

    def evaluate(self):
        """
        a function to evaluate the state if None just return 0
        Returns:
            float: evaluate the state
        """
        raise NotImplementedError

    def _get_actions(self):
        """

        Returns:
            all possible action from this state
        """
        raise NotImplementedError

    def _get_corresp(self):
        """

        Returns:
            a dict for visual representation of the state
            e.g {1: 'X', -1: 'O', 0: '-'} for connect4 or tictactoe...

        """
        raise NotImplementedError

    def __str__(self):
        return '\n'.join([''.join([self.corresp[y] for y in x]) for x in self.board.tolist()])

    def __repr__(self):
        return str(self.board)

    def render(self, logger):
        for row in self.board:
            logger.info(str([self.corresp[y] for y in row]))
        logger.info('-' * 50)
