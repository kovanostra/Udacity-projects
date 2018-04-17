"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import operator


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """
    This heuristic returns the result of the division between the moves of the player (times 10)
    and the oponent's (plus 1 to avoid division with zero)

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return (10*len(game.get_legal_moves(player))) / (len(game.get_legal_moves(game.get_opponent(player))) + 1)


def custom_score_2(game, player):
    """
    This heuristic returns the difference of the number of available moves of the
    current player and the oponent (with a weight value of 2 on the oponents moves)

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)) - 2*len(game.get_legal_moves(game.get_opponent(player))))


def custom_score_3(game, player):
    """
    This heuristic returns the average number of legal moves of both players

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return (len(game.get_legal_moves(player)) + len(game.get_legal_moves(game.get_opponent(player))))/2


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            # Handle any actions required after timeout as needed
            if game.get_legal_moves():
                return random.choice(game.get_legal_moves())
            else:
                return (-1, -1)
            # Return the best move from the last completed search iteration

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def max_value(game, depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or not game.get_legal_moves():
                return self.score(game, self)
            
            depth -= 1
            v = float("-inf")
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a), depth))
            return v


        def min_value(game, depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or not game.get_legal_moves():
                return self.score(game, self)

            depth -= 1
            v = float("inf")
            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), depth))
            return v

        max_v = float("-inf")
        best_move = None

        if len(game.get_legal_moves()) == 1:
            best_move = game.get_legal_moves()[0]
        else:
            for a in game.get_legal_moves():
                v = min_value(game.forecast_move(a), depth - 1)
                if v > max_v:
                    max_v = v
                    best_move = a
            if not best_move and game.get_legal_moves():
                best_move = random.choice(game.get_legal_moves())
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.last_move = []

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = None
        depth = 1
        self.last_move = []

        while True:
            try:
                depth += 1
                best_move = self.alphabeta(game, depth)
                self.last_move.append(best_move)
            except SearchTimeout:
                if not best_move and game.get_legal_moves():
                    best_move = random.choice(game.get_legal_moves())
                    self.last_move.append(best_move)
                return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        def max_value(game, depth, alpha, beta):
            '''
            This method initiates the max part of the alpha-beta search.

            Takes the same inputs as the alphabeta() method and returns a max value score
            '''
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or not game.get_legal_moves():
                return self.score(game, self)
            
            depth -= 1
            v = float("-inf")
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a), depth, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v


        def min_value(game, depth, alpha, beta):
            '''
            This method initiates the min part of the alpha-beta search.

            Takes the same inputs as the alphabeta() method and returns a min value score.
            '''
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or not game.get_legal_moves():
                return self.score(game, self)

            depth -= 1
            v = float("inf")
            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), depth, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def ab_prune(game, depth, alpha, beta):
            '''
            This method initiates the alpha-beta search.

            Takes the same inputs as the alphabeta() method and returns a best move (int, int)
            '''
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            max_v = alpha
            best_move = None

            for a in game.get_legal_moves():
                v = min_value(game.forecast_move(a), depth - 1, alpha, beta)
                if v > max_v:
                    max_v = v
                    best_move = a
                alpha = max(alpha, max_v)  
            return best_move

        def opening_book(game, depth, alpha, beta):

            '''
            This method is useful for the begining of the game (first 10 moves).
            At first the player uses the center of the board and then tries to mirror 
            the moves of the oponent.

            Takes the same inputs as the alphabeta() method and returns a best move (int, int) 
            '''
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if len(game.get_legal_moves()) == game.width * game.height:
                # In the beginning of the game the player is placed at the centre of the board
                return (game.width // 2, game.height // 2)

            elif len(self.last_move) > 1:
                # During the first moves, the player mirrors the oponent if possible
                db = [(-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1)]
                for d in db:
                    # Search over all theoretically possible positions
                    # print(self.last_move)
                    position = tuple(map(operator.add, self.last_move[-2], d))
                    mirror_position = tuple(map(operator.sub, self.last_move[-2], d))
                    if (self.last_move[-1] == position) and (mirror_position in game.get_legal_moves()):
                        # If a mirror position is legal then it is returned as best
                        return mirror_position

            # If it is not the start of the game and not possible to mirror, then alpha-beta search is implemented
            best_move = ab_prune(game, depth, alpha, beta)
            return best_move

        # If the game is at early stage, then the player tries to implement moves from the opening book
        if (game.width * game.height - len(game.get_blank_spaces())) <= 10:
            best_move = opening_book(game, depth, alpha, beta)
        else:
            best_move = ab_prune(game, depth, alpha, beta)
        
        return best_move




