# Import the API objects
from api import State, util
import random


class Bot:
    # How many samples to take per move
    __num_samples = -1
    # How deep to sample
    __depth = -1

    __max_depth = -1
    __randomize = True

    def __init__(self, num_samples=4, depth=8, randomize=True, alpha_depth=8):
        self.__num_samples = num_samples
        self.__depth = depth
        self.__randomize = randomize
        self.__max_depth = alpha_depth

    def get_move(self, state):

        if state.get_phase() == 1:

            # See if we're player 1 or 2
            player = state.whose_turn()

            # Get a list of all legal moves
            moves = state.moves()

            # Sometimes many moves have the same, highest score, and we'd like the bot to pick a random one.
            # Shuffling the list of moves ensures that.
            random.shuffle(moves)

            best_score = float("-inf")
            best_move = None

            scores = [0.0] * len(moves)

            for move in moves:
                for s in range(self.__num_samples):

                    # If we are in an imperfect information state, make an assumption.

                    sample_state = state.make_assumption() if state.get_phase() == 1 else state

                    score = self.evaluate(sample_state.next(move), player)

                    if score > best_score:
                        best_score = score
                        best_move = move

            return best_move  # Return the best scoring move

        elif state.get_phase() == 2:

            val, move = self.value(state)

            return move

    def evaluate(self,
                 state,  # type: State
                 player  # type: int
                 ):
        # type: () -> float
        """
		Evaluates the value of the given state for the given player
		:param state: The state to evaluate
		:param player: The player for whom to evaluate this state (1 or 2)
		:return: A float representing the value of this state for the given player. The higher the value, the better the
			state is for the player.
		"""

        score = 0.0

        for _ in range(self.__num_samples):

            st = state.clone()

            # Do some random moves
            for i in range(self.__depth):
                if st.finished():
                    break

                st = st.next(random.choice(st.moves()))

            score += self.heuristic(st, player)

        return score / float(self.__num_samples)

    def heuristic(self, state, player):
        return util.ratio_points(state, player)

    def value(self, state, alpha=float('-inf'), beta=float('inf'), depth=0):
        """
        Return the value of this state and the associated move
        :param State state:
        :param float alpha: The highest score that the maximizing player can guarantee given current knowledge
        :param float beta: The lowest score that the minimizing player can guarantee given current knowledge
        :param int depth: How deep we are in the tree
        :return val, move: the value of the state, and the best move.
        """

        if state.finished():
            winner, points = state.winner()
            return (points, None) if winner == 1 else (-points, None)

        if depth == self.__max_depth:
            return heuristic(state)

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        for move in moves:

            next_state = state.next(move)
            value, _ = self.value(next_state, alpha, beta)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = best_value
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                    beta = best_value

            # Prune the search tree
            # We know this state will never be chosen, so we stop evaluating its children
            if alpha >= beta:
                break

        return best_value, best_move


def maximizing(state):
    # type: (State) -> bool
    """
    Whether we're the maximizing player (1) or the minimizing player (2).

    :param state:
    :return:
    """
    return state.whose_turn() == 1


def heuristic(state):
    # type: (State) -> float
    """
    Estimate the value of this state: -1.0 is a certain win for player 2, 1.0 is a certain win for player 1

    :param state:
    :return: A heuristic evaluation for the given state (between -1.0 and 1.0)
    """
    return util.ratio_points(state, 1) * 2.0 - 1.0, None
