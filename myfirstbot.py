# Import the API objects
from api import State, util
import random
import os
from itertools import chain

import joblib

# Path of the model we will use. If you make a model
# with a different name, point this line to its path.
DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/model.pkl'


class Bot:
    # How many samples to take per move
    __num_samples = -1
    # How deep to sample
    __depth = -1

    __randomize = True

    __model = None

    def __init__(self, num_samples=4, depth=8, randomize=True, model_file=DEFAULT_MODEL):
        self.__num_samples = num_samples
        self.__depth = depth

        print(model_file)
        self.__randomize = randomize

        # Load the model
        self.__model = joblib.load(model_file)

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

        if state.get_phase() == 1:
            return util.ratio_points(state, player)

        elif state.get_phase() == 2:
            # Convert the state to a feature vector
            feature_vector = [features(state)]

            # These are the classes: ('won', 'lost')
            classes = list(self.__model.classes_)

            # Ask the model for a prediction
            # This returns a probability for each class
            prob = self.__model.predict_proba(feature_vector)[0]

            # Weigh the win/loss outcomes (-1 and 1) by their probabilities
            res = -1.0 * prob[classes.index('lost')] + 1.0 * prob[classes.index('won')]

            return res

    def value(self, state):
        """
        Return the value of this state and the associated move
        :param
        state:
        :return: val, move: the value of the state, and the best move.
        """

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        player = state.whose_turn()

        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a function call so that 'value' will
            # contain the predicted value of 'next_state'
            # NOTE: This is different from the line in the minimax/alphabeta bot
            value = next_state.get_points(player)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_value, best_move


def maximizing(state):
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1


def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.

    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    feature_set = []

    # Add player 1's points to feature set
    p1_points = state.get_points(1)

    # Add player 2's points to feature set
    p2_points = state.get_points(2)

    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(1)

    # Add plauer 2's pending points to feature set
    p2_pending_points = state.get_pending_points(2)

    # Get trump suit
    trump_suit = state.get_trump_suit()

    # Add phase to feature set
    phase = state.get_phase()

    # Add stock size to feature set
    stock_size = state.get_stock_size()

    # Add leader to feature set
    leader = state.leader()

    # Add whose turn it is to feature set
    whose_turn = state.whose_turn()

    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()


    ################## You do not need to do anything below this line ########################

    perspective = state.get_perspective()

    # Perform one-hot encoding on the perspective.
    # Learn more about one-hot here: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    perspective = [card if card != 'U'   else [1, 0, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'S'   else [0, 1, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P1H' else [0, 0, 1, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P2H' else [0, 0, 0, 1, 0, 0] for card in perspective]
    perspective = [card if card != 'P1W' else [0, 0, 0, 0, 1, 0] for card in perspective]
    perspective = [card if card != 'P2W' else [0, 0, 0, 0, 0, 1] for card in perspective]

    # Append one-hot encoded perspective to feature_set
    feature_set += list(chain(*perspective))

    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points/total_points if total_points > 0 else 0.)
    feature_set.append(p2_points/total_points if total_points > 0 else 0.)

    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points/total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points/total_pending_points if total_pending_points > 0 else 0.)

    # Convert trump suit to id and add to feature set
    # You don't need to add anything to this part
    suits = ["C", "D", "H", "S"]
    trump_suit_onehot = [0, 0, 0, 0]
    trump_suit_onehot[suits.index(trump_suit)] = 1
    feature_set += trump_suit_onehot

    # Append one-hot encoded phase to feature set
    feature_set += [1, 0] if phase == 1 else [0, 1]

    # Append normalized stock size to feature set
    feature_set.append(stock_size/10)

    # Append one-hot encoded leader to feature set
    feature_set += [1, 0] if leader == 1 else [0, 1]

    # Append one-hot encoded whose_turn to feature set
    feature_set += [1, 0] if whose_turn == 1 else [0, 1]

    # Append one-hot encoded opponent's card to feature set
    opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
    feature_set += opponents_played_card_onehot

    # Return feature set
    return feature_set
