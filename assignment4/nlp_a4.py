pip install dynet

import numpy as np
import json
from abc import abstractmethod
from enum import Enum
from collections import Counter
import matplotlib.pyplot as plt
import string
import dynet as dy
import pandas as pd
from sklearn.metrics import accuracy_score

from google.colab import drive
drive.mount('/content/drive')

"""# Data Preprocessing"""

SOS = '<start>'
EOS_TAG = '<end>'
EMPTY_STATE = '1:_ 2:_ 3:_ 4:_ 5:_ 6:_ 7:_'
BEAKER_COLORS = 'yorgbp'
NUM_COLORS = 6
NUM_BEAKER_POSITIONS = 4
NUM_BEAKERS = 7

train_x_data_path = 'drive/My Drive/NLP Assignments/A4/train.json'
dev_x_data_path = 'drive/My Drive/NLP Assignments/A4/dev.json'
test_x_data_path = 'drive/My Drive/NLP Assignments/A4/test_leaderboard.json'

train_interaction_y_data_path = 'drive/My Drive/NLP Assignments/A4/train_interaction_y.csv'
train_instruction_y_data_path = 'drive/My Drive/NLP Assignments/A4/train_instruction_y.csv'
dev_instruction_y_data_path = 'drive/My Drive/NLP Assignments/A4/dev_instruction_y.csv'
dev_interaction_y_data_path = 'drive/My Drive/NLP Assignments/A4/dev_interaction_y.csv'

dev_instruction_csv = pd.read_csv(dev_instruction_y_data_path, index_col="id")['final_world_state'].values
dev_interaction_csv = pd.read_csv(dev_interaction_y_data_path, index_col="id")['final_world_state'].values

# print(dev_instruction_csv[0:5])
# print(dev_interaction_csv[0:5])

def load_data(data):
    raw_data = json.load(open(data))
    raw_instructions = [[col['instruction'].translate(str.maketrans('', '', string.punctuation)) for col in data_line['utterances']] for data_line in raw_data]
    raw_actions = [[col['actions'] for col in data_line['utterances']] for data_line in raw_data]
    raw_initial_environments = [np.tile(data_line['initial_env'], len(data_line['utterances'])) for data_line in raw_data]
    raw_ids = [data_line['identifier'] for data_line in raw_data]
    
    processed_instructions = [char_item.replace("'", '') for item in raw_instructions for char_item in item]
    processed_actions = [([SOS] + sub_item + [EOS_TAG]) for item in raw_actions for sub_item in item]
    processed_initial_environments = [(sub_env) for item in raw_initial_environments for sub_env in item]
    
    final_data = list()
    final_data.append(processed_instructions)
    final_data.append(processed_actions)
    final_data.append(processed_initial_environments)
    final_data.append(raw_ids)
    return final_data

train_data = load_data(train_x_data_path)
dev_data = load_data(dev_x_data_path)
test_data = load_data(test_x_data_path)

conv_int_to_char = sorted(set(['g','o','r','b','p','y','_']))
conv_char_to_int = {c:i for i,c in enumerate(conv_int_to_char)}

flatten_instruction_words = [word for sentence in train_data[0] for word in sentence.split()]
flatten_action_words = [word for sentence in train_data[1] for word in sentence]
flatten_instruction_counter = Counter(flatten_instruction_words)

filtered_vocab_list = [word if flatten_instruction_counter[word] >= 7 else '<UNK>' for word in flatten_instruction_words]
filtered_vocab_set = set(filtered_vocab_list + ['<end>'])
vocab = sorted(filtered_vocab_set)
vocab_actions = sorted(set(flatten_action_words))

vocab_dict = dict()
for idx, word in enumerate(vocab):
    vocab_dict[word] = idx

vocab_actions_dict = dict()
for i, word in enumerate(vocab_actions):
    vocab_actions_dict[word] = i

"""# Provided Classes"""

EOS = "_EOS"
ACTION_SEP = ";"
NO_ARG = "_NONE"

class WorldState():
    """Abstract class for a world state."""
    @abstractmethod
    def execute_seq(self, actions):
        """Execute a sequence of actions on a world state.
        Args:
            actions (list of any): The sequence of actions to execute.
        """
        pass

    @abstractmethod
    def distance(self, other_world_state):
        """ Computes a distance between itself and another world state of the same type.
        Args:
            other_world_state (WorldState): the world state to compare with.
        Returns:
            float, representing the distance.
        """
        pass


class ExecutionFSA():
    """Abstract class for an FSA that can execute various actions."""
    @abstractmethod
    def is_valid(self):
        """Returns whether the current FSA state is valid."""
        pass

    @abstractmethod
    def is_in_action(self):
        """Returns whether the current FSA state is in an action."""
        pass

    @abstractmethod
    def world_state(self):
        """Returns the current world state."""
        pass

    @abstractmethod
    def valid_feeds(self):
        """Returns the valid actions that can be executed."""
        pass

    @abstractmethod
    def peek_complete_action(self, action, arg1, arg2):
        """Returns the world state that would happen if executing action with arg1 and arg2."""
        pass

    @abstractmethod
    def feed_complete_action(self, action, arg1, arg2):
        """Updates the world state of the FSA using action with arg1 and arg2."""
        pass

COLORS = ['y', 'o', 'r', 'g', 'b', 'p']
ACTION_POP = 'pop'
ACTION_PUSH = 'push'
ACTIONS = [ACTION_POP, ACTION_PUSH]

# FSA states of the execution FSA.
class FSAStates(Enum):
    """Contains the possible FSA states the Alchemy FSA can be in."""
    NO_ACTION = 1
    PUSH = 2
    PUSH_BEAKER = 3
    PUSH_BEAKER_COLOR = 4
    POP = 5
    POP_BEAKER = 6
    INVALID = 7

def token_is_beaker(token):
    """Returns whether a token represents a beaker.
    Inputs:
        token (str): The token.
    Returns:
        Boolean if token represents a beaker.
    """
    return token.isdigit() and 1 <= int(token) <= 7

def valid_feeds_push_beaker():
    """Returns all colors -- only can pass these after pushing a location."""
    # Can push all colors.
    return COLORS

def valid_feeds_push_beaker_color():
    """Returns valid actions after pushing a color to a beaker -- action sep."""
    # Must complete action.
    return [ACTION_SEP]

def valid_feeds_pop_beaker():
    """After popping, must return the action separator."""
    # Must complete action.
    return [ACTION_SEP]

def valid_feeds_invalid():
    """If invalid, just push the EOS."""
    # Nothing is valid, just wrap it up.
    return [EOS]

# Execution FSA.
class AlchemyFSA(ExecutionFSA):
    """FSA for the Alchemy domain.
    Attributes:
        _world_state (AlchemyState): The current world state.
        _fsa_world_state (FSAStates): The current FSA state.
        _current_beaker (int or None): The current beaker being used.
        _current_color (char or None): The current color being pushed.
    """
    def __init__(self, world_state):
        self._world_state = world_state
        self._fsa_world_state = FSAStates.NO_ACTION
        self._current_beaker = None
        self._current_color = None

    def is_in_action(self):
        return self._fsa_world_state == FSAStates.NO_ACTION

    def is_valid(self):
        return self._fsa_world_state != FSAStates.INVALID

    def world_state(self):
        return self._world_state

    def _valid_feeds_no_action(self):
        # If all beakers are empty, only 'push' is possible, or wrap it up.
        if all([not b for b in self._world_state.beakers()]):
            return [ACTION_PUSH, EOS]
        return ACTIONS + [EOS]

    def _valid_feeds_push(self):
        # Can push to all beakers. Beaker IDs start at 1.
        return map(str, range(1, len(self._world_state.beakers()) + 1))

    def _valid_feeds_pop(self):
        # Can pop from all beakers that have something in them.
        return map(lambda i: str(i + 1),
                   filter(lambda i: len(self._world_state.beakers()[i]) != 0,
                          range(len(self._world_state.beakers()))))

    def valid_feeds(self):
        valid_funcs = {
            FSAStates.NO_ACTION: self._valid_feeds_no_action,
            FSAStates.PUSH: self._valid_feeds_push,
            FSAStates.PUSH_BEAKER: valid_feeds_push_beaker,
            FSAStates.PUSH_BEAKER_COLOR: valid_feeds_push_beaker_color,
            FSAStates.POP: self._valid_feeds_pop,
            FSAStates.POP_BEAKER: valid_feeds_pop_beaker,
            FSAStates.INVALID: valid_feeds_invalid
        }
        return valid_funcs[self._fsa_world_state]()

    def valid_actions(self):
        """Returns the valid actions in the FSA given the current world and FSA state"""
        valid_actions = [(EOS, NO_ARG, NO_ARG)]
        for location in range(7):
            for color in COLORS:
                valid_actions.append(("push", str(location + 1), color))
            if self._world_state.beakers()[location]:
                valid_actions.append(("pop", str(location + 1), NO_ARG))
        return valid_actions

    def peek_complete_action(self, action, arg1, arg2):
        if self._fsa_world_state != FSAStates.NO_ACTION:
            return None

        if action == ACTION_POP and token_is_beaker(arg1) and arg2 == NO_ARG:
            world_state = self._world_state.pop(int(arg1))
            return world_state

        if action == ACTION_PUSH and token_is_beaker(arg1) and arg2 in COLORS:
            world_state = self._world_state.push(
                int(arg1), arg2)
            return world_state

        raise Exception('should never happen')

    def feed_complete_action(self, action, arg1, arg2):
        if self._fsa_world_state != FSAStates.NO_ACTION:
            self._fsa_world_state = FSAStates.INVALID
            return None

        if action == ACTION_POP and token_is_beaker(arg1) and arg2 == NO_ARG:
            self._world_state = self._world_state.pop(int(arg1))
            if self._world_state is None:
                self._fsa_world_state = FSAStates.INVALID
            else:
                self._fsa_world_state = FSAStates.NO_ACTION
            return self._world_state

        if action == ACTION_PUSH and token_is_beaker(arg1) and arg2 in COLORS:
            self._world_state = self._world_state.push(
                int(arg1), arg2)
            if self._world_state is None:
                self._fsa_world_state = FSAStates.INVALID
            else:
                self._fsa_world_state = FSAStates.NO_ACTION
            return self._world_state

        raise Exception('should never happen')

class AlchemyWorldState(WorldState):
    """ The Alchemy world state definition.
    Attributes:
        _beakers (list of list of str): Beakers in the world state.
    """
    def __init__(self, string=None):
        self._beakers = [[]] * 7
        if string:
            string = [beaker.split(':')[1] for beaker in string.split()]
            self._beakers = []
            for beaker in string:
                if beaker == '_':
                    self._beakers.append([])
                else:
                    self._beakers.append(list(beaker))
        else:
            self._beakers = [[]] * 7

    def __eq__(self, other):
        return isinstance(other, AlchemyWorldState) and self._beakers == other.beakers()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return ' '.join([str(i) +
                         ':' +
                         ''.join(beaker) if beaker else str(i) +
                         ':_' for i, beaker in zip(range(1, 8), self._beakers)])

    def __len__(self):
        return len(self._beakers)

    def __iter__(self):
        return self._beakers.__iter__()

    def beakers(self):
        """ Returns the beakers for the world state. """
        return self._beakers

    def components(self):
        """Returns the beakers."""
        return self.beakers()

    def set_beakers(self, beakers):
        """ Sets the beakers of this class to something else.
        Inputs:
            beakers (list of list of str): The beakers to set.
        """
        self._beakers = beakers

    def set_beaker(self, index, new_value):
        """ Resets the units for a specific beaker.
        Inputs:
            index (int): The beaker to reset.
            new_value (list of str): The new values for the beaker.
        """
        self._beakers[index] = new_value

    def pop(self, beaker):
        """ Removes a unit from a beaker.
        Inputs:
            beaker (int): The beaker to pop from.
        Returns:
            AlchemyWorldState, representing the world state after popping.
        """
        beaker -= 1
        if self._beakers[beaker]:
            new_world_state = AlchemyWorldState()
            new_world_state.set_beakers(self._beakers[:])
            new_world_state.set_beaker(beaker, self._beakers[beaker][:-1])
            return new_world_state
        return None

    def push(self, beaker, color):
        """ Adds a new unit to a beaker.
        Inputs:
            beaker (int): The beaker to add to.
            color (str): The color to add.
        Returns:
            AlchemyWorldState, representing the world state after pushing.
        """
        beaker -= 1
        new_world_state = AlchemyWorldState()
        new_world_state.set_beakers(self._beakers[:])
        new_world_state.set_beaker(beaker, self._beakers[beaker] + [color])
        return new_world_state

    def execute_seq(self, actions):
        fsa = AlchemyFSA(self)
        for action in actions:
            peek_world_state = fsa.peek_complete_action(*action)
            if peek_world_state:
                fsa.feed_complete_action(*action)
        return fsa.world_state()

    def distance(self, other_world_state):
        """Implement this if you want."""
        return 0.

def execute(world_state, action_sequence):
    """Executes an action sequence on a world state.
    Inputs:
        world_state (str): String representing an AlchemyWorldState.
        action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
            (like in the JSON file).
    """
    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)

    for action in action_sequence:
        split = action.split(" ")
        act = split[0]
        arg1 = split[1]
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]
        fsa.feed_complete_action(act, arg1, arg2)
    return fsa.world_state()

"""# Model"""

# Model implementation courtesy of: https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py, https://talbaumel.github.io/blog/attention/
class Model():
    """Model predicts a sequence of actions, given an instruction and a starting world state.
    """
    def __init__(self, vocab_actions, vocab, char_to_int):
        # TODO: implement this function. Should construct all subcomponents of the model,
        # e.g., the RNNs, or any parameters as part of the parameter collection. Should
        # initialize the optimizer.
        self._vocab_actions = vocab_actions
        self._vocab = vocab
        self._char_to_int = char_to_int
        
        self._LSTM_NUM_OF_LAYERS = 2
        self._EMBEDDINGS_SIZE = 50
        self._CHAR_EMBEDDINGS_SIZE = 20 
        self._STATE_SIZE = 100 
        self._ATTENTION_SIZE = 100
        
        self._VOCAB_SIZE = len(vocab)
        self._ACTION_SIZE = len(vocab_actions)
        self._VOCAB_SIZE_CHAR = len(conv_int_to_char)

        self._pc = dy.ParameterCollection()
        
        self._ENC_FWD_LSTM = dy.LSTMBuilder(self._LSTM_NUM_OF_LAYERS, self._EMBEDDINGS_SIZE, self._STATE_SIZE, self._pc)
        self._DEC_LSTM = dy.LSTMBuilder(self._LSTM_NUM_OF_LAYERS, self._EMBEDDINGS_SIZE + self._STATE_SIZE*2, self._STATE_SIZE, self._pc)
        self._ENC_FWD_LSTM_CHAR = dy.LSTMBuilder(self._LSTM_NUM_OF_LAYERS, 50, 75, self._pc)

        self._input_lookup = self._pc.add_lookup_parameters((self._VOCAB_SIZE, self._EMBEDDINGS_SIZE))
        self._char_lookup = self._pc.add_lookup_parameters((self._VOCAB_SIZE_CHAR, self._EMBEDDINGS_SIZE))
        self._output_lookup = self._pc.add_lookup_parameters((self._ACTION_SIZE, self._EMBEDDINGS_SIZE))
        self._pos_lookup = self._pc.add_lookup_parameters((7, 25))
        
        self._R = self._pc.add_parameters((self._ACTION_SIZE, self._STATE_SIZE))
        self._bias = self._pc.add_parameters((self._ACTION_SIZE))
        self._attention_v = self._pc.add_parameters((1, self._ATTENTION_SIZE))
        self._attention_w1 = self._pc.add_parameters((self._ATTENTION_SIZE, self._STATE_SIZE))
        self._attention_w2 = self._pc.add_parameters((self._ATTENTION_SIZE, self._STATE_SIZE))
        self._attention_b1 = self._pc.add_parameters((self._ATTENTION_SIZE, self._STATE_SIZE))
        self._attention_b2 = self._pc.add_parameters((self._ATTENTION_SIZE))
        
        self._trainer = dy.SimpleSGDTrainer(self._pc)
        
    def _encode_seq(self, input_seq):
        """Encodes an input sequence.
        TODO: implement this function. "input_seq" is intentionally vague, and could
        include any combination of inputs: a single utterance, a sequence of utterances,
        the past utterances concatenate... it's up to you! In general, this function
        should use an RNN to encode input (natural language) tokens and return some
        hidden states (as a sequence) corresponding to the tokens.
        """
        sentence = input_seq + ' <end>'
        remove_spaces_input_seq = sentence.split(' ')
        s_to_vec = list()
        for action_word in remove_spaces_input_seq:
            if action_word in vocab_dict:
                s_to_vec.append(vocab_dict[action_word])
            else:
                s_to_vec.append(vocab_dict["<UNK>"])
        return s_to_vec

    def _encode_world_state(self, world_state):
        """Encodes an Alchemy World State.
        TODO: implement this function. Like _encode, this function can encode a single
        world state (i.e., the current world state before executing the action),
        or a sequence of world states.
        It's up to you on how to encode world states. You can use RNNs, MLPs, etc.
        """
        remove_spaces_world_state = world_state.split(' ')
        encoded_world_state = list()
        for beaker_info in remove_spaces_world_state:
            bkr_colors = beaker_info.split(':')
            encoded_world_state.append(bkr_colors[1])
        return encoded_world_state

    def _decode(self, encode_utterance, encode_state, action_word, curr_state):
        """Computes probability distributions over a sequence of output tokens.
        TODO: implement this function. Consider a few things:
        1) What inputs does this function take? At minimum, it should probably take
           representations of the current utterance and the current world state. 
        2) The difference between teacher-forcing and using predicted tokens to decode.
           In the first, you feed gold tokens into the decoder at each step. This is
           mostly used during training. But during prediction, you need to feed each
           predicted token into the decoder RNN. You may want to split this function
           into two, one where you feed gold tokens and another where you feed predicted ones.
           The reason: during training, you don't need to compute an argmax (get the
           predicted tokens). You just need the probability distributions (and as dy.Expressions).
           Hint: make sure not to call .value() or any forward passes during training, and think
           about how to batch the losses per-token so the code is efficient! 
           During evaluation, you will obviously need acccess to the values of the
           probability distributions over actions in order to decode greedily (or using beam search).
        3) What's your output space look like? You could predict single-token-by-token,
           or you could collapse each action-arg1-arg2 into a single prediction. For simplicity,
           I suggest trying token-by-token first.
        """
        curr_state = curr_state.add_input(dy.concatenate([action_word, encode_utterance, encode_state]))
        calc_softmax = (self._R) * curr_state.output() + self._bias
        sftmx_probs = dy.softmax(calc_softmax)
        pred = np.argsort(sftmx_probs.npvalue())[-1]
        return sftmx_probs, pred

    def train(self, batch):
        """Updates model parameters, e.g. in a batch (suggested).
        TODO: implement this function. Suggested impleemntation is, for each
        item in the batch, compute the loss (cross entropy) over the entire sequence,
        then backpropagate the gradient of the average (over output sequence length)
        loss (as a dy.Expression). Remember to renew the graph at the beginning!
        In general, you will want to encode the inputs (utterances and world states), then
        decode a sequence of probability distributions over output tokens.
        """
        sentence, alchemy_env_state, prev_loss, output = batch[0], batch[1], batch[2], batch[3]
        sentence_vector, character_vector, loss, action_predictions = list(), list(), list(), list()
        sentence_enc_fwd, character_enc_fwd = self._ENC_FWD_LSTM.initial_state(), self._ENC_FWD_LSTM_CHAR.initial_state()
        sentence_to_vec = self._encode_seq(sentence)

        for beaker_idx, beaker_states in enumerate(self._encode_world_state(alchemy_env_state)):
            for color in beaker_states:
                color_to_vec = self._char_to_int[color]
                character_enc_fwd = character_enc_fwd.add_input(self._char_lookup[color_to_vec])
            character_enc_fwd_output = character_enc_fwd.output()
            expression = dy.concatenate([character_enc_fwd_output, self._pos_lookup[beaker_idx]])
            character_vector.append(expression)

        for word in sentence_to_vec:
            sentence_enc_fwd = sentence_enc_fwd.add_input(self._input_lookup[word])
            calc_softmax = self._attention_w1 * sentence_enc_fwd.output() + self._attention_b2
            sentence_vector.append(dy.softmax(calc_softmax))
        
        dy_sentence_vector, dy_character_vector = dy.concatenate(sentence_vector, d=1), dy.concatenate(character_vector, d=1)

        encode_output = sentence_enc_fwd.output()
        s_decoded_state = self._DEC_LSTM.initial_state(sentence_enc_fwd.s())
        s_decoded_state_output = s_decoded_state.output()
        
        idx, starting_action, isEnd = 1, "<start>", False # since <start> is @ index 0
        while not (output[idx] == '<end>'):
            word, curr_action = vocab_actions_dict[starting_action], vocab_actions_dict[output[idx]]
            out_vec = [dy.dot_product(item, s_decoded_state_output) for item in sentence_vector]
            att_calc = self._attention_b1 * s_decoded_state_output
            out_vec_char = [self._attention_v * dy.tanh(self._attention_w2 * item + att_calc) for item in character_vector]
            
            weight, weight_char = dy.softmax(dy.concatenate(out_vec)), dy.softmax(dy.concatenate(out_vec_char))
            encode_output, encode_state = dy_sentence_vector * weight, dy_character_vector * weight_char
            word_weight, action_weight = self._input_lookup[word], self._output_lookup[word]
            s_decoded_state = s_decoded_state.add_input(dy.concatenate([word_weight, encode_output, encode_state]))

            probs, pred = self._decode(encode_output, encode_state, action_weight, s_decoded_state)
            pred_action = list(vocab_actions_dict.keys())[list(vocab_actions_dict.values()).index(pred)]                        
            action_predictions.append(pred_action)           
            loss.append(-dy.log(dy.pick(probs, curr_action)))
            
            idx, starting_action = idx + 1, pred_action

            if pred_action == EOS_TAG:
                continue
            
            alchemy_env_state = str(execute(alchemy_env_state, [starting_action]))
          
            if alchemy_env_state == 'None':
                alchemy_env_state = EMPTY_STATE
        
        loss = dy.esum(loss)
        indices_to_del = [i for i in range(len(action_predictions)) if action_predictions[i] == SOS]
        for index in sorted(indices_to_del, reverse=True):
            del action_predictions[index]
        
        prev_loss = sentence_enc_fwd.output()
        return loss, action_predictions, prev_loss


    def predict(self, example):
        """Returns a predicted sequence given an example.
        TODO: implement this function. You will want to encode the inputs, and
        then decode an action sequence. You are welcome to return the action sequence,
        or resulting world state from executing the sequence, or both, whichever works for your code.
        """
        sentence, alchemy_env_state, prev_loss = example[0], example[1], example[2]
        sentence_vector, character_vector, loss, action_predictions = list(), list(), list(), list()
        sentence_enc_fwd, character_enc_fwd = self._ENC_FWD_LSTM.initial_state(), self._ENC_FWD_LSTM_CHAR.initial_state()
        sentence_to_vec = self._encode_seq(sentence)

        for beaker_idx, beaker_states in enumerate(self._encode_world_state(alchemy_env_state)):
            for color in beaker_states:
                color_to_vec = self._char_to_int[color]
                character_enc_fwd = character_enc_fwd.add_input(self._char_lookup[color_to_vec])
            character_enc_fwd_output = character_enc_fwd.output()
            expression = dy.concatenate([character_enc_fwd_output, self._pos_lookup[beaker_idx]])
            character_vector.append(expression)
        
        for word in sentence_to_vec:
            sentence_enc_fwd = sentence_enc_fwd.add_input(self._input_lookup[word])
            calc_softmax = self._attention_w1 * sentence_enc_fwd.output() + self._attention_b2
            sentence_vector.append(dy.softmax(calc_softmax))
        
        dy_sentence_vector, dy_character_vector = dy.concatenate(sentence_vector, d=1), dy.concatenate(character_vector, d=1)

        encode_output = sentence_enc_fwd.output()
        s_decoded_state = self._DEC_LSTM.initial_state(sentence_enc_fwd.s())
        s_decoded_state_output = s_decoded_state.output()

        threshold = 0
        idx, starting_action, isEnd = 1, "<start>", False # since <start> is @ index 0

        while (starting_action != '<end>'):
            threshold = threshold + 1

            word = vocab_actions_dict[starting_action]
            out_vec = [dy.dot_product(item, s_decoded_state_output) for item in sentence_vector]
            att_calc = self._attention_b1 * s_decoded_state_output
            out_vec_char = [self._attention_v * dy.tanh(self._attention_w2 * item + att_calc) for item in character_vector]
                        
            weight, weight_char = dy.softmax(dy.concatenate(out_vec)), dy.softmax(dy.concatenate(out_vec_char))
            encode_output, encode_state = dy_sentence_vector * weight, dy_character_vector * weight_char
            word_weight, action_weight = self._input_lookup[word], self._output_lookup[word]
            s_decoded_state = s_decoded_state.add_input(dy.concatenate([word_weight, encode_output, encode_state]))

            probs, _ = self._decode(encode_output, encode_state, action_weight, s_decoded_state)
            highest_so_far = 0
            
            pred_action = 'random'
            while pred_action != '<end>':
                highest_so_far = highest_so_far + 1
                if (highest_so_far >= 50):
                    highest_so_far = 1
                    break            
                pred = np.argsort(probs.vec_value())[-highest_so_far]
                pred_action = list(vocab_actions_dict.keys())[list(vocab_actions_dict.values()).index(pred)]
                if (pred_action == '<start>'): 
                    continue
                new_env_state = str(execute(alchemy_env_state, [pred_action]))
                if new_env_state == 'None':
                    continue
                break

            pred = np.argsort(probs.vec_value())[-highest_so_far]
            pred_action = list(vocab_actions_dict.keys())[list(vocab_actions_dict.values()).index(pred)]
            starting_action = pred_action
            
            if threshold >= 10:
                break
            
            action_predictions.append(pred_action)           
            alchemy_env_state = str(execute(alchemy_env_state, [starting_action]))
            
            if alchemy_env_state == 'None':
                alchemy_env_state = EMPTY_STATE

        indices_to_del = [i for i in range(len(action_predictions)) if action_predictions[i] == SOS]
        for index in sorted(indices_to_del, reverse=True):
            del action_predictions[index]
        
        prev_loss = sentence_enc_fwd.output()
        return action_predictions, prev_loss

def predict(model, data, outname):
    """Makes predictions for data given a saved model.
    
    This function should predict actions (and call the AlchemyFSA to execute them),
    and save the resulting world states in the CSV files (same format as *.csv).
    
    TODO: you should implement both "gold-previous" and "entire-interaction"
        prediction.
    
    In the first case ("gold-previous"), for each utterance, you start in the previous gold world state,
    rather than the on you would have predicted (if the utterance is not the first one).
    This is useful for analyzing errors without the problem of error propagation,
    and you should do this first during development on the dev data.
    
    In the second case ("entire-interaction"), you are continually updating
    a single world state throughout the interaction; i.e. for each instruction, you start
    in whatever previous world state you ended up in with your prediction. This method can be
    impacted by cascading errors -- if you made a previous incorrect prediction, it's hard
    to recover (or your instruction might not make sense). 
    
    For test labels, you are expected to predict /final/ world states at the end of each
    interaction using "entire-interaction" prediction (you aren't provided the gold
    intermediate world states for the test data).
    Inputs:
        model (Model): A seq2seq model for this task.
        data (list of examples): The data you wish to predict for.
        outname (str): The filename to save the predictions to.
    """    
    instruction_sentence, alchemy_env = data[0], data[2]
    prev_loss, num_instructions, action_states = None, 0, list()

    for curr_sentence, curr_alchemy_env_state in zip(instruction_sentence, alchemy_env):
        if num_instructions % 5 == 0: # if it's the first instruction for a new sequence of actions (i.e. instruction 1/5)
            dy.renew_cg()
            interaction_history_sentence, new_alchemy_env_state = curr_sentence, curr_alchemy_env_state
        else:
            interaction_history_sentence = prev_sentence + ' ' + EOS_TAG + ' ' + curr_sentence  # encoding history utterance from previous sentence / adding interaction history
            execute_new_env = execute(new_alchemy_env_state, generate)
            new_alchemy_env_state = str(execute_new_env)
            if new_alchemy_env_state == 'None':
                new_alchemy_env_state = EMPTY_STATE

        num_instructions, prev_sentence = num_instructions + 1, curr_sentence

        example = list()
        example.append(interaction_history_sentence)
        example.append(new_alchemy_env_state)
        example.append(prev_loss)
        generate, prev_loss = model.predict(example)
        action_states.append(generate)

        indices_to_del = [i for i in range(len(generate)) if generate[i] == SOS]
        for index in sorted(indices_to_del, reverse=True):
            del generate[index] 
    
    progressive_alchemy_env_states_list, final_alchemy_env_states_list = list(), list()
    for idx, env in enumerate(alchemy_env):
        if idx % 5 == 0: #If it's @ instruction 1/5, then no new env
            new_env_state = env
        
        new_env_state = str(execute(new_env_state, action_states[idx]))
        
        if new_env_state == 'None':
            new_env_state = EMPTY_STATE
        
        progressive_alchemy_env_states_list.append(new_env_state)
        
        if idx % 5 == 4: #If it's @ instruction 5/5, then you're done
            final_alchemy_env_states_list.append(new_env_state)
    
    return progressive_alchemy_env_states_list, final_alchemy_env_states_list

def train(model, train_data):
    """Finds parameters in the model given the training data.
    TODO: implement this function -- suggested implementation iterates over epochs,
        computing loss over training set (in batches, maybe), evaluates on a held-out set
        at each round (you are welcome to split train_data here, or elsewhere), and
        saves the final model parameters.
    Inputs:
        model (Model): The model to train.
        train_data (list of examples): The training examples given.
    """
    instruction_sentence, action_seq, alchemy_env = train_data[0], train_data[1], train_data[2]

    NUM_EPOCHS = 100
    prev_loss = 'None'
    
    for epoch in range(NUM_EPOCHS):   
        print('Current Epoch %d' % epoch)
        num_instructions, batch_loss = 0, list()
        
        for curr_sentence, curr_action_seq, curr_alchemy_env_state in zip(instruction_sentence, action_seq, alchemy_env):
            if num_instructions % 5 == 0: # if it's the first instruction for a new sequence of actions (i.e. instruction 1/5)
                interaction_history_sentence, new_alchemy_env_state = curr_sentence, curr_alchemy_env_state
            else:
                interaction_history_sentence = prev_sentence + ' ' + EOS_TAG + ' ' + curr_sentence  # encoding history utterance from previous sentence / adding interaction history
                execute_new_env = execute(new_alchemy_env_state, generate)
                new_alchemy_env_state = str(execute_new_env)
                if new_alchemy_env_state == 'None':
                    new_alchemy_env_state = EMPTY_STATE
            
            num_instructions, prev_sentence = num_instructions + 1, curr_sentence

            batch = list()
            batch.append(interaction_history_sentence)
            batch.append(new_alchemy_env_state)
            batch.append(prev_loss)
            batch.append(curr_action_seq)
            
            loss, generate, prev_loss = model.train(batch)
            batch_loss.append(loss)

            indices_to_del = [i for i in range(len(generate)) if generate[i] == SOS]
            for index in sorted(indices_to_del, reverse=True):
                del generate[index]          
            
            if len(batch_loss) >= 5:
                losses = dy.average(batch_loss)
                losses.forward()
                losses.backward()
                model._trainer.update()
                dy.renew_cg()
                batch_loss = list()

model = Model(vocab_actions, vocab, conv_char_to_int)
train(model, train_data)

temp1, temp2 = predict(model, dev_data, outname=None)