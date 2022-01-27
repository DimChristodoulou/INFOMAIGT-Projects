# Put your name and student ID here before submitting!
# Name (ID)
import copy
import random

# uncomment one of the two lines below, depending on which library you want to use
import numpy as np
import tensorflow as tf


# import torch

class NNAgent:
    def __init__(self, id):
        self.id = id
        self.model = tf.keras.models.load_model('nn1_model_3')

    def make_move(self, game):
        board_free_positions = game.board.free_positions()
        one_hot_encoded_free_positions = np.zeros((len(board_free_positions), 75), dtype=int)
        board_counter = 0

        for move in board_free_positions:
            temp_board_copy = copy.deepcopy(game.board)
            temp_board_copy.place(move, self.id)

            for x in range(5):
                for y in range(5):
                    one_hot_base_index = (y * 3) + (x * 15)

                    if temp_board_copy.value((x, y)) == 0:
                        one_hot_encoded_free_positions[board_counter, one_hot_base_index] = 1
                    elif temp_board_copy.value((x, y)) == self.id:
                        one_hot_encoded_free_positions[board_counter, one_hot_base_index + 2] = 1
                    else:
                        one_hot_encoded_free_positions[board_counter, one_hot_base_index + 1] = 1

            board_counter += 1

        p_best_move = -1
        move_counter = 0
        predictions = self.model.predict(one_hot_encoded_free_positions)
        current_best_move = game.board.random_free()

        for prediction in predictions:
            # We consider wins and ties as good options, so we add their prediction scores
            total_pred = prediction[1] + prediction[0]

            if p_best_move < total_pred:
                p_best_move = total_pred
                current_best_move = board_free_positions[move_counter]
            move_counter += 1

        return current_best_move

    def __str__(self):
        return f'Player {self.id} (NNAgent)'
