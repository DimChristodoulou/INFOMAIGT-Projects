# Put your name and student ID here before submitting!
# Name (ID)

import random
import tensorflow as tf
import os

# uncomment one of the two lines below, depending on which library you want to use
# import tensorflow
# import torch

class NNAgent:
    def __init__(self, id):
        self.id = id
        # initialise your neural network here
        #train_dataset_fp = tf.keras.utils.get_file(fname='out.txt',
        #                                           origin='out.txt')
        #print("Local copy of the dataset file: {}".format(train_dataset_fp))

    def make_move(self, game):
        # use your neural network to make a move here
        return game.board.random_free()

    def __str__(self):
        return f'Player {self.id} (NNAgent)'
