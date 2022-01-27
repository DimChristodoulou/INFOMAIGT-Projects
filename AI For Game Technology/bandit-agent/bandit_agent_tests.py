# Put your name and student ID here before submitting!
# Dimitris Christodoulou (5141761)

import copy
import math
import random
from collections import defaultdict
from typing import List
from uuid import uuid4

import numpy

import game as g
from random_agent import RandomAgent


def not_free_positions(board):
    return numpy.argwhere(board != 0)


def update_avg_rewards(pos, avg_moves, move_in_square, moves):
    avg_moves[pos] = float(moves[pos]) / float(move_in_square[pos])


def pick_current_best_move(avg_score, game):
    best_move_score = 0
    best_moves = []

    for i, j in numpy.ndindex(avg_score.shape):
        if avg_score[i, j] > best_move_score:
            best_move_score = avg_score[i, j]
            best_moves = [(i, j)]
        elif avg_score[i, j] == best_move_score:
            best_moves.append((i, j))

    if best_moves:
        return random.choice(best_moves)
    else:
        return game.board.random_free()


class BanditAgent:
    epsilon = 0.3

    def __init__(self, iterations, id):
        self.iterations = iterations
        self.id = id

    def make_move(self, game):
        i = 0
        epsilon = self.epsilon

        # Initialization
        average_rewards = numpy.zeros(game.board.shape)
        rewards = numpy.zeros(game.board.shape)
        num_of_moves = numpy.zeros(game.board.shape)

        # UCB ONLY
        ucb_rewards = numpy.zeros(game.board.shape)

        for x, y in not_free_positions(game.board.board):
            average_rewards[x, y] = -1000
            rewards[x, y] = None

            # UCB ONLY
            ucb_rewards[x] = None

        # perform the given number of iterations
        while i < self.iterations:
            i += 1

            # Create a deep clone of the current board
            copied_board = copy.deepcopy(game.board)

            # Create 2 random agent players to play the fake game
            new_players = [RandomAgent(1), RandomAgent(2)]

            # Epsilon greedy logic here
            factor = numpy.random.uniform(0, 1)

            if factor < epsilon or i < self.iterations/10:
                # Explore
                pos = game.board.random_free()
            else:
                # Exploit
                pos = pick_current_best_move(average_rewards, game)

            # upper confidence bound is here
            # pos = numpy.unravel_index(ucb_rewards.argmax(), ucb_rewards.shape)
            # max_upper_bound = 0

            # for j, k in game.board.free_positions():
            #     if num_of_moves[j, k] > 0:
            #         update_avg_rewards((j, k), average_rewards, num_of_moves, rewards)
            #         delta_i = 0.3 * math.sqrt(math.log(i + 1) / num_of_moves[(j, k)])
            #         ucb_rewards[j, k] = average_rewards[j, k] + delta_i
            #     else:
            #         ucb_rewards[j, k] = 1000000
            #
            #     if ucb_rewards[j, k] > max_upper_bound:
            #         max_upper_bound = ucb_rewards[j, k]
            #         pos = (j, k)

            copied_board.place(pos, new_players[1].id)

            new_game = g.Game.from_board(copied_board, game.objectives, new_players, None)
            winner = new_game.play()

            num_of_moves[pos] += 1
            if winner is not None:
                # if the player for which the move we picked won, then we consider this move as 'good' and award +1
                if winner.id == new_players[1].id:
                    rewards[pos] += 1
                if winner.id == new_players[0].id:
                    rewards[pos] -= 1
            if winner is None:
                rewards[pos] += 0.5

            update_avg_rewards(pos, average_rewards, num_of_moves, rewards)

        # return the best move you've found here
        return pick_current_best_move(average_rewards, game)

    def __str__(self):
        return f'Player {self.id} (BanditAgent)'
