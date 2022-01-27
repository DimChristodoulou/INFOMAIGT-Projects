#! /usr/bin/env -S python -u
from game import Game
from board import Board
from random_agent import RandomAgent
from bandit_agent import BanditAgent
from score_agent import ScoreAgent
from neural_network_agent import NNAgent

import argparse, time, cProfile
import numpy as np
import multiprocessing as mp
from collections import Counter
from itertools import starmap
import tensorflow as tf
from tensorflow import keras


def main(args):
    if args.input:
        data = read_games(args.input)
        total_states = 0

        for winner, game in data:
            total_states += len(game)

        x_train = np.zeros((total_states, 75), dtype=int)
        y_train = np.zeros(total_states, dtype=int)
        board_index = 0

        for winner, game in data:
            for player, move, board in game:
                # We can get double the games if we train the nn for both players and just set player 2 as 1
                temp = []
                temp = encode_input_data(player, board, temp)
                x_train[board_index] = np.array(temp)

                if winner == 0:
                    y_train[board_index] = 0
                elif winner == player:
                    y_train[board_index] = 1
                else:
                    y_train[board_index] = 2

                board_index += 1

        mask = np.random.rand(len(x_train)) <= 0.8
        training_data = x_train[mask]
        training_labels = y_train[mask]
        testing_data = x_train[~mask]
        testing_labels = y_train[~mask]

        # Create the tf.keras.Sequential model by stacking layers.
        # Choose an optimizer and loss function for training.
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=75),  # array with 75 objects
            tf.keras.layers.Dense(250, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(3, activation='softmax')  # win/loss/draw, so 3
        ])

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(training_data, training_labels, batch_size=256, epochs=8)

        # Checks the models performance
        model.evaluate(testing_data, testing_labels, verbose=2)

        model.save("nn1_model_3", overwrite=False)

    work = []
    for i in range(args.games):
        # swap order every game
        if i % 2 == 0:
            players = [NNAgent(1), RandomAgent(2)]
        else:
            players = [RandomAgent(2), NNAgent(1)]

        work.append((args.size,
                     read_objectives(args.objectives),
                     players,
                     args.print_board))

    start = time.perf_counter()

    # the tests can be run in parallel, or sequentially
    # it is recommended to only use the parallel version for large-scale testing
    # of your agent, as it is harder to debug your program when enabled
    if args.parallel == None:
        results = starmap(play_game, work)
    else:
        # you probably shouldn't set args.parallel to a value larger than the
        # number of cores on your CPU, as otherwise agents running in parallel
        # may compete for the time available during their turn
        with mp.Pool(args.parallel) as pool:
            results = pool.starmap(play_game, work)

    stats = Counter(results)
    end = time.perf_counter()

    print(f'{stats[1]}/{stats[2]}/{stats[0]}')
    # print(f'Total time {end - start} seconds.')

    for t in [10, 100, 500]:
        work = []
        for i in range(args.games):
            # swap order every game
            if i % 2 == 0:
                players = [NNAgent(1), BanditAgent(t, 2)]
            else:
                players = [BanditAgent(t, 2), NNAgent(1)]

            work.append((args.size,
                         read_objectives(args.objectives),
                         players,
                         args.print_board))

        start = time.perf_counter()

        # the tests can be run in parallel, or sequentially
        # it is recommended to only use the parallel version for large-scale testing
        # of your agent, as it is harder to debug your program when enabled
        if args.parallel == None:
            results = starmap(play_game, work)
        else:
            # you probably shouldn't set args.parallel to a value larger than the
            # number of cores on your CPU, as otherwise agents running in parallel
            # may compete for the time available during their turn
            with mp.Pool(args.parallel) as pool:
                results = pool.starmap(play_game, work)

        stats = Counter(results)
        end = time.perf_counter()

        print(f'{stats[1]}/{stats[2]}/{stats[0]}')
        # print(f'Total time {end - start} seconds.')

    work = []
    for i in range(args.games):
        # swap order every game
        if i % 2 == 0:
            players = [NNAgent(1), ScoreAgent(2)]
        else:
            players = [ScoreAgent(2), NNAgent(1)]

        work.append((args.size,
                     read_objectives(args.objectives),
                     players,
                     args.print_board))

    start = time.perf_counter()

    # the tests can be run in parallel, or sequentially
    # it is recommended to only use the parallel version for large-scale testing
    # of your agent, as it is harder to debug your program when enabled
    if args.parallel == None:
        results = starmap(play_game, work)
    else:
        # you probably shouldn't set args.parallel to a value larger than the
        # number of cores on your CPU, as otherwise agents running in parallel
        # may compete for the time available during their turn
        with mp.Pool(args.parallel) as pool:
            results = pool.starmap(play_game, work)

    stats = Counter(results)
    end = time.perf_counter()

    print(f'{stats[1]}/{stats[2]}/{stats[0]}')
    # print(f'Total time {end - start} seconds.')


def play_game(boardsize, objectives, players, output, print_board=None):
    game = Game.new(boardsize, objectives, players, print_board == 'all')

    if output:
        with open(output, 'a') as outfile:
            print(boardsize, file=outfile)
            winner = game.play(outfile)
            print(f'winner={winner.id if winner else 0}', file=outfile)
    else:
        winner = game.play()

    if print_board == 'final':
        game.print_result(winner)

    return 0 if winner == None else winner.id


def read_objectives(filename):
    with open(filename) as file:
        lines = [line.strip() for line in file]

    i = 0
    shapes = []
    while i < len(lines):
        shape = []

        # shapes are separated by blank lines
        while i < len(lines) and lines[i].strip() != '':
            shape_line = []
            for char in lines[i].strip():
                shape_line.append(char == 'x')
            shape.append(shape_line)
            i += 1

        shapes.append(np.transpose(np.array(shape)))
        i += 1

    return shapes


def read_games(filename):
    with open(filename) as file:
        lines = list(file)

        games = []

        i = 0
        while i < len(lines):
            game = []
            boardsize = int(lines[i])
            i += 1

            while not lines[i].startswith('winner'):
                turn = int(lines[i])
                i += 1
                move = [int(x) for x in lines[i].split(',')]
                i += 1
                board = np.zeros((boardsize, boardsize), dtype=int)
                for y in range(boardsize):
                    row = lines[i].split(',')
                    for x in range(boardsize):
                        board[(x, y)] = int(row[x])
                    i += 1

                game.append((turn, move, board))

            winner = int(lines[i].split('=')[1])
            games.append((winner, game))

            i += 1

        return games


def encode_input_data(player, board, lst):
    for x in range(5):
        for y in range(5):
            # When position value is 0 (no move)
            if board[x, y] == 1:
                # Append 0 0 1
                if player == 1:
                    lst.extend([0, 0, 1])
                elif player == 2:
                    lst.extend([0, 1, 0])
                # When position value is 2 (player 2 move)
            elif board[x, y] == 2:
                # Append 0 1 0
                if player == 1:
                    lst.extend([0, 1, 0])
                elif player == 2:
                    lst.extend([0, 0, 1])
                # When position value is 0 (no player move yet)
            else:
                # Append 1 0 0
                lst.extend([1, 0, 0])
    return lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=10,
                        help='The size of the board.')

    parser.add_argument('--games', type=int, default=1,
                        help='The number of games to play.')

    parser.add_argument('--time', type=int, default=10,
                        help='The allowed time per move, in milliseconds.')

    parser.add_argument('--print-board', choices=['all', 'final'],
                        help='Show the board state, either every turn or only at the end.')

    parser.add_argument('--parallel', type=int,
                        help='Run multiple games in parallel. Only use for large-scale '
                             'testing.')

    parser.add_argument('--output',
                        help='Write training data to the given file.')

    parser.add_argument('--input',
                        help='Read training data from the given file.')

    parser.add_argument('objectives',
                        help='The name of a file containing the objective shapes. The file '
                             'should contain a rectangle with x on positions that should be '
                             'occupied, and dots on other positions. Separate objective shapes '
                             'should be separated by a blank line.')

    args = parser.parse_args()
    # cProfile.run('main(args)')
    main(args)
