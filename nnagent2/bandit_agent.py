import game as g
from random_agent import RandomAgent

import random, copy, time
import numpy as np

class BanditAgent:
    def __init__(self, iterations, id):
        self.epsilon = 0.3
        self.iterations = iterations
        self.id = id

    def make_move(self, game):
        self.initialise_state(game.board)

        # run until time is up
        i = 0
        while i < self.iterations:
            if random.random() < self.epsilon:
                self.explore(game)
            else:
                self.exploit(game)
            i += 1

        return self.best_move(game.board)

    # we do optimistic initialisation
    def initialise_state(self, board):
        # for each possible move, self.state stores the number of times it
        # resulted in victory, and the total number of times it was explored
        self.state = np.stack([
                                np.full(board.shape, 2),
                                np.ones(board.shape)
                              ],
                              axis = -1)

    def explore(self, game):
        self.simulate(game.board.random_free(), game)

    def exploit(self, game):
        move = self.best_move(game.board)
        self.simulate(move, game)

    def best_move(self, board):
        best = [(-1, -1)]
        best_score = -1

        for x in range(self.state.shape[0]):
            for y in range(self.state.shape[1]):
                if not board.is_free((x, y)):
                    continue

                score = self.state[x][y][0] / self.state[x][y][1]
                if score > best_score:
                    best = [(x, y)]
                    best_score = score
                elif score == best_score:
                    best.append((x, y))

        return random.choice(best)

    # simulate from the given move, using random agents
    def simulate(self, move, game):
        x, y = move
        self.state[x][y][1] += 1

        temp_board = copy.deepcopy(game.board)
        temp_board.place(move, self.id)

        # we just simulated our move, so the other player goes first
        player1 = RandomAgent(3 - self.id)
        player2 = RandomAgent(self.id)

        simulation = g.Game.from_board(temp_board,
                                       game.objectives,
                                       [player1, player2],
                                       False)
        # winning move
        if simulation.victory(move, self.id):
            self.state[x][y][0] += 1
            return

        result = simulation.play()

        # we won the simulation
        if result == player2:
            self.state[x][y][0] += 1
        # the simulation was a draw
        elif result == None:
            self.state[x][y][0] += 0.5

    def __str__(self):
        return f'Player {self.id} (BanditAgent[e={self.epsilon}])'
