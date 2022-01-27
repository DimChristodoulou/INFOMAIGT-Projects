import random, copy
import numpy as np

def score_state(board, objectives, playerid):
        total_score = 0
        for objective in objectives:
            total_score += score_shape(board.board, objective, playerid)

        return total_score

def score_shape(board, shape, playerid):
    total_score = 0
    shape_count = np.count_nonzero(shape)
    for x in range(board.shape[0] - shape.shape[0] + 1):
        for y in range(board.shape[1] - shape.shape[1] + 1):
            count = 0
            impossible = False
            player = 0
            for xx in range(shape.shape[0]):
                if impossible:
                    break

                for yy in range(shape.shape[1]):
                    if not shape[xx][yy]:
                        continue

                    if board[x + xx][y + yy] == 0:
                        continue

                    if player == 0:
                        player = board[x + xx][y + yy]
                    elif player != board[x + xx][y + yy]:
                        impossible = True
                        break

                    count += 1

            if not impossible and count > 0:
                if player == playerid:
                    total_score += score(count, shape_count)
                else:
                    total_score -= score(count, shape_count)

    return total_score

def score2(count, goal):
    if goal == count:
        return 32
    if goal - count == 1:
        return 1
    return 0

def score(count, goal):
    return 2**(8 - (goal - count) * 4)

class ScoreAgent:
    def __init__(self, id):
        self.id = id

    def make_move(self, game):
        return self.best_move(game)

    def best_move(self, game):
        options = game.board.free_positions()

        best_moves = []
        best_score = float('-inf')
        for move in options:
            board = copy.deepcopy(game.board)
            board.place(move, self.id)

            move_score = score_state(board, game.objectives, self.id)
            if move_score > best_score:
                best_moves = [move]
                best_score = move_score
            elif move_score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def __str__(self):
        return f'Player {self.id} (BaseAgent)'
