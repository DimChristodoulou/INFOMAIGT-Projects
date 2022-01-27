import copy
import math
import numpy
import random

from random_agent import RandomAgent

def count_agent_moves(board, pid):
    return numpy.argwhere(board.board == pid).size

class Agent:
    def __init__(self, iterations, id):
        self.id = id
        self.iterations = iterations

    def make_move(self, game):
        temp_board = copy.deepcopy(game.board)

        root_node = Node(None, None, temp_board, None)

        i = 0
        players = [RandomAgent(2), RandomAgent(1)]

        scores = dict()
        free_pos = temp_board.free_positions()

        for pos in free_pos:
            temp_board = copy.deepcopy(game.board)
            temp_board.place(tuple(pos), players[1].id)

            child = root_node.create_child(temp_board, pos, None)
            temp_game = game.from_board(temp_board, game.objectives, players, None)

            if temp_game.victory(tuple(pos), players[1].id):
                child.score = math.inf
            elif temp_game.victory(tuple(pos), players[0].id):
                child.score = -math.inf
            else:
                child.score = 0

                for pos2 in child.board_at_node.free_positions():
                    child_temp_board_p0 = copy.deepcopy(child.board_at_node)
                    child_temp_board_p1 = copy.deepcopy(child.board_at_node)

                    child_temp_board_p0.place(tuple(pos2), players[0].id)
                    temp_game_p0 = game.from_board(child_temp_board_p0, game.objectives, players, None)

                    if temp_game_p0.victory(tuple(pos2), players[0].id):
                        child.score -= 1

                    child_temp_board_p1.place(tuple(pos2), players[1].id)
                    temp_game_p1 = game.from_board(child_temp_board_p1, game.objectives, players, None)

                    if temp_game_p1.victory(tuple(pos2), players[1].id):
                        child.score += 1

        max = -math.inf
        for child in root_node.children:
            if child.score > max:
                max = child.score
                maxpos = child.played_pos

        return maxpos


    def __str__(self):
        return f'Player {self.id} (your agent)'


class Node:
    # Tree related information
    children = list()
    score = 0
    num_selections = 0
    score_str = ""

    parent = None

    # Game Related Information
    played_pos = tuple()
    board_at_node = None  # Board object
    player_id = None   # Pid

    def __init__(self, pos, parent=None, board=None, player_id=None):
        self.children = list()
        self.score = 0  # Number of winning games after playing this move
        self.num_selections = 0  # Number of total games after playgin this move

        self.played_pos = pos
        self.board_at_node = board
        self.player_id = player_id

    def create_child(self, board_state, pos, current_player):
        child = Node(pos, self, board_state, current_player)
        self.children.append(child)
        return child