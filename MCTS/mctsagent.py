# Put your name and student ID here before submitting!
# Dimitris Christodoulou (5141761)
import copy
import math
import random

import game
import numpy
from random_agent import RandomAgent


def player_positions_count(pid, board):
    return len(numpy.argwhere(board.board == pid))


def determine_current_player(players, board):
    p1 = player_positions_count(players[0].id, board)
    p2 = player_positions_count(players[1].id, board)

    if p1 > p2:
        return players[1].id
    elif p1 == p2:
        return players[0].id


def ucb1(wins_n, total_plays_n, c, total_plays_parent_n):
    return float(wins_n) / float(total_plays_n) + c * math.sqrt(math.log(total_plays_parent_n) / float(total_plays_n))


def select(copied_board, tree_root):
    # If the tree is empty, create the root node and select it (return it)
    if tree_root is None:
        tree_root = Node(None, None, copied_board, 2)
        return tree_root
    else:
        temp_node = tree_root
        while temp_node.children is not None and len(temp_node.children) != 0:
            if temp_node.player_id == 1:
                # 1 is our agent so we need the next move to be min (for the other player)
                next_node = temp_node.get_child_with_lowest_ucb()
            else:
                # 2 is the other agent so we need the next move to be max (for our player)
                next_node = temp_node.get_child_with_highest_ucb()

            if next_node == temp_node:
                break
            else:
                temp_node = next_node

        return temp_node


def expand(selected_node, objectives, players):
    temp_players = players
    if selected_node.player_id == players[1].id:
        temp_players = [RandomAgent(2), RandomAgent(1)]

    temp_copied_board = copy.deepcopy(selected_node.board_at_node)
    new_game = game.Game.from_board(temp_copied_board, objectives, temp_players, None)

    if selected_node.played_pos is not None:
        if new_game.victory(selected_node.played_pos, selected_node.player_id) or \
                new_game.victory(selected_node.played_pos, 3-selected_node.player_id):
            # We have a victory just by placing the first move, so no need to simulate anything,
            # this is a leaf node
            return selected_node

    all_pos = temp_copied_board.free_positions()
    all_children = list()

    for (i, j) in all_pos:
        temp_copied_board = copy.deepcopy(selected_node.board_at_node)
        temp_copied_board.place((i, j), 3 - selected_node.player_id)
        child = selected_node.create_child(temp_copied_board, (i, j), 3 - selected_node.player_id)
        all_children.append(child)

    if all_children:
        # Our list is not empty so we choose a random element
        return random.choice(all_children)
    else:
        # Our list is empty. Since we already check for victories and eliminate children,
        # this is a draw so we return the same node
        return selected_node


def simulate(new_node, board, objectives, players):
    temp_players = players
    if new_node.player_id == players[1].id:
        temp_players = [RandomAgent(2), RandomAgent(1)]

    temp_copied_board = copy.deepcopy(new_node.board_at_node)
    new_game = game.Game.from_board(temp_copied_board, objectives, temp_players, None)

    if new_game.victory(new_node.played_pos, new_node.player_id):
        # We have a victory just by placing the first move, so no need to simulate anything,
        # this is a leaf node
        winner = new_node.player_id
    else:
        winner = new_game.play()
        if winner is not None:
            winner = winner.id

    new_node.num_selections += 1

    if winner is not None:
        if winner == players[0].id:
            score = 1
        else:
            score = -1
    else:
        score = 0.5

    return score


def backpropagate(new_node, score):
    temp_node = new_node
    pid = new_node.player_id

    while temp_node.parent is not None:
        temp_node = temp_node.parent

        if temp_node.player_id == pid:
            temp_node.score += score

        temp_node.num_selections += 1

        if temp_node.parent is not None:
            temp_node.ucb_score = temp_node.ucb1(temp_node.score, temp_node.num_selections,
                                                 temp_node.parent.num_selections)
        else:
            temp_node.ucb_score = temp_node.ucb1(temp_node.score, temp_node.num_selections, 0)
        temp_node.score_str = str(temp_node.ucb_score)


class Agent:
    def __init__(self, iterations, id):
        self.iterations = iterations
        self.id = id

    def make_move(self, game):
        i = 0
        tree_root = None
        copied_board = copy.deepcopy(game.board)

        # Create 2 random agent players to play the fake game
        new_players = [RandomAgent(1), RandomAgent(2)]

        # run until time is up
        while i < self.iterations:
            # do MCTS on top of RandomAgent here

            # For each iteration, perform the following steps
            # 1 - Select a node
            # 2 - Expand it
            # 3 - Simulate a game with the move from this expansion and produce a score according to the game
            # 4 - Backpropagate this score to the node we selected

            selected_node = select(copied_board, tree_root)

            if tree_root is None:
                tree_root = selected_node

            new_node = expand(selected_node, game.objectives, new_players)
            score = simulate(new_node, copied_board, game.objectives, new_players)
            backpropagate(new_node, score)
            i += 1

        # After finishing, check all 1st level children for the best UCB score
        # pptree.print_tree(tree_root, horizontal=True, nameattr='score_str')
        return tree_root.get_child_with_highest_ucb().played_pos

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

        if parent is not None:
            self.ucb_score = self.ucb1(self.score, self.num_selections, parent, c=0.5)  # UCB score of this node
            self.parent = parent

    def get_child_with_lowest_ucb(self):
        return self.get_lowest_ucb_node(self.children)

    def get_child_with_highest_ucb(self):
        return self.get_highest_ucb_node(self.children)

    def get_child_or_self_with_lowest_ucb(self):
        nodes_to_visit = self.children.copy()
        nodes_to_visit.append(self)

        return self.get_lowest_ucb_node(nodes_to_visit)

    def get_lowest_ucb_node(self, nodes_to_visit):
        min_ucb = math.inf
        all_min_children = list()

        for node in nodes_to_visit:
            if node.ucb_score < min_ucb:
                all_min_children = list()
                all_min_children.append(node)
                min_ucb = node.ucb_score
            elif node.ucb_score == min_ucb:
                all_min_children.append(node)

        return random.choice(all_min_children)

    def get_highest_ucb_node(self, nodes_to_visit):
        max_ucb = -math.inf
        all_max_children = list()

        for node in nodes_to_visit:
            if node.ucb_score > max_ucb:
                all_max_children = list()
                all_max_children.append(node)
                max_ucb = node.ucb_score
            elif node.ucb_score == max_ucb:
                all_max_children.append(node)

        return random.choice(all_max_children)

    def ucb1(self, wins_n, total_plays_n, total_plays_parent_n, c=0.5):
        if total_plays_n != 0 and total_plays_parent_n != 0:
            return float(wins_n) / float(total_plays_n) + c * math.sqrt(
                math.log(total_plays_parent_n) / float(total_plays_n))
        else:
            return math.inf

    def create_child(self, board_state, pos, current_player):
        child = Node(pos, self, board_state, current_player)
        self.children.append(child)
        return child
