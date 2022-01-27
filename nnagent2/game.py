from board import Board

import numpy as np

class Game:
    def __init__(self, board, objectives, players, print_board, real_game=0, game_file = None):
        self.board = board
        self.objectives = objectives
        self.players = players
        self.print_board = print_board
        self.real_game = real_game
        self.game_file = game_file


    @classmethod
    def new(cls, boardsize, objectives, players, print_board, real_game = 0):
        board = Board(boardsize)
        return cls(board, objectives, players, print_board, real_game)

    @classmethod
    def from_board(cls, board, objectives, players, print_board):
        return cls(board, objectives, players, print_board)

    def play(self, output = None):
        draw = False
        do_write = self.real_game ==1
        if do_write:
            file1 = open("games.txt", "a")
            file1.write("5\n")
        while not draw:

            for player in self.players:
                if self.board.full():
                    draw = True
                    if do_write:
                        file1.write("winner=0\n")
                    break

                pos = player.make_move(self)
                if not self.board.is_free(pos):
                    print(f'Illegal move attempted!'
                          f'Position ({pos[0]}, {pos[1]}) is not free.')
                    return None
                if do_write:
                    file1.write(str(player.id)+"\n")
                    file1.write(str(pos[0]) + "," + str(pos[1]) + "\n")

                self.board.place(pos, player.id)
                if do_write:
                    for x in range(5):
                        for y in range(5):
                            file1.write(str(self.board.board[y][x]))
                            if y != 4:
                                file1.write(",")
                        file1.write("\n")

                if output:
                    self.output_state(player, pos, output)
                if self.print_board:
                    print(self.board)

                if self.victory(pos, player.id):
                    if do_write:
                        file1.write("winner=" + str(player.id) + "\n")
                    return player
        if do_write == 1:
            file1.close()
        return None

    # tests whether player with given id has made the given shape by placing
    # a marker at hint position
    def victory(self, hint, playerid):
        xh, yh = hint

        i = 0
        for shape in self.objectives:
            i += 1
            for xo in range(shape.shape[0]):
                for yo in range(shape.shape[1]):
                    if not shape[xo, yo]:
                        continue

                    if xo > xh or yo > yh:
                        continue

                    if (shape.shape[0] - xo > self.board.shape[0] - xh) or \
                       (shape.shape[1] - yo > self.board.shape[1] - yh):
                        continue

                    fits = True
                    for x in range(shape.shape[0]):
                        for y in range(shape.shape[1]):
                            if not shape[x, y]:
                                continue

                            pos = (xh - xo + x, yh - yo + y)
                            if self.board.value(pos) != playerid:
                                fits = False
                                break

                        if not fits:
                            break

                    if fits:
                        return True

        return False

    def output_state(self, player, move, file):
        print(player.id, file = file)
        print(f'{move[0]},{move[1]}', file = file)
        self.board.output_state(file)

    def print_result(self, player = None):
        if player == None:
            print('Draw!')
        else:
            print(f'{player} wins!')
        print(self.board, flush = True)
