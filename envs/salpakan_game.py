import numpy as np
import random
import math
import tkinter as tk

PIECE_CONF = [1, 2, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
WIDTH = 9
HEIGHT = 8
SPECIAL_MOVES_N = 1

MOVE_NORMAL = 1
MOVE_CAPTURE = 2
MOVE_CAPTURE_LOSE = 3
MOVE_WIN = 4
MOVE_PASS = 5
MOVE_INVALID = 6

CHANNEL_TROOPS = 0
CHANNEL_PERCEPTION = 1
CHANNEL_SPY_PERCEPTION = 2

PLAYER_1 = 0
PLAYER_2 = 1

TROOP_FLAG = 1
TROOP_SPY = 2
TROOP_PRIVATE = 3
TROOP_FIVE_STAR = 15


def _parse_move(move):
    direction = (move - SPECIAL_MOVES_N) % 4
    square_id = math.floor((move - SPECIAL_MOVES_N) / 4)
    x = square_id % HEIGHT
    y = math.floor(square_id / WIDTH)

    if direction == 0:
        _x = x - 1
        _y = y
    elif direction == 1:
        _x = x + 1
        _y = y
    elif direction == 2:
        _x = x
        _y = y - 1
    elif direction == 3:
        _x = x
        _y = y + 1
    else:
        raise Exception("Invalid direction")
    return square_id, x, y, _x, _y, direction


def _normalize_board(player, board):
    n_board = np.copy(board)
    if player == 1:
        n_board[:, :, CHANNEL_TROOPS] = n_board[:, :, CHANNEL_TROOPS] * -1
    return n_board


class SalpakanGame:
    def __init__(self):
        # Create board
        # width x height x channels
        # channels = troops, perception
        self.board = np.zeros((9, 8, 3))

        # generate troops
        self._generate_troops(0)
        self._generate_troops(1)

        self.turn = random.randint(0, 1)

        self.winner = None

    def _generate_troops(self, player):
        assert player == PLAYER_1 or player == PLAYER_2

        if player == 0:
            y_min = 0
            y_max = HEIGHT / 2 - 1
        else:
            y_min = HEIGHT / 2
            y_max = HEIGHT - 1

        for (i, p) in enumerate(PIECE_CONF):
            for r in range(p):
                for _ in range(36):
                    y = random.randint(y_min, y_max)
                    x = random.randint(0, WIDTH - 1)
                    # if no troop
                    if self.board[x][y][CHANNEL_TROOPS] == 0:
                        # set troops, negative if player 2
                        self.board[x][y][CHANNEL_TROOPS] = i + 1 if player == PLAYER_1 else (i + 1) * -1
                        self.board[x][y][CHANNEL_PERCEPTION] = 1
                        break

    def move(self, move):
        """
        :param move:
        :return:
        """
        player = self.turn

        if move != 0:
            x, y, _x, _y = move

            if not self._is_valid_move(player, (x, y), (_x, _y)):
                return MOVE_INVALID

            move_type = self._get_move_type(player, (x, y), (_x, _y))

            if move_type == MOVE_NORMAL:
                self.board[x][y], self.board[_x][_y] = np.copy(self.board[_x][_y]), np.copy(self.board[x][y])
            elif move_type == MOVE_CAPTURE:
                normalized_board = _normalize_board(player, self.board)
                me = normalized_board[x][y][CHANNEL_TROOPS]
                him = normalized_board[_x][_y][CHANNEL_TROOPS]

                if me == him:  # cancel out
                    win = 0
                elif me == TROOP_SPY and him != -TROOP_PRIVATE:  # spy captures
                    win = 1
                elif me == TROOP_PRIVATE and him == -TROOP_SPY:  # private captures spy
                    win = 1
                else:  # normal rank based clash
                    win = 1 if me > -him else -1
                    # if lost
                    if win < 0:
                        # and i was 5 star
                        if me == TROOP_FIVE_STAR:
                            # surely a spy
                            self.board[x][y][CHANNEL_SPY_PERCEPTION] = 1
                        else:
                            self.board[_x][_y][CHANNEL_SPY_PERCEPTION] = max(self.board[_x][_y][CHANNEL_SPY_PERCEPTION],
                                                                         me / 15)

                if win > 0:  # win
                    self.board[_x][_y] = self.board[x][y]
                    self.board[x][y] = self.board[x][y] * 0
                elif win < 0:  # lose
                    self.board[_x][_y][CHANNEL_PERCEPTION] = max(abs(self.board[x][y][CHANNEL_TROOPS]) + 1,
                                                                 self.board[_x][_y][CHANNEL_PERCEPTION])
                    self.board[x][y] = self.board[x][y] * 0

                    move_type = MOVE_CAPTURE_LOSE
                else:
                    self.board[_x][_y][CHANNEL_PERCEPTION] = self.board[x][y][CHANNEL_TROOPS]
                    self.board[x][y] = self.board[x][y] * 0
                    self.board[_x][_y] = self.board[_x][_y] * 0
                    move_type = MOVE_NORMAL
            elif move_type == MOVE_WIN:
                self.board[_x][_y] = self.board[x][y]
                self.board[x][y] = self.board[x][y] * 0
                self.winner = player
        else:
            move_type = MOVE_PASS

        # change turn
        self.turn = PLAYER_2 if self.turn == PLAYER_1 else PLAYER_1

        # return move type
        return move_type

    def get_board(self):
        return _normalize_board(self.turn, self.board)

    def _is_valid_move(self, player, src, destination):

        board = _normalize_board(player, self.board)

        if src[0] >= board.shape[0] or src[0] < 0:
            return False
        if destination[0] >= board.shape[0] or destination[0] < 0:
            return False
        if src[1] >= board.shape[1] or src[1] < 0:
            return False
        if destination[1] >= board.shape[1] or destination[1] < 0:
            return False

        # check if moving own piece
        if board[src[0]][src[1]][CHANNEL_TROOPS] <= 0:
            return False

        # check if destination not capturing own
        if board[destination[0]][destination[1]][CHANNEL_TROOPS] > 0:
            return False

        return True

    def _get_move_type(self, player, src, destination):
        board = _normalize_board(player, self.board)
        me = board[src[0]][src[1]][CHANNEL_TROOPS]
        him = board[destination[0]][destination[1]][CHANNEL_TROOPS]

        if me > 0 and him == -TROOP_FLAG:
            return MOVE_WIN
        elif me > 0 and 0 > him:
            return MOVE_CAPTURE
        else:
            return MOVE_NORMAL

    def is_valid_move(self, move):
        x, y, _x, _y = move
        return self._is_valid_move(self.turn, (x, y), (_x, _y))

    def generate_mask(self):
        mask = [1 if (self.is_valid_move(i)) else 0 for i in range(289)]
        return np.array(mask)


class Renderer:
    def __init__(self):

        self.font = "Arial 10 bold"

        self.width = 281
        self.height = 250

        self.x_tiles = 9
        self.y_tiles = 8

        self.tile_width = self.width / self.x_tiles
        self.tile_height = self.height / self.y_tiles

        self.view = None
        self.canvas = None

        window_width = self.width * 2
        window_height = self.height * 2

        self.view = tk.Tk()
        self.view.geometry('{}x{}'.format(window_width, window_height))
        self.view.resizable(width=False, height=False)

        self.canvas = self._create_canvas(self.view, (0, 0))
        self.canvas2 = self._create_canvas(self.view, (0, 1))
        self.canvas3 = self._create_canvas(self.view, (1, 0))
        self.canvas4 = self._create_canvas(self.view, (1, 1))

    def render(self, game, state):

        self._clear(self.canvas)
        self._draw_board(self.canvas)
        self._draw_pieces(self.canvas, game.board)

        if game.turn == 0:
            self._clear(self.canvas2)
            self._clear(self.canvas3)
            self._clear(self.canvas4)

            self._draw_board(self.canvas2)
            self._draw_channel(self.canvas2, state[:, :, 0])

            self._draw_board(self.canvas3)
            self._draw_channel(self.canvas3, state[:, :, 1])

            self._draw_board(self.canvas4)
            self._draw_channel(self.canvas4, state[:, :, 2])

        self.view.update_idletasks()

    def _create_canvas(self, parent, grid):
        canvas = tk.Canvas(parent, width=self.width, height=self.height, bg='white')
        canvas.grid(row=grid[0], column=grid[1])
        return canvas

    @staticmethod
    def _clear(canvas):

        # clear
        canvas.delete("all")

    def _draw_board(self, canvas):
        # add lines
        for i in range(self.x_tiles):
            canvas.create_line(self.tile_width * i, 0, self.tile_width * i, self.height)
        for i in range(self.y_tiles):
            canvas.create_line(0, self.tile_height * i, self.width, self.tile_height * i)

    def _draw_pieces(self, canvas, board):
        # Draw cells
        for x, col in enumerate(board):
            for y, cell in enumerate(col):
                x1 = self.tile_width * x
                y1 = self.tile_height * y
                x2 = self.tile_width * x + self.tile_width
                y2 = self.tile_height * y + self.tile_height
                # Draw pieces
                if cell[CHANNEL_TROOPS] != 0:
                    canvas.create_rectangle(x1, y1, x2, y2, fill='red' if cell[CHANNEL_TROOPS] > 0 else 'black')
                    canvas.create_text(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2,
                                       fill='white',
                                       font=self.font,
                                       text=str(int(cell[0])))

    def _draw_channel(self, canvas, board):
        # Draw cells
        for x, col in enumerate(board):
            for y, cell in enumerate(col):
                x1 = self.tile_width * x
                y1 = self.tile_height * y
                x2 = self.tile_width * x + self.tile_width
                y2 = self.tile_height * y + self.tile_height

                value = 255 - math.floor(cell * 255)
                hex_value = value.to_bytes(1, 'big').hex()
                canvas.create_rectangle(x1, y1, x2, y2, fill='#{0}{0}{0}'.format(hex_value))
                canvas.create_text(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2,
                                   fill='red',
                                   font=self.font,
                                   text='{:.1f}'.format(cell))

    def _draw_info_board(self, canvas, game):

        if game.winner == 0:
            winner_text = "Win!"
        elif game.winner == 1:
            winner_text = "Lose!"
        else:
            winner_text = "Playing..."

        canvas.create_text(self.width / 2, self.height / 2,
                           fill='red',
                           font=self.font,
                           text=winner_text)
