from gym import Env, spaces
import numpy as np
from .salpakan_game import SalpakanGame, Renderer, \
    MOVE_NORMAL, MOVE_CAPTURE, MOVE_CAPTURE_LOSE, MOVE_WIN, MOVE_LOSE, TROOP_SPY, TROOP_FLAG, TROOP_PRIVATE

OBSERVATION_SHAPE = (9, 8, 5)
MAX_STEPS = 1000


def _troop_only(item, troop):
    return 1 if item == troop else 0


def _troop_except(troop, class_num):
    exceptions = TROOP_SPY, TROOP_PRIVATE, TROOP_FLAG
    return troop / class_num if troop not in exceptions else 0


def _troop_normalize_state(troop, class_num):
    return troop / class_num


class SalpakanEnv(Env):

    def __init__(self):

        self.observation_space = spaces.Box(low=0, high=1, shape=OBSERVATION_SHAPE, dtype=np.float16)
        self.action_space = spaces.Discrete(8 * 9 * 4 + 1)
        self.game = None
        self.view = None
        self.canvas = None
        self.steps = 0
        self.renderer = Renderer()
        self.done = False

    def step(self, action):
        move_type, me, him = self.game.move(action)
        ob = self._get_state()
        done = self.game.winner is not None or self.steps > MAX_STEPS
        self.done = self.done or done
        self.steps += 1

        if move_type == MOVE_NORMAL:
            reward = -10
        elif move_type == MOVE_CAPTURE:
            reward = him
        elif move_type == MOVE_CAPTURE_LOSE:
            reward = 0.5  # reward for trying
        elif move_type == MOVE_WIN:
            reward = 100
        elif move_type == MOVE_LOSE:
            reward = -100
        else:
            reward = 0

        return ob, reward, self.done, {}

    def reset(self):
        self.done = False
        self.steps = 0
        self.game = SalpakanGame()
        return self._get_state()

    def render(self, mode='human'):
        self.renderer.render(self.game, self._get_state())

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)

    def _get_state(self):

        observation = np.zeros(shape=OBSERVATION_SHAPE)

        board = self.game.get_board()

        my_troops = np.clip(board[:, :, 0], 0, None)
        enemy_troops = np.clip(board[:, :, 0] * -1, 0, None)

        v_troop_adjust = np.vectorize(_troop_normalize_state)
        v_troop_only = np.vectorize(_troop_only)
        v_troop_except = np.vectorize(_troop_except)

        # my units
        observation[:, :, 0] = v_troop_except(my_troops, 16)
        # enemy perception
        observation[:, :, 1] = v_troop_adjust(np.clip(enemy_troops, 0, 1) * board[:, :, 1], 16)
        # my troops
        observation[:, :, 2] = v_troop_only(my_troops, TROOP_SPY)
        observation[:, :, 3] = v_troop_only(my_troops, TROOP_PRIVATE)
        observation[:, :, 4] = v_troop_only(my_troops, TROOP_FLAG)

        return observation

    def possible_moves(self):
        valid_moves = []
        for y in range(8):
            for x in range(9):
                self._add_move_if_valid(valid_moves, x, y, x - 1, y)
                self._add_move_if_valid(valid_moves, x, y, x + 1, y)
                self._add_move_if_valid(valid_moves, x, y, x, y - 1)
                self._add_move_if_valid(valid_moves, x, y, x, y + 1)
        return valid_moves

    def get_turn(self):
        return self.game.turn

    def _add_move_if_valid(self, move_list, x, y, _x, _y):
        if self.game.is_valid_move((x, y, _x, _y)):
            move_list.append((x, y, _x, _y))
