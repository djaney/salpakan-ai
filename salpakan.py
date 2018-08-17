#!/usr/bin/env python3
import numpy as np
import gym
import random
import os
import math
from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, concatenate
from keras.models import Model
from keras.utils import to_categorical
from collections import deque
import envs

ENV_NAME = 'Salpakan-v0'
WEIGHTS_PATH = '.models/dqn_{}_weights.h5f'.format(ENV_NAME)
MEMORY = 50000
GAMMA = 0.95
SAMPLE_SIZE = 32
EXPLORE_RATE = 0.1


def create_model():
    state_input = Input(shape=(9, 8, 5))
    action_input = Input(shape=(4,))

    conv_1 = Conv2D(100, 2)(state_input)
    conv1_a = LeakyReLU()(conv_1)
    flat_layer = Flatten()(conv1_a)
    dense_1 = Dense(512)(flat_layer)
    state_output = LeakyReLU()(dense_1)
    merged_layer = concatenate([state_output, action_input])
    dense_2 = Dense(512)(merged_layer)
    dense_2_a = LeakyReLU()(dense_2)
    output = Dense(1)(dense_2_a)

    model = Model([state_input, action_input], output)
    model.compile('adam', loss='mse')
    model.summary()
    return model


model = create_model()

if os.path.isfile(WEIGHTS_PATH) and os.access(WEIGHTS_PATH, os.R_OK):
    model.load_weights(WEIGHTS_PATH)

memory = deque(maxlen=MEMORY)
wins = deque(maxlen=10)
env = gym.make(ENV_NAME)


def get_action(ob, training=True):
    moves = env.possible_moves()

    if len(moves) == 0:
        return None

    if training and random.uniform(0, 1) < EXPLORE_RATE:
        action = random.choice(moves)
    else:
        action = evaluate(ob, moves)

    return action


def evaluate(observations, moves):
    q_values = []

    observations = np.expand_dims(observations, 0)

    for move in moves:
        x1, y1, x2, y2 = move

        x1 = to_categorical(x1, num_classes=9)
        y1 = to_categorical(y1, num_classes=8)
        x2 = to_categorical(x2, num_classes=9)
        y2 = to_categorical(y2, num_classes=8)

        x1 = np.expand_dims(x1, 0)
        y1 = np.expand_dims(y1, 0)
        x2 = np.expand_dims(x2, 0)
        y2 = np.expand_dims(y2, 0)

        inp = [observations, x1, y1, x2, y2]
        inp = crop_input(inp)
        prediction = model.predict(inp)[0][0]
        q_values.append(prediction)

    if len(q_values) == 0:
        return None

    index = np.argmax(q_values)
    return moves[index]


def get_next_max_q(observations):
    moves = env.possible_moves()
    q_values = []

    observations = np.expand_dims(observations, 0)

    for move in moves:
        x1, y1, x2, y2 = move

        x1 = to_categorical(x1, num_classes=9)
        y1 = to_categorical(y1, num_classes=8)
        x2 = to_categorical(x2, num_classes=9)
        y2 = to_categorical(y2, num_classes=8)

        x1 = np.expand_dims(x1, 0)
        y1 = np.expand_dims(y1, 0)
        x2 = np.expand_dims(x2, 0)
        y2 = np.expand_dims(y2, 0)

        inp = [observations, x1, y1, x2, y2]
        inp = crop_input(inp)
        prediction = model.predict(inp)[0][0]
        q_values.append(prediction)

    if len(q_values) == 0:
        return 0

    return np.max(q_values)


def remember(ob, action, next_ob, reward):
    memory.append((ob, action, next_ob, reward))


def crop_input(inp):
    board_state = inp[0]
    x = np.argmax(inp[1][0])
    y = np.argmax(inp[2][0])
    _x = np.argmax(inp[3][0])
    _y = np.argmax(inp[4][0])
    width = 8
    height = 7
    center_x = math.floor(width / 2)
    center_y = math.floor(height / 2)
    shift_x = center_x - x
    shift_y = center_y - y

    zero_pad = (0, 0)

    shifted = np.copy(board_state)

    # move right
    if shift_x > 0:
        shifted = np.pad(shifted, (zero_pad, (shift_x, 0), zero_pad, zero_pad), mode='constant')[:, :-shift_x, :, :]
    # move left
    if shift_x < 0:
        shifted = np.pad(shifted, (zero_pad, (0, -shift_x), zero_pad, zero_pad), mode='constant')[:, -shift_x:, :, :]
    # move down
    if shift_y > 0:
        shifted = np.pad(shifted, (zero_pad, zero_pad, (shift_y, 0), zero_pad), mode='constant')[:, :, :-shift_y, :]
    # move up
    if shift_y < 0:
        shifted = np.pad(shifted, (zero_pad, zero_pad, (0, -shift_y), zero_pad), mode='constant')[:, :, -shift_y:, :]

    delta_x = _x - x
    delta_y = _y - y

    direction = None
    if delta_x < 0 and delta_y == 0:
        direction = 0
    elif delta_x > 0 and delta_y == 0:
        direction = 1
    elif delta_y < 0 and delta_x == 0:
        direction = 2
    elif delta_y > 0 and delta_x == 0:
        direction = 3

    direction = np.expand_dims(to_categorical(direction, num_classes=4), 0)

    new_inp = [shifted, direction]

    return new_inp


def train():
    if SAMPLE_SIZE > len(memory):
        return

    sample_moves = random.sample(memory, SAMPLE_SIZE)
    q_history = []
    for move in sample_moves:
        ob, a, next_ob, reward = move
        # next q is enemy's best move minus your reward
        next_q = get_next_max_q(next_ob)
        q_value = reward + GAMMA * -next_q

        x1, y1, x2, y2 = a

        ob = np.expand_dims(ob, 0)
        x1 = to_categorical(x1, num_classes=9)
        y1 = to_categorical(y1, num_classes=8)
        x2 = to_categorical(x2, num_classes=9)
        y2 = to_categorical(y2, num_classes=8)

        x1 = np.expand_dims(x1, 0)
        y1 = np.expand_dims(y1, 0)
        x2 = np.expand_dims(x2, 0)
        y2 = np.expand_dims(y2, 0)

        inp = [ob, x1, y1, x2, y2]
        inp = crop_input(inp)
        model.fit(inp, [q_value], verbose=0)
        q_history.append(q_value)
    if env.game.winner == 0:
        wins.append(1)
    elif env.game.winner == 1:
        wins.append(0)
    print('Max: {:.4f} Min: {:.4f} Mean: {:.4f}, Win: {}, WL: {:.4f}'
          .format(np.max(q_history), np.min(q_history), np.mean(q_history), env.game.winner, np.mean(wins)))


# train
while True:

    ob = env.reset()
    a = get_action(ob)
    ob, reward, done, info = env.step(a)
    while True:
        possible_moves = env.possible_moves()
        turn = env.get_turn()
        a = get_action(ob, training=turn == 0)

        if a is None:
            break  # no more action

        next_ob, next_reward, done, info = env.step(a)

        if turn == 0:
            remember(ob, a, next_ob, reward)

        ob = next_ob
        reward = next_reward

        env.render()

        if done:
            break

    # think
    train()
    model.save_weights(WEIGHTS_PATH)
