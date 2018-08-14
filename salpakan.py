#!/usr/bin/env python3
import numpy as np
import gym
import random
import os
from keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, concatenate
from keras.models import Model
from keras.utils import to_categorical
from collections import deque
import envs

ENV_NAME = 'Salpakan-v0'
WEIGHTS_PATH = '.models/dqn_{}_weights.h5f'.format(ENV_NAME)
MEMORY = 1000
GAMMA = 0.95
SAMPLE_SIZE = 32


def create_model():
    state_input = Input(shape=(9, 8, 3))
    x_action_input = Input(shape=(9,))
    y_action_input = Input(shape=(8,))
    x2_action_input = Input(shape=(9,))
    y2_action_input = Input(shape=(8,))

    conv_1 = Conv2D(64, 3, padding='same')(state_input)
    conv1_a = LeakyReLU()(conv_1)
    flat_layer = Flatten()(conv1_a)
    dense_1 = Dense(512)(flat_layer)
    state_output = LeakyReLU()(dense_1)
    merged_layer = concatenate([state_output, x_action_input, y_action_input, x2_action_input, y2_action_input])
    dense_2 = Dense(512)(merged_layer)
    dense_2_a = LeakyReLU()(dense_2)
    output = Dense(1)(dense_2_a)

    model = Model([state_input, x_action_input, y_action_input, x2_action_input, y2_action_input],
                  output)
    model.compile('adam', loss='mse')
    model.summary()
    return model


model = create_model()

if os.path.isfile(WEIGHTS_PATH) and os.access(WEIGHTS_PATH, os.R_OK):
    model.load_weights(WEIGHTS_PATH)

memory = deque(maxlen=MEMORY)


def get_action(ob, training=True):
    possible_moves = env.possible_moves()

    if training and random.uniform(0, 1) < 0.1:
        action = random.choice(possible_moves)
    else:
        action = evaluate(ob, possible_moves)

    return action


def evaluate(ob, possible_moves):
    q_values = []

    ob = np.expand_dims(ob, 0)

    for move in possible_moves:
        x1, y1, x2, y2 = move

        x1 = to_categorical(x1, num_classes=9)
        y1 = to_categorical(y1, num_classes=8)
        x2 = to_categorical(x2, num_classes=9)
        y2 = to_categorical(y2, num_classes=8)

        x1 = np.expand_dims(x1, 0)
        y1 = np.expand_dims(y1, 0)
        x2 = np.expand_dims(x2, 0)
        y2 = np.expand_dims(y2, 0)

        inp = [ob, x1, y1, x2, y2]
        prediction = model.predict(inp)[0][0]
        q_values.append(prediction)
    index = np.argmax(q_values)
    return possible_moves[index]


def get_next_max_q(ob):
    possible_moves = env.possible_moves()
    q_values = []

    ob = np.expand_dims(ob, 0)

    for move in possible_moves:
        x1, y1, x2, y2 = move

        x1 = to_categorical(x1, num_classes=9)
        y1 = to_categorical(y1, num_classes=8)
        x2 = to_categorical(x2, num_classes=9)
        y2 = to_categorical(y2, num_classes=8)

        x1 = np.expand_dims(x1, 0)
        y1 = np.expand_dims(y1, 0)
        x2 = np.expand_dims(x2, 0)
        y2 = np.expand_dims(y2, 0)

        inp = [ob, x1, y1, x2, y2]
        prediction = model.predict(inp)[0][0]
        q_values.append(prediction)
    return np.max(q_values)


def remember(ob, action, next_ob, reward):
    memory.append((ob, action, next_ob, reward))


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

        model.fit(inp, [q_value], verbose=0)
        q_history.append(q_value)

    print('Max: {} Min: {} Mean: {}'.format(np.max(q_history), np.min(q_history), np.mean(q_history)))


# train
env = gym.make(ENV_NAME)
while True:

    ob = env.reset()
    a = get_action(ob)
    ob, reward, done, info = env.step(a)
    while True:
        possible_moves = env.possible_moves()
        turn = env.get_turn()
        a = get_action(ob, training=turn == 0)
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
