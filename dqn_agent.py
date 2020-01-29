from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, multiply
from keras.models import Model
from keras.optimizers import RMSprop
from snake import BOARD_HEIGHT, BOARD_WIDTH, NUM_CHANNELS, NUM_ACTIONS
from collections import deque
import random
import numpy as np
import keras


class DQNAgent(object):
    def __init__(self, gamma, batch_size, min_replay_memory_size, replay_memory_size):
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_memory_size = min_replay_memory_size

        self.model = self._create_model()
        self.replay_memory = deque(maxlen=replay_memory_size)

    def _create_model(self):
        input_1 = Input(shape=(BOARD_HEIGHT, BOARD_WIDTH, NUM_CHANNELS))
        x1 = Conv2D(32, (3, 3), activation='relu')(input_1)
        x1 = Dropout(0.2)(x1)
        x1 = Conv2D(32, (3, 3), activation='relu')(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Flatten()(x1)
        x1 = Dense(32, activation='relu')(x1)

        input_2 = Input(shape=(BOARD_HEIGHT+BOARD_WIDTH+2,))
        x2 = Dense(32, activation='relu')(input_2)
        y = multiply([x1, x2])
        y = Dense(32, activation='relu')(y)
        output = Dense(NUM_ACTIONS)(y)
        model = Model(inputs=[input_1, input_2], outputs=output)
        model.compile(RMSprop(), 'MSE')
        return model

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def get_q_values(self, input_1, input_2):
        return self.model.predict({'input_1': input_1, 'input_2': input_2})

    def train(self):
        # guarantee the minimum number of samples
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # get current q values and next q values
        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = {'input_1': np.stack((sample[0][0] for sample in samples)),
                         'input_2': np.stack((sample[0][1] for sample in samples))}
        current_q_values = self.get_q_values(**current_input)
        next_input = {'input_1': np.stack((sample[3][0] for sample in samples)),
                      'input_2': np.stack((sample[3][1] for sample in samples))}
        next_q_values = self.get_q_values(**next_input)

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        # fit model
        self.model.fit(current_input, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = keras.models.load_model(filename)