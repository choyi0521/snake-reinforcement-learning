from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential
from snake import NUM_CHANNELS, NUM_ACTIONS
from collections import deque
import random
import numpy as np
import keras


class DQNAgent:
    def __init__(self, field_size, gamma, batch_size, min_replay_memory_size, replay_memory_size, target_update_freq):
        self.gamma = gamma
        self.field_height, self.field_width = field_size
        self.batch_size = batch_size
        self.min_replay_memory_size = min_replay_memory_size
        self.target_update_freq = target_update_freq

        self.model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

    def _create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=(self.field_height, self.field_width, NUM_CHANNELS), activation='relu'),
            Dropout(0.1),
            Conv2D(32, (3, 3), activation='relu'),
            Dropout(0.1),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(NUM_ACTIONS)
        ])
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def get_q_values(self, x):
        return self.model.predict(x)

    def train(self):
        # guarantee the minimum number of samples
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # get current q values and next q values
        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_values = self.model.predict(current_input)
        next_input = np.stack([sample[3] for sample in samples])
        next_q_values = self.target_model.predict(next_input)

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        # fit model
        hist = self.model.fit(current_input, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)
        loss = hist.history['loss'][0]
        return loss

    def increase_target_update_counter(self):
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save(self, model_filepath, target_model_filepath):
        self.model.save(model_filepath)
        self.target_model.save(target_model_filepath)

    def load(self, model_filepath, target_model_filepath):
        self.model = keras.models.load_model(model_filepath)
        self.target_model = keras.models.load_model(target_model_filepath)
