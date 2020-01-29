import random
import numpy as np
import os
import tensorflow as tf
from dqn_agent import DQNAgent
from tqdm import tqdm
from snake import Snake, NUM_ACTIONS
import pickle


class DQNTrainer(object):
    def __init__(self,
                 n_episodes=100000,
                 epsilon=1,
                 epsilon_decay=0.9975,
                 min_epsilon=0.001,
                 max_steps=1000,
                 render_freq=100,
                 render_intermediate=True,
                 render_delay=50,
                 save_dir='save',
                 save_freq=100,
                 gamma=0.99,
                 batch_size=32,
                 min_replay_memory_size=1000,
                 replay_memory_size=10000,
                 seed=42
                 ):
        self._set_random_seed(seed)

        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.render_freq = render_freq
        self.render_intermediate = render_intermediate
        self.render_delay = render_delay
        self.save_dir = save_dir
        self.save_freq = save_freq

        self.agent = DQNAgent(
            gamma=gamma,
            batch_size=batch_size,
            min_replay_memory_size=min_replay_memory_size,
            replay_memory_size=replay_memory_size
        )
        self.env = Snake()
        self.current_episode = 0

    def _set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.set_random_seed(seed)

    def train(self):
        pbar = tqdm(initial=self.current_episode, total=self.n_episodes, unit='episodes')
        while self.current_episode < self.n_episodes:
            self.env.reset()
            current_state = self.env.state.embedded()

            done = False
            steps = 0
            while not done and steps < self.max_steps:
                if random.random() > self.epsilon:
                    input1, input2 = np.array([current_state[0]]), np.array([current_state[1]])
                    action = np.argmax(self.agent.get_q_values(input1, input2))
                else:
                    action = np.random.randint(NUM_ACTIONS)

                next_state, reward, done = self.env.step(action)

                self.agent.update_replay_memory(current_state, action, reward, next_state, done)
                self.agent.train()

                current_state = next_state
                steps += 1

            # decay epsilon
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.min_epsilon)

            # increase episode and save model, training info
            self.current_episode += 1
            if self.current_episode % self.save_freq == 0:
                self.save()
                print('episode:', self.current_episode, 'epsilon:', self.epsilon)

            # preview
            if self.render_intermediate and self.current_episode % self.render_freq == 0:
                self.play()

            # update pbar
            pbar.update(1)

    def play(self):
        self.env.reset()
        current_state = self.env.state.embedded()

        done = False
        steps = 0
        while not done and steps < self.max_steps:
            if random.random() > self.epsilon:
                input1, input2 = np.array([current_state[0]]), np.array([current_state[1]])
                action = np.argmax(self.agent.get_q_values(input1, input2))
            else:
                action = np.random.randint(NUM_ACTIONS)

            next_state, reward, done = self.env.step(action)
            self.env.render(delay=self.render_delay)

            current_state = next_state
            steps += 1

    def save(self):
        self.agent.save(self.save_dir+'/dqn_agent_{0}.h5'.format(self.current_episode))
        dic = {'current_episode': self.current_episode,
               'epsilon': self.epsilon}
        with open(self.save_dir+'/training_info_{0}.pkl'.format(self.current_episode), 'wb') as fout:
            pickle.dump(dic, fout)

    def load(self, episode):
        self.agent.load(self.save_dir+'/dqn_agent_{0}.h5'.format(episode))
        with open(self.save_dir+'/training_info_{0}.pkl'.format(episode), 'rb') as fin:
            dic = pickle.load(fin)
        self.current_episode = dic['current_episode']
        self.epsilon = dic['epsilon']


if __name__ == '__main__':
    trainer = DQNTrainer()
    trainer.load(2000)
    trainer.play()