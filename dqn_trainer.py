import random
import numpy as np
import tensorflow as tf
from dqn_agent import DQNAgent
from tqdm import tqdm
from snake import Snake, NUM_ACTIONS
import pickle
import os
from summary import Summary
from level_loader import LevelLoader


class DQNTrainer(object):
    def __init__(self,
                 level_filepath,
                 episodes=40000,
                 initial_epsilon=1.,
                 min_epsilon=0.01,
                 exploration_ratio=0.8,
                 max_steps=10000,
                 render_freq=500,
                 enable_render=True,
                 render_fps=20,
                 save_dir='checkpoints',
                 enable_save=True,
                 save_freq=500,
                 gamma=0.99,
                 batch_size=64,
                 min_replay_memory_size=1000,
                 replay_memory_size=100000,
                 seed=42
                 ):
        self.set_random_seed(seed)

        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.exploration_ratio = exploration_ratio
        self.render_freq = render_freq
        self.enable_render = enable_render
        self.render_fps = render_fps
        self.save_dir = save_dir
        self.enable_save = enable_save
        self.save_freq = save_freq

        if enable_save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        level_loader = LevelLoader(level_filepath)

        self.agent = DQNAgent(
            level_loader.get_field_size(),
            gamma=gamma,
            batch_size=batch_size,
            min_replay_memory_size=min_replay_memory_size,
            replay_memory_size=replay_memory_size
        )
        self.env = Snake(level_loader)
        self.summary = Summary()
        self.current_episode = 0
        self.max_average_length = 0

        self.epsilon_decay = (initial_epsilon-min_epsilon)/(exploration_ratio*episodes)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.set_random_seed(seed)

    def train(self):
        pbar = tqdm(initial=self.current_episode, total=self.episodes, unit='episodes')
        while self.current_episode < self.episodes:
            current_state = self.env.reset()

            done = False
            steps = 0
            while not done and steps < self.max_steps:
                if random.random() > self.epsilon:
                    action = np.argmax(self.agent.get_q_values(np.array([current_state])))
                else:
                    action = np.random.randint(NUM_ACTIONS)

                next_state, reward, done = self.env.step(action)

                self.agent.update_replay_memory(current_state, action, reward, next_state, done)
                self.summary.add('loss', self.agent.train())

                current_state = next_state
                steps += 1

            self.summary.add('length', self.env.state.get_length())
            self.summary.add('reward', self.env.tot_reward)
            self.summary.add('steps', steps)

            # decay epsilon
            self.epsilon = max(self.epsilon-self.epsilon_decay, self.min_epsilon)

            self.current_episode += 1

            # save model, training info
            if self.enable_save and self.current_episode % self.save_freq == 0:
                self.save(str(self.current_episode))

                average_length = self.summary.get_average('length')
                if average_length > self.max_average_length:
                    self.max_average_length = average_length
                    self.save('best')
                    print('best model saved - average_length: {}'.format(average_length))

                self.summary.write(self.current_episode, self.epsilon)
                self.summary.clear()

            # update pbar
            pbar.update(1)

            # preview
            if self.enable_render and self.current_episode % self.render_freq == 0:
                self.preview(self.render_fps)

    def preview(self, render_fps, disable_exploration=False, save_dir=None):
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_state = self.env.reset()

        self.env.render(fps=render_fps)
        if save_dir is not None:
            self.env.save_image(save_path=save_dir+'/0.png')

        done = False
        steps = 0
        while not done and steps < self.max_steps:
            if disable_exploration or random.random() > self.epsilon:
                action = np.argmax(self.agent.get_q_values(np.array([current_state])))
            else:
                action = np.random.randint(NUM_ACTIONS)

            next_state, reward, done = self.env.step(action)
            current_state = next_state
            steps += 1

            self.env.render(fps=render_fps)
            if save_dir is not None:
                self.env.save_image(save_path=save_dir+'/{}.png'.format(steps))

        return self.env.state.get_length()

    def quit(self):
        self.env.quit()

    def save(self, name):
        self.agent.save(self.save_dir+'/model_{}.h5'.format(name))
        dic = {
            'current_episode': self.current_episode,
            'epsilon': self.epsilon,
            'replay_memory': self.agent.replay_memory,
            'summary': self.summary,
            'max_average_length': self.max_average_length
        }
        with open(self.save_dir+'/training_info_{}.pkl'.format(name), 'wb') as fout:
            pickle.dump(dic, fout)

    def load(self, name):
        self.agent.load(self.save_dir+'/model_{}.h5'.format(name))
        with open(self.save_dir+'/training_info_{}.pkl'.format(name), 'rb') as fin:
            dic = pickle.load(fin)
        self.current_episode = dic['current_episode']
        self.epsilon = dic['epsilon']
        self.agent.replay_memory = dic['replay_memory']
        self.summary = dic['summary']
        self.max_average_length = dic['max_average_length']
