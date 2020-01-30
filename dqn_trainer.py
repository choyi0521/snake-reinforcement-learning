import random
import numpy as np
import os
import tensorflow as tf
from dqn_agent import DQNAgent
from tqdm import tqdm
from snake import Snake, NUM_ACTIONS
import pickle
from collections import deque
import os


class DQNTrainer(object):
    def __init__(self,
                 n_episodes=30000,
                 epsilon=1,
                 min_epsilon=0.1,
                 exploration_ratio=0.5,
                 max_steps=2000,
                 render_freq=100,
                 enable_render=True,
                 render_fps=10,
                 summary_freq=50,
                 enable_summary=True,
                 save_dir='checkpoints',
                 enable_save=True,
                 save_freq=1000,
                 gamma=0.99,
                 batch_size=64,
                 min_replay_memory_size=1000,
                 replay_memory_size=10000,
                 seed=42
                 ):
        self._set_random_seed(seed)

        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.exploration_ratio = exploration_ratio
        self.render_freq = render_freq
        self.enable_render = enable_render
        self.render_fps = render_fps
        self.enable_summary = enable_summary
        self.summary_freq = summary_freq
        self.save_dir = save_dir
        self.enable_save = enable_save
        self.save_freq = save_freq

        if enable_save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.agent = DQNAgent(
            gamma=gamma,
            batch_size=batch_size,
            min_replay_memory_size=min_replay_memory_size,
            replay_memory_size=replay_memory_size
        )
        self.env = Snake()
        self.summary = deque(maxlen=summary_freq)
        self.current_episode = 0
        self.epsilon_decay = (epsilon-min_epsilon)/(exploration_ratio*n_episodes)

    def _set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.set_random_seed(seed)

    def train(self):
        pbar = tqdm(initial=self.current_episode, total=self.n_episodes, unit='episodes')
        while self.current_episode < self.n_episodes:
            current_state = self.env.reset()

            done = False
            steps = 0
            loss_list = []
            while not done and steps < self.max_steps:
                if random.random() > self.epsilon:
                    action = np.argmax(self.agent.get_q_values(np.array([current_state])))
                else:
                    action = np.random.randint(NUM_ACTIONS)

                next_state, reward, done = self.env.step(action)

                self.agent.update_replay_memory(current_state, action, reward, next_state, done)
                loss = self.agent.train()
                if loss is not None:
                    loss_list.append(loss)

                current_state = next_state
                steps += 1

            self.summary.append((
                self.env.state.get_length(),
                sum(loss_list)/len(loss_list) if len(loss_list)>0 else 0
            ))

            # decay epsilon
            self.epsilon = max(self.epsilon-self.epsilon_decay, self.min_epsilon)

            self.current_episode += 1
            if self.enable_summary and self.current_episode % self.summary_freq == 0:
                max_length = 0
                avg_length = 0
                avg_loss = 0
                for x in self.summary:
                    max_length = max(max_length, x[0])
                    avg_length += x[0]
                    avg_loss += x[1]
                avg_length /= self.summary_freq
                avg_loss /= self.summary_freq

                s = 'episode: {}, maximum length: {}, average length: {}, average loss: {}, epsilon: {}'.format(
                    self.current_episode,
                    max_length,
                    avg_length,
                    avg_loss,
                    self.epsilon
                )
                with open('log.txt', 'a') as fout:
                    fout.write(s+'\n')
                print(s)

            # save model, training info
            if self.enable_save and self.current_episode % self.save_freq == 0:
                self.save()

            # update pbar
            pbar.update(1)

            # preview
            if self.enable_render and self.current_episode % self.render_freq == 0:
                self.preview()

    def preview(self, disable_exploration=False):
        current_state = self.env.reset()

        done = False
        steps = 0
        while not done and steps < self.max_steps:
            if disable_exploration or random.random() > self.epsilon:
                action = np.argmax(self.agent.get_q_values(np.array([current_state])))
            else:
                action = np.random.randint(NUM_ACTIONS)

            next_state, reward, done = self.env.step(action)
            self.env.render(fps=self.render_fps)

            current_state = next_state
            steps += 1

    def save(self):
        self.agent.save(self.save_dir+'/dqn_agent_{0}.h5'.format(self.current_episode))
        dic = {
            'current_episode': self.current_episode,
            'epsilon': self.epsilon,
            'replay_memory': self.agent.replay_memory,
            'summary': self.summary
        }
        with open(self.save_dir+'/training_info_{0}.pkl'.format(self.current_episode), 'wb') as fout:
            pickle.dump(dic, fout)

    def load(self, episode):
        self.agent.load(self.save_dir+'/dqn_agent_{0}.h5'.format(episode))
        with open(self.save_dir+'/training_info_{0}.pkl'.format(episode), 'rb') as fin:
            dic = pickle.load(fin)
        self.current_episode = dic['current_episode']
        self.epsilon = dic['epsilon']
        self.agent.replay_memory = dic['replay_memory']
        self.summary = dic['summary']


if __name__ == '__main__':
    trainer = DQNTrainer()
    trainer.load(15000)
    #trainer.train()
    trainer.preview(disable_exploration=True)