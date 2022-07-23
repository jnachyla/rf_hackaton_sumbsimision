import gym
import random
from tensorflow.keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import numpy as np
env = gym.make('CartPole-v0').env
env.seed(0)
np.random.seed(0)

'''
The original code is of the base network: https://github.com/shivaverma/OpenAIGym/blob/master/cart-pole/CartPole-v0.py

'''
class DeepQLearningModel:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.lr = 0.0001
        self.memory_model = deque(maxlen=10000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        print(model.summary())
        return model

    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def learn(self):

        sampled_batch = random.sample(self.memory_model, self.batch_size)
        states = np.array([i[0] for i in sampled_batch])
        actions = np.array([i[1] for i in sampled_batch])
        rewards = np.array([i[2] for i in sampled_batch])
        next_states = np.array([i[3] for i in sampled_batch])
        dones = np.array([i[4] for i in sampled_batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        q_values = self.q_learning_equation(dones, next_states, rewards)
        ind = np.array([i for i in range(self.batch_size)])
        self.model.predict_on_batch(states)[[ind], [actions]] = q_values

        self.model.fit(states, self.model.predict_on_batch(states), epochs=1, verbose=0)
        self.eps_decay()

    def eps_decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def q_learning_equation(self, dones, next_states, rewards):
        return rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)


def train_dqn(episodes_num, max_steps):

    rewards = []
    dqn_model = DeepQLearningModel(env.action_space.n, env.observation_space.shape[0])
    for e in range(episodes_num):
        state = env.reset()
        state = np.reshape(state, (1, 4))
        sum_of_rewards = 0

        for i in range(max_steps):
            env.render()
            action = dqn_model.get_action(state)
            next_state, reward, done, _ = env.step(action)
            sum_of_rewards += reward
            next_state = np.reshape(next_state, (1, 4))
            dqn_model.memory_model.append((state, action, reward, next_state, done))
            state = next_state
            dqn_model.learn()
            if done:
                break
        rewards.append(sum_of_rewards)
    return rewards

if __name__ == '__main__':

    episodes = 100
    max_steps = 1000
    test_ep = 10
    reward_sum = train_dqn(episodes, max_steps)
    plt.plot([i + 1 for i in range(0, episodes, 2)], reward_sum[::2])
    plt.show()