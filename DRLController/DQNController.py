import numpy as np
import gym
import time
import random
from collections import deque

class DQNController:
    def __init__(self, evname):
        self.start_time = time.time()
        self.env = gym.make(envname)
        self.statespace_size = env.observation_space.shape[0]
        self.actionspace_size = env.action_space.n
        print("Aktionsraum {}".format(env.action_space))
        print("Zustandsraum {}".format(env.observation_space))
        self.memory = deque(maxlen = 500000)
        self.alpha = 0.001
        self.gamma = 0.99
        self.batch_size = 32
        self.num_episodes = 100
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.max_epsilon = 1.0
        self.decay_epsilon = 0.005
        self.rewardList = []
        self.ave_rewards = []

    def getAction(self):
        if random.uniform(0, 1) > self.epsilon:
            QValue = self.model.predict(state)
            return np.argmax(QValue[0])
        else:
            return random.randrange(self.actionspace_size)

    def learn(self):
        pass

    def control(self):
        pass