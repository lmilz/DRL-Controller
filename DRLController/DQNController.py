import numpy as np
import gym
import time
import random
from collections import deque
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\DRLController")

from utils import *

class DQNController:
    def __init__(self, envname):
        self.start_time = time.time()
        self.model = None
        self.env = gym.make(envname)
        self.statespace_size = self.env.observation_space.shape[0]
        self.actionspace_size = self.env.action_space.n
        print("Aktionsraum {}".format(self.env.action_space))
        print("Zustandsraum {}".format(self.env.observation_space))
        self.memory = deque(maxlen = 500000)
        self.alpha = 0.001
        self.gamma = 0.99
        self.batch_size = 32
        self.num_episodes = 25
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.max_epsilon = 1.0
        self.decay_epsilon = 0.005
        self.rewardList = []
        self.ave_rewards = []

    def getAction(self, state):
        if random.uniform(0, 1) > self.epsilon:
            QValue = self.model.predict(state)
            return np.argmax(QValue[0])
        else:
            return random.randrange(self.actionspace_size)

    def learn(self):
        for eps in range(self.num_episodes+1):       
            state = self.env.reset()
            state = np.reshape(state, [1, self.statespace_size])
            done = False
            total_rewards = 0

            while not done:
                action = self.getAction(state)
                
                nextState, reward, done, info = self.env.step(action)
                
                nextState = np.reshape(nextState, [1, self.statespace_size])
                self.memory.append((state, action, reward, nextState, done))
                state = nextState
                total_rewards += reward

                if len(self.memory) > self.batch_size:  
                    randomSample = random.sample(self.memory, self.batch_size)
                    for stateSample, actionSample, rewardSample, nextStateSample, doneSample in randomSample:
                        QUpdate = rewardSample + self.gamma * (np.amax(self.model.predict(nextStateSample)[0])) * (1 - doneSample)
                        QValue = self.model.predict(stateSample)
                        QValue[0][actionSample] = QUpdate
                        
                        self.model.fit(stateSample, QValue, verbose=0) 

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_epsilon*eps)

            print("Episode: {}, Reward: {}".format(eps, total_rewards))
            self.rewardList.append(total_rewards)

            if eps % 10 == 0:
                average = np.mean(self.rewardList)
                self.ave_rewards.append(average)

                if average > 250:
                    self.model.save('Episode_' + str(eps) + '_weight.h5')
                    print('Aufgabe gelöst!')
                    break
                    
            if eps % 50 == 0:
                self.model.save('Episode_' + str(eps) + '_weight.h5')
                
        print("--- %s hours ---" % str((time.time() - self.start_time)/3600))
        plotRewards(self.rewardList, self.ave_rewards, 10)

    def control(self):
        pass

    def close(self):
        self.env.close()    