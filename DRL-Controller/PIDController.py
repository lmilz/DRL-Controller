import numpy as np
import gym
import time
import random

class CartPoleController:
    def __init__(self, envname):
        self.env = gym.make(envname)
        print("Aktionsraum {}".format(self.env.action_space))
        print("Zustandsraum {}".format(self.env.observation_space))
        self.ref_value = 0.0
        self.initstate = self.env.reset()
        self.angle = []
        self.angle_velo = []
        self.cart_pos = []
        self.cart_velo = []
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 75.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.ptermlist = []
        self.itermlist = []
        self.dtermlist = []    
        self.outputlist = []  
        self.pterm = 0.0
        self.iterm = 0.0
        self.dterm = 0.0
        self.frames = []
        self.statelist = []
        
    def getAction(self, val):
        pass

    def control(self, error):
        state = self.env.reset()
        done = False

        while not done:
            #error = -state[2]
            current_time = time.time()
            dt = self.prev_time - current_time

            self.pterm = error
            self.iterm += error*dt
            self.dterm = (self.prev_error - error)/dt

            self.ptermlist.append(self.pterm)
            self.itermlist.append(self.iterm)
            self.dtermlist.append(self.dterm)

            self.prev_error = error
            self.prev_time = current_time

            pid = self.kp*self.pterm + self.ki*self.iterm + self.kd*self.dterm

            self.outputlist.append(pid)
            self.frames.append(self.env.render(mode="rgb_array"))
            received_action = self.getAction(pid)
            next_state, reward, done, info = self.env.step(received_action)

            state = next_state
            self.statelist.append(state)

    def close(self):
        env.close()