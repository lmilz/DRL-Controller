import numpy as np
import gym
import time
import random

class PIDController:
    def __init__(self, envname):
        self.env = gym.make(envname)
        print("Aktionsraum {}".format(self.env.action_space))
        print("Zustandsraum {}".format(self.env.observation_space))
        self.ref_value = 0.0
        self.angle = []
        self.angle_velo = []
        self.cart_pos = []
        self.cart_velo = []
        self.error = 0.0
        self.done = False
        self.state = self.env.reset()
        self.reward = 0.0
        self.kp = 1.0
        self.ki = 1.0
        self.kd = 1.0
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
        
    def getAction(self, val):
        pass

    def control(self):
        current_time = time.time()
        dt = self.prev_time - current_time

        self.pterm = self.error
        self.iterm += self.error*dt
        self.dterm = (self.prev_error - self.error)/dt

        self.ptermlist.append(self.pterm)
        self.itermlist.append(self.iterm)
        self.dtermlist.append(self.dterm)

        self.prev_error = self.error
        self.prev_time = current_time

        pid = self.kp*self.pterm + self.ki*self.iterm + self.kd*self.dterm

        self.outputlist.append(pid)
        self.frames.append(self.env.render(mode="rgb_array"))
        received_action = self.getAction(pid)
        next_state, self.reward, self.done, info = self.env.step(received_action)

        self.state = next_state

    def close(self):
        self.env.close()