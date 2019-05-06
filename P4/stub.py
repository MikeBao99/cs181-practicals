# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt
import random
import traceback
import seaborn as sns
from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, Q):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.acceleration = 0.
        self.n = 0.
        self.a = 15 # horizontal
        self.b = 10 # vertical
        self.v = 6 # velocity
        self.gamma = 1
        self.epsilon = 0.1
        a = self.a
        b = self.b
        v = self.v
        self.Q = Q
        if self.Q is None or Q.shape != (2,b,v,a,b,b,2):
            self.Q = np.zeros((2,b,v,a,b,b,2))
            for ii in range(2):
                for i in range(b):
                    for j in range(v):
                        for k in range(a):
                            for l in range(b):
                                for m in range(b):
                                    if i <= l and j <= v/2:
                                        self.Q[ii][i][j][k][l][m][1] = 0.3
                                    elif i >= m:
                                        self.Q[ii][i][j][k][l][m][0] = 0.3
        self.eta = 0.8
        # acceleration, monkey location, monkey velocity, distance, tree bot, tree top

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.acceleration = 0.
        self.n = 0.
        self.eta *= 0.95
        self.epsilon *= 0.8
        # self.Q = np.zeros((b,v,a,b,b,2))

        
    def cat_hor(self, x):
        return int(x / (600 / self.a))
    
    def cat_vert(self, x):
        return int(x / (400 / self.b))

    def cat_vel(self, x):
        if x > 10:
            return self.v - 1
        elif x < -30:
            return 0
        else:
            return int((x + 30) / (40 / (self.v - 2))) + 1

    
    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        try:
            m_loc,m_vel,dist,t_bot,t_top = 0,0,0,0,0
            m_loc = self.cat_vert(self.last_state['monkey']['bot'])
            m_vel = self.cat_vel(self.last_state['monkey']['vel'])
            dist = self.cat_hor(self.last_state['tree']['dist'])
            t_bot = self.cat_vert(self.last_state['tree']['bot'])
            t_top = self.cat_vert(self.last_state['tree']['top'])

            m_loc_c,m_vel_C,dist_c,t_bot_c,t_top_c = 0,0,0,0,0
            m_loc_c = self.cat_vert(state['monkey']['bot'])
            # print(state['monkey']['vel'])
            m_vel_c = self.cat_vel(state['monkey']['vel'])
            dist_c = self.cat_hor(state['tree']['dist'])
            t_bot_c = self.cat_vert(state['tree']['bot'])
            t_top_c = self.cat_vert(state['tree']['top'])

            if self.last_action == 0:
                self.acceleration = self.last_state['monkey']['vel'] - state['monkey']['vel']
                if self.acceleration > 2:
                    self.acceleration = 1
                else:
                    self.acceleration = 0
            self.Q[self.acceleration, m_loc,m_vel,dist,t_bot,t_top,self.last_action] -= self.eta*(self.Q[self.acceleration, m_loc,m_vel,dist,t_bot,t_top,self.last_action]  - self.last_reward - 
                self.gamma * max(self.Q[self.acceleration, m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,0],self.Q[self.acceleration, m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,1]))
            # print(self.Q[m_loc,m_vel,dist,t_bot,t_top,self.last_action])
            if self.Q[self.acceleration, m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,0] >= self.Q[self.acceleration, m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,1]:
                self.last_action = 0
            else:
                self.last_action = 1

            if m_loc_c >= self.b - 1:
                self.last_action = 0
            if m_loc_c == 0:
                self.last_action = 1

            if random.random() < 0.5 * self.epsilon:
                self.last_action = 1 - self.last_action

            # print(np.count_nonzero(self.Q))
          
          
            self.last_state  = state

            return self.last_action
        except Exception as e:
            print(traceback.format_exc())
            # print(e)
            # print("holy crap")
            self.last_state = state
            self.last_action = 0
            return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
        return self.last_reward


def run_games(learner, hist, iters = 100, t_len = 1):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    try:
        # Q = np.load('Q.npy')
        Q = None
    except:
        Q = None
    agent = Learner(Q)
    eta_orig = agent.eta
    gamma_orig = agent.gamma
    epsilon_orig = agent.epsilon

    # Create list to save history.
    try:
        # hist = np.load('hist.npy')
        # hist = hist.tolist()
        hist = []
    except:
        hist = []

    # Run games. 
    run_games(agent, hist, 50, 0)

    plt.scatter(range(1, len(hist)+1), hist)
    plt.title(fr"Monkey's Scores ($\eta$ = {eta_orig}, "
        fr"$\gamma$ = {gamma_orig}, $\epsilon$ = {epsilon_orig})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.show()
    print(f"Max Score: {max(hist)}")
    sns.distplot(hist, rug=True).set(xlim=(0,None))
    plt.show()
    print('Mean:    %f\nStd Dev: %f' % (np.mean(hist), np.std(hist)))
    # Save history and Q
    np.save('hist', np.array(hist))
    np.save('Q', agent.Q)
