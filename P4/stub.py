# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Q = np.zeros((4,3,3,4,4,2))
        self.eta = 0.8
        # monkey location, distance, tree bot, tree top, monkey velocity

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Q = np.zeros((4,3,3,4,4,2))

        
    def cat_hor(self,x):
        if x < 200:
            return 0
        if x < 400:
            return 1
        else:
            return 2
    
    def cat_vert(self,x):
        if x < 100:
            return 0
        if x < 200:
            return 1
        if x < 300:
            return 2
        else:
            return 3
    
    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.


        try:        
            m_loc,m_vel,dist,t_bot,t_top= 0,0,0,0,0
            m_loc = self.cat_vert(self.last_state['monkey']['bot'])
            if state['monkey']['vel'] > 0:
                m_vel = 2
            elif state['monkey']['vel'] > -20:
                m_vel = 1
            else:
                m_vel = 0
            dist = self.cat_hor(self.last_state['tree']['dist'])
            t_bot = self.cat_vert(self.last_state['tree']['bot'])
            t_top = self.cat_vert(self.last_state['tree']['top'])

            m_loc_c,m_vel_C,dist_c,t_bot_c,t_top_c= 0,0,0,0,0
            m_loc_c = self.cat_vert(state['monkey']['bot'])
            print(state['monkey']['vel'])
            if state['monkey']['vel'] > 0:
                m_vel_c = 2
            elif state['monkey']['vel'] > -20:
                m_vel_c = 1
            else:
                m_vel_c = 0
            dist_c = self.cat_hor(state['tree']['dist'])
            t_bot_c = self.cat_vert(state['tree']['bot'])
            t_top_c = self.cat_vert(state['tree']['top'])

            self.Q[m_loc,m_vel,dist,t_bot,t_top,self.last_action] -= self.eta*(self.Q[m_loc,m_vel,dist,t_bot,t_top,self.last_action]  - self.last_reward - max(self.Q[m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,0] ,self.Q[m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,1]))
            print(self.Q[m_loc,m_vel,dist,t_bot,t_top,self.last_action])
            if self.Q[m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,0] >= self.Q[m_loc_c,m_vel_c,dist_c,t_bot_c,t_top_c,1]:
                self.last_action = 0
            else:
                self.last_action = 1

            if m_loc_c == 0:
                self.last_action = 1
          
          
            self.last_state  = state

            return self.last_action
        except Exception as e:
            print(e)
            print("holy crap")
            self.last_state = state
            self.last_action = 0
            return 0

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if reward < 0:
            self.last_reward = reward
        else:
            self.last_reward = 2


def run_games(learner, hist, iters = 100, t_len = 100):
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
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 50, 10)

    # Save history. 
    np.save('hist',np.array(hist))


