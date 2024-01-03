#Lab1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:02:06 2023

@author: shubhankardutta
"""

import gymnasium as gym
import time

env = gym.make('FrozenLake-v1',render_mode = 'human' )
env.reset()
env.render()

#print state space
print(env.observation_space)

#print action space
print(env.action_space)

print(env.P[0][2])

env.reset()
(next_state, reward, done, trans_prob,info) = env.step(2)
env.render()
rnd_action = env.action_space.sample()
env.step(rnd_action)
env.render()
print("Action taken:", rnd_action)
total_return=0
#generate 10 episode
for  e in range(10):
   num_timesteps = 20
   ret = 0
   env.reset()
   for t in range(num_timesteps):
     rnd_action = env.action_space.sample()
     (next_state, reward, done, trans_probability,info) = env.step(rnd_action)
     ret = ret + reward
     env.render()
     time.sleep(2)
     if done:
            print("Episode {}: The return of this episode is {}".format(e, ret))
            total_return += ret  
            break

print("Total return across all episodes is:", total_return)

env.close()

