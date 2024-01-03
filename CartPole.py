#lab2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:43:37 2023

@author: shubhankardutta
"""


import gymnasium as gym 

env = gym.make("CartPole-v1", render_mode = "human")

env.reset()

env.render()

#print state space which is continuous
print(env.observation_space)

#print action space
print(env.action_space)


env.reset()
#implement with a random policy


n_episodes = 50
n_timesteps = 50

for i in range(n_episodes):
    
    Return = 0
    for t in range(n_timesteps):
        env.render()
       
        rnd_action = env.action_space.sample()
        if rnd_action == 0:
            print("Left")
        else:
            print("Right")
        next_state, reward, done, infor, prob = env.step(rnd_action)
        Return = Return + reward
        if done:
            env.reset()
            break
    if i%10 == 0:
        print("Episode : {}, Return : {}".format(i+1, Return))

env.close()
