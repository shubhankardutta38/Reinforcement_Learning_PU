#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:01:26 2023

@author: shubhankardutta
"""
#Temporal Difference (TD) prediction
import pandas as pd
import gymnasium as gym
env=gym.make('FrozenLake-v1',render_mode = 'human')
env.reset()
env.render()
print(env.observation_space)
print(env.action_space)

def random_policy():
    return env.action_space.sample()

v = {}
for s in range(env.observation_space.n):
    v[s]=0.0

alpha=0.8
gamma=0.9
t_s=200
es=200


for i in range(es):
    s = env.reset()
    s = s[0]
    for t in range(t_s):
        a = random_policy()
        n_s, r, done, _, _ = env.step(a)
        v[s] += alpha * (r + gamma * v[n_s] - v[s])
        s = n_s
        if done:
            break
df = pd.DataFrame(list(v.items()), columns = ['state','value'])
print(df)
env.close()
    
