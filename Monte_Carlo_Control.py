#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:14:02 2023

@author: shubhankardutta
"""
# Monte Carlo Control 
import gymnasium as gym
import pandas as pd
import random

from collections import defaultdict
env= gym.make('Blackjack-v1',render_mode='human')
Q = defaultdict(float)
total_return = defaultdict(float)
N= defaultdict(int)
def epsilon_greedy_policy(state,Q):
    epsilon = 0.5
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)),key = lambda x:
                   Q[(state,x)])
                   
num_timesteps = 100
def generate_episode(Q):
    episode = []
    state = env.reset()
    state = state[0]
    for t in range(num_timesteps):
        action = epsilon_greedy_policy(state, Q)
        next_state, reward, done , info , trans_prob = env.step(action)
        episode.append((state,action,reward))
        if done:
            break
        
        state = next_state
    return episode

num_iteration =100
for i in range(num_timesteps):
    episode = generate_episode(Q)
    all_state_action_pairs  = [(s,a) for (s,a,r) in episode]
    rewards = [r for (s,a,r) in episode]
    
    for t, (state,action,_) in enumerate(episode):
        if not(state,action,_) in all_state_action_pairs[0:t]:
            R = sum(rewards[t:])
            total_return[(state,action)] = total_return[(state,action)] + R
            N[(state, action)] +=1
            Q[(state,action)] = total_return[(state, action)] / N[(state,action)]
            
df = pd.DataFrame(Q.items(),columns=['state_action_pair','value'])
print(df.head(11))
print(df[124:126])
env.close()
