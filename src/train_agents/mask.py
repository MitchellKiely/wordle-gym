import inspect
import time
from statistics import mean, stdev
import numpy as np
import os
import gym
import sys
sys.path.append("..")
from wordle_env import WordleEnv
from rl import RLagent

if __name__ == "__main__":
    env = WordleEnv()
    obs_lst = [0,0,5,5,5]
    obs = np.array(obs_lst)
    prev_guess=61 #Word = alone
    valid_words = []

    w_ext = []
    w_ext.extend(env._valid_words[prev_guess][0])

    for i in range(len(env._valid_words)):
        loop_ext = []
        loop_ext.extend(env._valid_words[i][0])
        tot_count = obs.tolist().count(5)
        let_counter = 0

        for j in range(len(obs)):
            if obs[j] == 5:
                if w_ext[j] == loop_ext[j]:
                    let_counter += 1
                    if let_counter == tot_count:
                        valid_words.append(i)

    breakpoint()


   
    