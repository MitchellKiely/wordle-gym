import inspect
import sys
import time
from statistics import mean, stdev
import numpy as np
import os
import gym
import sys
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

sys.path.append("..")
from wordle_env import WordleEnv
from rl import RLagent

MAX_EPS = 100
NUM_GUESSES = 6
steps = []
correct = 0
if __name__ == "__main__":
    for i in range(MAX_EPS):
    
        env = WordleEnv()
        #agent = RLagent(env, model_name=r"C:\Users\RICT\Documents\wordle\src\train_agents\PPO trained for 5mill steps.zip", agent_type="PPO")
        agent = RLagent(env, model_name=r"C:\Users\RICT\Documents\wordle\src\evaluation\PPO trained for 0mill steps.zip", agent_type="PPO")
       
        guesses = []
        obs = env.reset()
        done = False
        for j in range(NUM_GUESSES):
            action, _states = agent.get_action(obs)
            obs, rewards, done, info = env.step(action)
            guesses.append(env._valid_words[action][0])
            #print(j)
            if done == True:
                steps.append(j)
                print(f'Solution: {env.solution}.....Guesses:f{guesses}......Reward: {rewards}')

    list = ['1st Guess', '2nd Guess', '3rd Guess', '4th Guess', '5th Guess', 'My bot sucks :(']

    plt.hist(steps, edgecolor='black', bins=6)
    plt.xticks(range(6), list )
    plt.xlim(-0.2,5.3)
    plt.savefig('2nd Attempt')
    plt.show()
            

