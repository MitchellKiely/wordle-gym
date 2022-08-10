import inspect
import time
from statistics import mean, stdev
import numpy as np
import os
import gym
import sys

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

if __name__ == "__main__":
    
    RL_algos = ["PPO"] #, "A2C", "DQN"]

    timesteps = 10000

    steps = round(timesteps/1000000, 2)
    

    for RL_algo in RL_algos:

        env = WordleEnv()

        model = RLagent(env=env, agent_type = RL_algo)

        model.train(timesteps=int(timesteps), log_name =f"{RL_algo} training for {steps} million steps")

        model.save(f"{RL_algo} trained for {round(steps)}mill steps")
    
