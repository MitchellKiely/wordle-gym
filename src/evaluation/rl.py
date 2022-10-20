import inspect
import numpy as np
import os

from stable_baselines3 import PPO, A2C, DQN, HER, DDPG, SAC, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from wordle_env import WordleEnv



class RLagent():

    def __init__(self, env, agent_type, model_name=None):
        super().__init__()
        self.agent_type = agent_type

        if model_name is None:
            if agent_type == "PPO":
                self.model = PPO('MlpPolicy', env, tensorboard_log= "./tb_logs/", verbose=1)
            elif agent_type == "A2C":
                self.model = A2C('MlpPolicy',env, tensorboard_log= "./tb_logs/", verbose=1 )
            elif agent_type == "DQN":
                self.model = DQN('MlpPolicy',env, tensorboard_log= "./tb_logs/", verbose=1, exploration_fraction = 0.90)
            elif agent_type == "MaskablePPO":
                env = ActionMasker(env, env.valid_action_mask) 
                self.model = MaskablePPO(MaskableActorCriticPolicy, env, tensorboard_log= "./tb_logs/", verbose=1)
            else:
                raise Exception("Unknown Agent Type {}".format(agent_type))
        else:
            self.load(agent_type, model_name)

    def train(self, timesteps,log_name, callback = None):
        eval_callback = EvalCallback(WordleEnv(), best_model_save_path='trained_models/',
                        log_path=log_name, eval_freq=1000,
                        deterministic=True, render=False)
        self.model.learn(timesteps, tb_log_name=log_name, callback=eval_callback)

    def run_trained_model(self, env, num_episodes=10):
        for ep in range(num_episodes):
            print(f"Episode: {ep}")
            obs = env.reset()
            done = False
            reward_total = 0
            while not done:
                obs, reward, done = env.step(self.get_action(obs))
                env.render()
                reward_total += reward
            print(f"Total Reward: {reward_total}, Num guesses: {env.guess_no}\n")

    def get_action(self, observation, valid_actions=None):

        if self.agent_type == "DQN":
            action = self.model.predict(observation, deterministic=True)
        elif self.agent_type == "MaskablePPO":
            action, _states = self.model.predict(observation, action_masks=valid_actions)
        else:
            action = self.model.predict(observation)
        return action

    def save(self, name = None):
        if name is None:
            name = "{}-{}".format(datetime.datetime.now(), self.agent_type)
        self.model.save(name)

    def load(self,agent_type, model_name):
        if agent_type == "PPO":
            self.model = PPO.load(model_name)
        elif agent_type == "A2C":
            self.model = A2C.load(model_name)
        elif agent_type == "DQN":
            self.model = DQN.load( model_name)
        elif agent_type == "MaskablePPO":
            self.model = MaskablePPO.load(model_name)


        self.agent_type = "{} ({})".format(self.agent_type,  model_name)
    def __str_(self):
        return self.agent_type

