from ctypes.wintypes import WORD
import os

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from enum import Enum
from collections import Counter
import numpy as np
import random

from Levenshtein import distance as levenshtein_distance

WORD_LENGTH = 5
allowed_guesses = 6
SOLUTION_PATH = "words/solution_wordle.csv"
VALID_WORDS_PATH = "words/guess_wordle.csv"


max_green_matches = ['sauce', 'saucy', 'soapy', 'saute', 'sense', 'spree', 'gooey', 'scree', 'sooty', 'slate', 'saint', 'suite', 'slice', 'seize', 'sassy', 'puree', 'slimy', 'since', 'melee', 'sleet']
max_green_letters = ['slate', 'sauce', 'slice', 'shale', 'saute', 'share', 'sooty', 'shine', 'suite', 'crane', 'saint', 'soapy', 'shone', 'shire', 'saucy', 'slave', 'cease', 'sense', 'saner', 'snare']
max_green_yellow_letters = ['alert', 'alter', 'later', 'irate', 'arose', 'stare', 'arise', 'raise', 'learn', 'renal', 'saner', 'snare', 'cater', 'crate', 'react', 'trace', 'clear', 'least', 'slate', 'stale']
min_levenshtein_words = ['slate', 'crane', 'shale', 'share', 'saner', 'saute', 'stale', 'slice', 'suite', 'crate', 'shine', 'stare', 'shone', 'scale', 'saint', 'cease', 'crone', 'shore', 'snare', 'scare']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class WordleEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def _current_path(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _read_solutions(self):
        return open(os.path.join(self._current_path(), SOLUTION_PATH)).read().splitlines()
    
    def _get_valid_words(self):
        words = []
        for word in open(os.path.join(self._current_path(), VALID_WORDS_PATH)).read().splitlines():
            words.append((word, Counter(word)))
        return words


    def __init__(self, action_space):
        self._solutions = self._read_solutions()
        print(len(self._solutions))
        self._valid_words = self._get_valid_words()
        #self.action_space = spaces.Discrete(len(self._valid_words))
        self.action_space = action_space
        if isinstance(self.action_space, spaces.MultiDiscrete):
            self.observation_space = spaces.Box(0,len(self._valid_words), shape=(13,), dtype=np.int32)
        else:
            self.observation_space = spaces.Box(0,len(self._valid_words), shape=(3,), dtype=np.int32)
        self.guess_no = 0
        self.prev_guess = 0
        self.episode = 0
        self.choices = []

    def step(self, action):
        if isinstance(self.action_space, spaces.MultiDiscrete):
            multidisc = True
            action, choice = action[0]
            if action == 4:
                if min_levenshtein_words[choice] not in self.choices:
                    self.choices.append(min_levenshtein_words[choice])
        else:
            multidisc = False
            action, choice = action
        if action == 0:
            if multidisc:
                guess = self._valid_words[choice][0]
            else:
                guess = self._valid_words[np.random.randint(len(self._valid_words))][0]
                while guess in self.guesses:
                    guess = self._valid_words[np.random.randint(len(self._valid_words))][0]
        elif action == 1:
            if multidisc:
                guess = max_green_matches[choice]
            else:
                guess = random.choice(max_green_matches)
                while guess in self.guesses:
                    guess = random.choice(max_green_matches)
        elif action == 2:
            if multidisc:
                guess = max_green_letters[choice]
            else:
                guess = random.choice(max_green_letters)
                while guess in self.guesses:
                    guess = random.choice(max_green_letters)
        elif action == 3:
            if multidisc:
                guess = max_green_yellow_letters[choice]
            else:
                guess = random.choice(max_green_yellow_letters)
                while guess in self.guesses:
                    guess = random.choice(max_green_yellow_letters)
        elif action == 4:
            if multidisc:
                guess = min_levenshtein_words[choice]
            else:
                guess = random.choice(min_levenshtein_words)
                while guess in self.guesses:
                    guess = random.choice(min_levenshtein_words)
        elif action == 5:
            guess = random.choice(self.possible_words())[0]

        if tuple(self.obs[:2]) in self.action_dist:
            if action in self.action_dist[tuple(self.obs[:2])]:
                self.action_dist[tuple(self.obs[:2])][action] += 1
            else:
                self.action_dist[tuple(self.obs[:2])][action] = 1
        else:
            self.action_dist[tuple(self.obs[:2])] = {action: 1}

        done = False
        reward = -1
        yellow_added_this_step = []
        if guess == self.solution:
            done = True
            reward = 10
            if isinstance(self.action_space, spaces.MultiDiscrete):
                self.obs = [5,0,0]+self.obs[3:]
                self.obs[self.guess_no] = self._solutions.index(guess)
            else:
                self.obs = [5, 0, 0]
            self.greens = dict(zip(range(WORD_LENGTH), list(self.solution)))
        else:
            if guess in self.guesses:
                reward = -5
            else:
                guess_l = list(guess)
                sol_l = list(self.solution)
                for i in range(WORD_LENGTH):
                    if guess_l[i] == sol_l[i]:
                        #reward += 5
                        self.greens[i] = guess_l[i]
                    elif guess_l[i] in sol_l:
                        if guess_l[i] in yellow_added_this_step: continue
                        indices = [j for j, x in enumerate(sol_l) if x == guess_l[i]]
                        accounted_for = True
                        for idx in indices:
                            if sol_l[idx] != guess_l[idx]:
                                accounted_for = False
                        if not accounted_for:
                            if guess_l[i] in self.yellows:
                                self.yellows[guess_l[i]].append(i)
                            else:
                                self.yellows[guess_l[i]] = [i]
                            if len(indices) == len(self.yellows[guess_l[i]]):
                                yellow_added_this_step.append(guess_l[i])
                        #reward +=1 
                    else:
                        self.greys.append(guess_l[i])
            if isinstance(self.action_space, spaces.MultiDiscrete):
                self.obs = [len(self.greens), sum([len(v) for v in self.yellows.values()]), len(self.greys)]+self.obs[3:]
                self.obs[3+self.guess_no] = self._solutions.index(guess)
            else:
                self.obs = [len(self.greens), sum([len(v) for v in self.yellows.values()]), len(self.greys)]
        #print(f"Greens: {self.greens}")
        #print(f"Yellow: {self.yellows}")
        #print(f"Greys: {self.greys}\n")
        self.guess_no += 1
        self.guesses.append(guess)
        if self.guess_no == 10:
            done = True
        if self.episode % 20 == 0:
            self.render()
        return self.obs, reward, done, {} 

    def reset(self):
        self.solution = self._solutions[np.random.randint(len(self._solutions))]
        #self.solution = "denim"
        self.solution_ct = Counter(self.solution)
        self.guess_no = 0
        self.guesses = []
        if isinstance(self.action_space, spaces.MultiDiscrete):
            self.obs = [0,0,0,-1,-1,-1,-1,-1,-1, -1, -1, -1, -1]
        else:
            self.obs = [0,0,0]
        self.guesses = []
        self.rewards = []
        self.prev_guess = 0
        self.greens = {}
        self.yellows = {}
        self.greys = []
        if self.episode % 20 == 0:
            print(f"\nEpisode: {self.episode}")
        '''
        if self.episode != 0:
            if ((self.episode < 5000 and self.episode % 500 == 0) or 
            (self.episode >= 5000 and self.episode % 50000 == 0)):
                with open("action_distribution.txt", "a") as f:
                    f.write(f"Episode: {self.episode}\n")
                    f.write(f"{self.action_dist}\n")
                self.action_dist = {}
        '''
        if self.episode == 0:
            self.action_dist = {}
        
        self.episode += 1
        return self.obs
    
    def possible_words(self):
        possible_words = []
        for i, word in enumerate(self._valid_words):
            if word[0] in self.guesses:
                continue
            word_l = list(word[0])
            green_idxs = self.greens.keys()
            possible_flag = True
            for i, letter in enumerate(word_l):
                if i in green_idxs and self.greens[i] != letter:
                    possible_flag = False
                    break
                elif letter in self.greys:
                    possible_flag = False
                    break
            if not possible_flag: continue
            
            for yellow, yellow_not_idxs in self.yellows.items():
                if yellow not in word_l:
                    possible_flag = False
                    break
                elif yellow in word_l:
                    indices = [j for j, x in enumerate(word_l) if x == yellow]
                    if np.all(indices==yellow_not_idxs):
                        possible_flag = False
                        break
            if possible_flag:
                possible_words.append(word)
        return possible_words
        
    def render(self):
        yellow_letters = []
        for v in self.yellows.values():
            yellow_letters.extend(v)
        for i, letter in enumerate(self.guesses[-1]):
            if i in self.greens and self.greens[i] == letter:
                print(bcolors.OKGREEN+f"{letter}"+bcolors.ENDC, end="")
            elif letter in self.yellows and i in self.yellows[letter]:
                print(bcolors.WARNING+f"{letter}"+bcolors.ENDC, end="")
            else:
                print(letter, end="")
        print()

    def close(self):
        pass


def get_levenshtein_distance():
    levenshteins = {}
    for i, word in enumerate(env._valid_words):
        word = word[0]
        print(f"{i}/{len(env._valid_words)}")
        for guess_word in env._valid_words:
            guess_word = guess_word[0]
            levenshteins[guess_word] = levenshteins.get(guess_word, 0) + levenshtein_distance(guess_word, word)
    min_lev_words = sorted(list(zip(levenshteins.keys(), levenshteins.values())), key=lambda k: k[1])
    print(min_lev_words[:20])

def get_all_shared():
    shared_letters = {}
    for i, word in enumerate(env._valid_words):
        print(f"{i}/{len(env._valid_words)}")
        for guess_word in env._valid_words:
            overlap = list((word[1] & guess_word[1]).elements())
            shared_letters[guess_word[0]] = shared_letters.get(guess_word[0], 0) + len(overlap)
    max_match_words = sorted(list(zip(shared_letters.keys(), shared_letters.values())), key=lambda k: k[1], reverse=True)    
    print(max_match_words[:20])

def get_max_greens():
    greens = {}
    greens_letters = {}
    for i, word in enumerate(env._valid_words):
        word = word[0]
        print(f"{i}/{len(env._valid_words)}")
        for j, guess_word in enumerate(env._valid_words):
            guess_word = guess_word[0]
            if word == guess_word: continue
            word_l = list(word)
            guess_word_l = list(guess_word)
            greens_sum = sum(np.array(word_l)==np.array(guess_word_l))
            if greens_sum > 0:
                greens[guess_word] = greens.get(guess_word, 0)+1
            greens_letters[guess_word] = greens_letters.get(guess_word, 0)+greens_sum

    max_match_words = sorted(list(zip(greens.keys(), greens.values())), key=lambda k: k[1], reverse=True)
    max_match_letters = sorted(list(zip(greens_letters.keys(), greens_letters.values())), key=lambda k: k[1], reverse=True)

    print(max_match_words[:20])
    print(max_match_letters[:20])

'''
if __name__ == "__main__":
    action_space = spaces.MultiDiscrete((6,20))
    env = WordleEnv(action_space)
    #max_green_matches = [('sauce', 1122), ('saucy', 1115), ('soapy', 1114), ('saute', 1104), ('sense', 1100), ('spree', 1099), ('gooey', 1098), ('scree', 1097), ('sooty', 1095), ('slate', 1077), ('saint', 1077), ('suite', 1075), ('slice', 1069), ('seize', 1061), ('sassy', 1057), ('puree', 1048), ('slimy', 1043), ('since', 1042), ('melee', 1039), ('sleet', 1037)]
    #max_green_letters = [('slate', 1432), ('sauce', 1406), ('slice', 1404), ('shale', 1398), ('saute', 1393), ('share', 1388), ('sooty', 1387), ('shine', 1377), ('suite', 1376), ('crane', 1373), ('saint', 1366), ('soapy', 1361), ('shone', 1355), ('shire', 1347), ('saucy', 1346), ('slave', 1339), ('cease', 1337), ('sense', 1337), ('saner', 1334), ('snare', 1331)]
    #max_green_yellow_letters = [('alert', 4117), ('alter', 4117), ('later', 4117), ('irate', 4116), ('arose', 4093), ('stare', 4087), ('arise', 4067), ('raise', 4067), ('learn', 4000), ('renal', 4000), ('saner', 3970), ('snare', 3970), ('cater', 3917), ('crate', 3917), ('react', 3917), ('trace', 3917), ('clear', 3898), ('least', 3898), ('slate', 3898), ('stale', 3898)]
    #min_levenshtein_words = [('slate', 9972), ('crane', 10002), ('shale', 10004), ('share', 10032), ('saner', 10035), ('saute', 10043), ('stale', 10044), ('slice', 10045), ('suite', 10045), ('crate', 10047), ('shine', 10052), ('stare', 10060), ('shone', 10069), ('scale', 10070), ('saint', 10073), ('cease', 10080), ('crone', 10082), ('shore', 10083), ('snare', 10087), ('scare', 10091)]
    for ep in range(10):
        print(f"Episode: {ep}")
        env.reset()
        done = False
        reward_total = 0
        while not done:
            obs, reward, done, _ = env.step(action_space.sample())
            print(obs)
            env.render()
            reward_total += reward
        print(f"Total Reward: {reward_total}, Num guesses: {env.guess_no}\n")
'''