from ctypes.wintypes import WORD
import os

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from enum import Enum
from collections import Counter
import numpy as np

WORD_LENGTH = 5
allowed_guesses = 6
SOLUTION_PATH = "words/solution_wordle.csv"
VALID_WORDS_PATH = "words/guess_wordle.csv"

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


    def __init__(self):
        self._solutions = self._read_solutions()
        self._valid_words = self._get_valid_words()
        self.action_space = spaces.Discrete(len(self._valid_words))
        self.observation_space = spaces.Box(0,len(self._valid_words) , shape=(1+WORD_LENGTH,), dtype=np.int)
        self.guess_no = 0
        self.prev_guess = 0


    def step(self, action):
        """
        action: index of word in valid_words

        returns:
            observation: (WORD_LENGTH + previous_action)
            reward: +5 for every letter tin correct position, +1 for having a letter in the word, 
                    0 for incorrect letters, +20 if guess is correct, -10 if guess is the same as last guess
            done: True if game over, w/ or w/o correct answer
            additional_info: empty
        """
        #Expends str into 5 letters
        sol_lst = []
        sol_lst.extend(self.solution)

        #Turns selected word into 5 letters
        act_lst = []
        act_lst.extend(self._valid_words[action][0])
        self.guesses.append(self._valid_words[action][0])
        #Reinstatiates word each round
        reward=0

        #penalises for guessing the same word
        if action is self.prev_guess:
            reward -= 10

        #Appends current guess into a list
        self.prev_guess = action
        
        #Calculates Rewards
        for i in range(WORD_LENGTH):
            if act_lst[i] == sol_lst[i]:
                self.obs[i]=5
                reward += 5
            elif act_lst[i] in sol_lst:
                self.obs[i]=1
                reward +=1
            else:
                self.obs[i]=0
        
        self.obs[5] = action
        #Increments counter
        self.guess_no +=1

        #Checks to see if guessed action is equal to soltuon
        if self._valid_words[action][0] == self.solution:
            done = True
            reward += 25
        else:
            done=False
        self.rewards.append(reward)

        #Max number of guess is 6
        if self.guess_no >= allowed_guesses:
            done=True

        return self.obs, reward, done, {}

    def reset(self):
        self.solution = self._solutions[np.random.randint(len(self._solutions))]
        self.solution_ct = Counter(self.solution)
        self.guess_no = 0
        self.guesses = []
        self.obs = np.zeros((1+WORD_LENGTH,)).astype(np.int)
        self.guesses = []
        self.rewards = []
        self.prev_guess = 0
        return self.obs
    
    def valid_action_mask(self, env):
        if self.prev_guess != 0:
            valid_words = []

            w_ext = []
            w_ext.extend(self._valid_words[self.prev_guess][0])
            prev_guess_word = self._valid_words[self.prev_guess][0]

            guess_arr = np.array(list(prev_guess_word))
            solution_list = list(self.solution)
            solution_arr = np.array(solution_list)

            matched_bool = guess_arr == solution_arr
            matched_letters = solution_arr[matched_bool]
            matched_idx = np.where(matched_bool)[0]
            idx_letters = dict(list(zip(matched_idx,matched_letters)))
            partial_match_letters = []
            for i, l in enumerate(prev_guess_word):
                if l in solution_list:
                    if (i,l) in idx_letters:
                        continue
                    else:
                        partial_match_letters.append(l)
                        solution_list.remove(l) # in case of multiple partial matches of letter

            mask = []
            for i, word in enumerate(self._valid_words):
                if np.array(list(word[0]))[matched_bool]!=matched_letters:
                    mask.append(False)
                    continue
                else:
                    nonexact_matches = list(solution_arr[~matched_bool])
                    for l in partial_match_letters:
                        if l not in nonexact_matches:
                            mask.append(False)
                            continue
                        else:
                            nonexact_matches.remove(l)
                breakpoint()
                loop_ext = []
                loop_ext.extend(self._valid_words[i][0])
                tot_count = self.obs.tolist().count(5)
                let_counter = 0

                valid_flag = False
                # THIS ALWAYS RETURNS FALSE
                for j in range(len(self.obs)):
                    if self.obs[j] == 5:
                        if w_ext[j] == loop_ext[j]:
                            let_counter += 1
                            if let_counter == tot_count:
                                valid_words.append(i)
                                valid_flag = True
                if valid_flag: 
                    mask.append(True)
                else:
                    mask.append(False)

                #self._valid_words = valid_words
            breakpoint()
            return np.array(mask)
        else:
            return np.array([True]*len(self._valid_words))
        
    def render(self):
        pass

    def close(self):
        pass
