from gym.envs.registration import register

register(
    id="wordle-v0", entry_point="wordle_gym.envs.wordle_env:WordleEnv",
)

register(
    id="wordle-alpha-v0", entry_point="wordle_gym.envs.wordle_alpha_env:WordleEnv",
)