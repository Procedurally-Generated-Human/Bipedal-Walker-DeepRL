import gymnasium as gym
from gym_recorder import Recorder


env = gym.make("BipedalWalker-v3", render_mode="human")



env.reset()
for i in range(100):
    action = env.action_space.sample()
    env.step(action)