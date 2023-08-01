import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import TQC
import os 



env = gym.make("BipedalWalker-v3")
env.reset()




model = TQC("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_bipedal_tensorboard/")
model.learn(total_timesteps=100000)
model.save("./saveppo")


# load model
# model = PPO.load("./saveppo.zip", env)

# render agent
env = gym.make("BipedalWalker-v3", render_mode="human")
for i in range(10):
    state, _ = env.reset()
    while True:
        env.render()
        action, _ = model.predict(state)
        state, reward, done, _, _ = env.step(action)
        if done:
            break
env.close()
