import gym
import pickle
from network_functionality import Passing

passing = Passing()

env = gym.make("BipedalWalker-v3")
best_reward = float("-inf")

filehandler = open("best_network2.obj", 'rb') 
best_run = pickle.load(filehandler)

agent = best_run
passing.network = best_run.network

for i in range(10):
    state = env.reset()
    score = 0

    while True:
        env.render()

        action = passing.forward([state[4:8]])
        #action = env.action_space.sample()
        state_, reward, done, info = env.step(action)
        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            print("Agent {} Best Reward {} Last Reward {}".format(i+1, best_reward, score))
            break

    env.close()
