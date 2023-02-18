import math
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle 
from copy import deepcopy
from tqdm import tqdm
from network_generation import Network
from network_functionality import Passing
from neat_functionality import NEAT_functions

batch_size = 10
file = open('motorscaling.txt', 'r')
lines = file.readlines()
STORAGE = int(len(lines) / 3)

epochs = 100
threshold_species = 1
mutations = 5
coefficient = 0.1
network_amount = 10
action_space = 4
networks = []
best_run_throughs = []

for idx in range(network_amount):
    network = Network()
    for i in range(mutations):
        network.mutation()
    networks.append(deepcopy(network))

passing = Passing()
neat_functions = NEAT_functions()

average_reward = []
episodes = []
best_run = Network()

for epoch in range(epochs):
    average_general_score = 0
    total_error = 0

    for network in networks:
        error = 0
        network.score = 0
        passing.network = network.network
        indexs = random.sample(range(STORAGE), batch_size)

        for i in tqdm(range(len(indexs))):
            state = lines[indexs[i] * 3 + 1]
            state = state.rstrip()
            state = np.fromstring(state[1:-1], sep=' ')
            state = np.array([state])

            ground_truth = lines[indexs[i] * 3 + 2]
            ground_truth = ground_truth.rstrip()
            ground_truth = np.fromstring(ground_truth[1:-1], sep=' ')
            ground_truth = np.array([ground_truth])

            #print(state)
            #print(type(state))
            value = passing.forward(state)

            error += np.sum((ground_truth - value)**2)

            passing.backward(value - ground_truth)

        total_error += error
        network.score = 1/error
        average_general_score += network.score
        print(epoch, error, network.score)

        if best_run.score < network.score:
            best_run = deepcopy(network)

    average_general_score /= len(networks)
    # create species
    species = neat_functions.species_generation(networks, threshold_species)

    # cut worse half of species
    for isolated_species in species:
        isolated_species.sort(key=lambda x: x.score, reverse=True)
    for idx in range(len(species)):
        cutting = len(species[idx])//2
        new_species = species[idx][0:len(species[idx]) - cutting]
        species[idx] = new_species

    # determine number of new kids
    number_kids = []
    for idx in range(len(species)): number_kids.append(0)

    idx = 0
    for isolated_species in species:
        isolated_average = 0
        for network in isolated_species:
            isolated_average += network.score
        isolated_average /= len(isolated_species)

        amount = math.ceil(isolated_average/average_general_score * len(isolated_species))
        number_kids[idx] = amount
        idx += 1
    
    # repopulate
    new_kids = neat_functions.repopulate(species, number_kids)
    networks = new_kids
    networks.append(deepcopy(best_run)) 

    for network in networks:
        network.mutation()
    #networks += deepcopy(best_run_throughs[0])
    
    threshold_species += 0.1 * (len(species) * coefficient - len(species))

    print(f"average score {average_general_score} average error {total_error/len(networks)} for epoch {epoch}")
    
    average_reward.append(average_general_score)
    #average_reward.append(best_run.score)
    episodes.append(epoch)

print(best_run.score)
print("")
print(best_run.network)

filehandler = open("best_network.obj", 'wb') 
pickle.dump(best_run, filehandler)


plt.scatter(episodes, average_reward, color= "black", marker= "x")

slope, intercept = np.polyfit(episodes, average_reward, 1)
plt.plot(episodes, slope * np.array(episodes) + intercept)
plt.xlabel("Generation number")
plt.ylabel("Average score of network in generation")

plt.show()

