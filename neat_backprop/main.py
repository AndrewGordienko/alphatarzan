import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from network_generation import Network
from network_functionality import Passing
from neat_functionality import NEAT_functions
from keras.datasets import mnist

batch_size = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:batch_size].reshape(batch_size, 28*28)/255, y_train[0:batch_size])
x_test, y_test = (x_test[0:batch_size].reshape(batch_size, 28*28)/255, y_test[0:batch_size])

epochs = 5
threshold_species = 1
mutations = 5
coefficient = 0.1
network_amount = 10
action_space = 10
networks = []
best_run_throughs = []

for idx in range(network_amount):
    network = Network()
    for i in range(mutations):
        network.mutation()
    networks.append(deepcopy(network))

for i in range(int(network_amount * coefficient)):
    network = Network()
    for i in range(mutations):
        network.mutation()
    best_run_throughs.append(deepcopy(network))

passing = Passing()
neat_functions = NEAT_functions()

average_reward = []
episodes = []

for epoch in range(epochs):
    average_general_score = 0

    for network in networks:
        error = 0
        network.score = 0
        passing.network = network.network

        for i in tqdm(range(len(images))):
            image = images[i:i+1]
            ground_truth = np.zeros((1, 10))
            ground_truth[0][labels[i]] = 1

            value = passing.forward(image)

            error += np.sum((ground_truth - value)**2)

            passing.backward(value - ground_truth)

        network.score = 1/error
        average_general_score += network.score
        print(epoch, error, network.score)

    best_run_throughs.sort(key=lambda x: x.score)
    if best_run_throughs[0].score < network.score:
        best_run_throughs[0] = deepcopy(network)

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
    networks += new_kids

    for network in networks:
        network.mutation()
    #networks += deepcopy(best_run_throughs[0])
    
    threshold_species += 0.1 * (len(species) * coefficient - len(species))

    print(f"average score {average_general_score} for epoch {epoch}")

    average_reward.append(average_general_score)
    episodes.append(epoch)

plt.scatter(episodes, average_reward, color= "black", marker= "x")
plt.show()
