import random
import numpy as np
from copy import deepcopy
from network_generation import Network

observation_space, action_space = 4, 4

class NEAT_functions():
    def information(self, species1, species2):
        species1_connections, species2_connections = [], [] # all connections in network
        species1_weights, species2_weights = [], [] # all weights of connections in network

        for idx in range(2):
            if idx == 0: species = species1
            else: species = species2

            for layer in species.network:
                for node in layer:
                    for connection in node.connections:
                        input_identity, output_identity = node.identity, connection.target_identity
                        connection_attribute = [input_identity, output_identity]
                        weight = connection.weight

                        if idx == 0: 
                            species1_connections.append(connection_attribute)
                            species1_weights.append(weight)
                        else:
                            species2_connections.append(connection_attribute)
                            species2_weights.append(weight)
        
        return species1_connections, species2_connections, species1_weights, species2_weights

    def speciation(self, species1, species2):
        species1_connections, species2_connections, species1_weights, species2_weights = self.information(species1, species2)

        N = max(len(species1_connections), len(species2_connections))
        if N == 0:
            N = 1
        E = abs(species1.number - species2.number)

        D = 0
        both = species1_connections + species2_connections
        for i in range(len(both)):
            if both.count(both[i]) == 1:
                D += 1
        
        W = 0 # sum of weight differences for connections shared
        shorter_species = species1_connections
        if species1_connections <= species2_connections:
            shorter_species = species1_connections

        for i in range(len(shorter_species)):
            connection_identified = shorter_species[i]
            if connection_identified in species1_connections and connection_identified in species2_connections:
                index_species_one = species1_connections.index(connection_identified)
                index_species_two = species2_connections.index(connection_identified)
                
                W += abs(species1_weights[index_species_one] - species2_weights[index_species_two])

        number = E/N + D/N + 0.5*W
        return number

    def species_generation(self, networks, threshold_species):
        species = []
        species.append([networks[0]])

        for network_idx in range(1, len(networks)):
            added = False
            for species_idx in range(len(species)):
                constant = self.speciation(species[species_idx][0], networks[network_idx]) 
                if constant >= threshold_species:
                    species[species_idx].append(deepcopy(networks[network_idx]))
                    added = True
                    break
            
            if added == False: species.append([networks[network_idx]])
        
        return species

    def selecting_score(self, all_parents):
        fitness_scores = []
        for i in range(len(all_parents)):
            fitness_scores.append(all_parents[i].score)

        probabilities = []
        added_probs = []
        total_score = 0

        for i in range(len(fitness_scores)):
            total_score += fitness_scores[i]
        factor = 1/total_score

        for i in range(len(fitness_scores)):
            probabilities.append(round(fitness_scores[i] * factor, 2))
        
        added_probs.append(probabilities[0])
        for i in range(1, len(probabilities)):
            added_probs.append(added_probs[i-1] + probabilities[i])
        added_probs = [0] + added_probs

        roll = round(random.random(), 2)

        for i in range(1, len(added_probs)):
            if added_probs[i-1] <= roll <= added_probs[i]:
                return i-1
            if roll > added_probs[len(added_probs)-1]:
                return len(added_probs)-2

    def making_children(self, species1, species2, observation_space, action_space):
        species1_connections, species2_connections, species1_weights, species2_weights = self.information(species1, species2)

        parents = [species1, species2]
        index = np.argmax([int(species1.score), int(species2.score)])
        fit_parent = parents[index] # take the fitest parent
        less_fit_parent = parents[abs(1-index)]

        child = Network()
        child.network = fit_parent.network
        child.number = fit_parent.number

        both = species1_connections + species2_connections
        connections_both = []
        parent_chosen = []
        copy_indexs = []
        both_weights = species1_weights + species2_weights
        
        for i in range(len(both)): 
            if both.count(both[i]) == 2:
                if both[i] not in connections_both:
                    connections_both.append(both[i])

        for i in range(len(connections_both)):
            parent_chosen.append(random.randint(0, 1))

        for i in range(len(connections_both)):
            indices = [index for index, element in enumerate(both) if element == connections_both[i]]
            copy_indexs.append(indices)
        
        for layer in child.network:
            for node in layer:
                for connection in node.connections:
                    if connection.enabled:
                        for p in range(len(connections_both)):
                            if connections_both[p][0] == node.identity and connections_both[p][1] == connection.target_identity:
                                connection.weight = both_weights[copy_indexs[p][parent_chosen[p]]]
        
        return child


    def repopulate(self, species, number_kids):
        new_networks = []

        for species_idx in range(len(species)):
            for number_idx in range(number_kids[species_idx]):
                if number_idx == 1 or len(species[species_idx]) == 1: # if only one network, keep it
                    new_networks.append(species[species_idx][0])
                else:
                    generated = []
                    generated.append(int(self.selecting_score(species[species_idx])))

                    second_index = self.selecting_score(species[species_idx])
                    while second_index == generated[0]:
                        second_index = self.selecting_score(species[species_idx])
                    
                    generated.append(int(second_index))
                    first_parent = species[species_idx][generated[0]]
                    second_parent = species[species_idx][generated[1]]

                    child = self.making_children(first_parent, second_parent, observation_space, action_space)
                    new_networks.append(deepcopy(child))
        
        return new_networks