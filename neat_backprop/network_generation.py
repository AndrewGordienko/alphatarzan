import random
import numpy as np
import math
from copy import deepcopy

all_connections = []

# nodes
class Node():
    def __init__(self, identity, layer_number):
        self.identity = identity
        self.layer_number = layer_number
        self.value = 0
        self.delta = 0

        self.connections = []

# connections
class Connection():
    def __init__(self, target_identity, target_layer, innovation):
        self.target_identity = target_identity
        self.target_layer = target_layer
        self.weight = random.uniform(-1, 1)

        self.enabled = True
        self.innovation = innovation

# network
class Network():
    def __init__(self):
        self.observation_space = 784
        self.action_space = 10
        self.network = [[],[]]
        self.number = self.observation_space + self.action_space
        self.score = 0

        # initialize the nodes
        for i in range(self.observation_space):
            node = Node(i, 0)
            self.network[0].append(deepcopy(node))
        for i in range(self.action_space):
            node = Node(self.observation_space + i, 1)
            self.network[1].append(deepcopy(node))

        # initialize the connections
        for input_node in self.network[0]:
            for output_node in self.network[1]:
                innovation_number = self.finding_innovation(input_node.identity, output_node.identity)
                connection = Connection(output_node.identity, output_node.layer_number, innovation_number)
                input_node.connections.append(deepcopy(connection))
    
    def find_index(self, layer_index, target_identity):
        counter = 0
        for node in self.network[layer_index]:
            if node.identity == target_identity:
                return counter
            counter += 1
    
    def finding_innovation(self, input_node, output_node): # find innovation value
        connection_innovation = [input_node, output_node]

        if connection_innovation not in all_connections: all_connections.append(connection_innovation)
        innovation_number = all_connections.index(connection_innovation)

        return innovation_number
    
    def network_architecture(self):
        for layer_number in range(len(self.network)):
            print("--")
            print(f"layer number {layer_number}")

            for node in self.network[layer_number]:
                print(f"node {node.identity}")
                for connection in node.connections:
                    print(f"layer {connection.target_layer} number {connection.target_identity} ")
    
    def mutate_link(self): # new node between two connected nodes
        input_layer = random.randint(0, len(self.network)-2)
        input_index = random.randint(0, len(self.network[input_layer])-1)
        input_node = self.network[input_layer][input_index]

        connection = input_node.connections[random.randint(0, len(input_node.connections)-1)]
        connection.enabled = False

        target_layer, target_identity = connection.target_layer, connection.target_identity
        
        if abs(target_layer - input_layer) == 1: # no layer in between
            self.network.insert(input_layer+1, [])
            node = Node(self.number, input_layer+1)
            self.number += 1  
            
            innovation_number = self.finding_innovation(node.identity, target_identity)
            connection = Connection(target_identity, target_layer, innovation_number)
            node.connections.append(deepcopy(connection))

            self.network[input_layer+1].append(deepcopy(node))
            
            for layer in self.network:
                for node_idx in layer:
                    for connection in node_idx.connections:
                        if connection.target_layer >= node.layer_number:
                            connection.target_layer += 1
            
            innovation_number = self.finding_innovation(input_node.identity, node.identity)
            connection = Connection(node.identity, node.layer_number, innovation_number)
            input_node.connections.append(deepcopy(connection))
        
        else:
            node = Node(self.number, input_layer+1)
            self.number += 1

            innovation_number = self.finding_innovation(input_node.identity, node.identity)
            connection = Connection(node.identity, node.layer_number, innovation_number)
            input_node.connections.append(deepcopy(connection))
            
            innovation_number = self.finding_innovation(node.identity, target_identity)
            connection = Connection(target_identity, target_layer, innovation_number)
            node.connections.append(deepcopy(connection))

            self.network[input_layer+1].append(deepcopy(node))
                        
    def mutate_node(self): # connect two nodes that are not connected
        if len(self.network) > 2:
            input_layer = random.randint(0, len(self.network)-1)
            input_identity = random.randint(0, len(self.network[input_layer])-1)
            input_node = self.network[input_layer][input_identity]

            for layer in self.network[input_layer+1: len(self.network)]:
                for node in layer:
                    found = False
                    for connection in input_node.connections:
                        if connection.target_identity == node.identity and connection.target_layer == node.layer_number:
                            found = True
                    
                    if found == False:
                        innovation_number = self.finding_innovation(input_identity, node.identity)
                        connection = Connection(node.identity, node.layer_number, innovation_number)
                        input_node.connections.append(deepcopy(connection))
                        break
                
                if found == False: break
                
    def mutate_enable_disable(self):
        input_layer = random.randint(0, len(self.network)-2)
        input_identity = random.randint(0, len(self.network[input_layer])-1)
        input_node = self.network[input_layer][input_identity]
        connection = input_node.connections[random.randint(0, len(input_node.connections)-1)]

        if connection.enabled == True: connection.enabled = False
        else: connection.enabled = True
    
    def mutate_weight(self):
        input_layer = random.randint(0, len(self.network)-2)
        input_identity = random.randint(0, len(self.network[input_layer])-1)
        input_node = self.network[input_layer][input_identity]
        connection = input_node.connections[random.randint(0, len(input_node.connections)-1)]
        factor = random.uniform(0, 1)

        connection.weight *= factor

    def mutate_initialize_weight(self):
        input_layer = random.randint(0, len(self.network)-2)
        input_identity = random.randint(0, len(self.network[input_layer])-1)
        input_node = self.network[input_layer][input_identity]
        connection = input_node.connections[random.randint(0, len(input_node.connections)-1)]
        value = random.uniform(-2, 2)

        connection.weight = value

    def mutation(self):
        choice = random.randint(0, 10)
        if choice == 0:
            self.mutate_link() # problem
        if choice == 1:
            self.mutate_node()
        if choice == 2 or choice == 5 or choice == 8:
            self.mutate_enable_disable()
        if choice == 3 or choice == 6 or choice == 9:
            self.mutate_weight()
        if choice == 4 or choice == 7 or choice == 10:
            self.mutate_initialize_weight()
