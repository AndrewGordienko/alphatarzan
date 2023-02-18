import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

random.seed(0)

batch_size = 10

class Node():
    def __init__(self, identity, layer_number):
        self.identity = identity
        self.layer_number = layer_number
        self.value = 0
        self.delta = 0

        self.connections = []

# connections
class Connection():
    def __init__(self, target_identity, target_layer):
        self.target_identity = target_identity
        self.target_layer = target_layer
        self.weight = random.uniform(-1, 1)

        self.enabled = True
        self.innovation = None

class Passing():
    def __init__(self):
        self.network = None
        self.lr = 0.005
        
    def find_index(self, layer_index, target_identity):
        counter = 0
        for node in self.network[layer_index]:
            if node.identity == target_identity:
                return counter
            counter += 1
        
        print("it broke")
        return 0
    
    def ReLU(self, x):
        return x * (x > 0)

    def relu2deriv(self, input):
        return int(input > 0)
    

    def forward(self, image):
        index = 0
        for layer in self.network:
            node_index = 0
            for node in layer:
                node.value = 0
                if index == 0: node.value = image[0][node_index]
                node_index +=  1
            index += 1
        
        layer_counter = 0
        for layer in self.network:
            for node in layer:
                for connection in node.connections:
                    if connection.enabled:
                        target_index = self.find_index(connection.target_layer, connection.target_identity)
                        self.network[connection.target_layer][target_index].value += node.value * connection.weight
            
            if layer_counter != len(self.network):
                for node in self.network[layer_counter + 1]:
                    node.value = self.ReLU(node.value)
        
        values = []
        for node in self.network[len(self.network)-1]: values.append(node.value)
        return values

    def backward(self, error):
        index = 0
        for layer in self.network:
            node_index = 0
            for node in layer:
                node.delta = 0
                if index == len(self.network)-1: node.delta = error[0][node_index]
                node_index +=  1
            index += 1
        
        layer_index = 0
        for layer in reversed(self.network):
            if layer_index != 0:
                for node in layer:
                    for connection in node.connections:
                        if connection.enabled:
                            target_index = self.find_index(connection.target_layer, connection.target_identity)
                            node.delta += connection.weight * self.network[connection.target_layer][target_index].delta
                for node in layer:
                    node.delta *= self.relu2deriv(node.value)
            layer_index += 1

        layer_index = 0
        for layer in reversed(self.network):
            if layer_index != 0:
                for node in layer:
                    for connection in node.connections:
                        target_index = self.find_index(connection.target_layer, connection.target_identity)
                        connection.weight -= self.lr * node.value * self.network[connection.target_layer][target_index].delta
            layer_index += 1