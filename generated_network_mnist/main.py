import numpy as np
from copy import deepcopy
from tqdm import tqdm
from network_generation import Network
from in2 import Passing
from keras.datasets import mnist

batch_size = 5
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:batch_size].reshape(batch_size, 28*28)/255, y_train[0:batch_size])
x_test, y_test = (x_test[0:batch_size].reshape(batch_size, 28*28)/255, y_test[0:batch_size])

network = Network()
for i in range(10):
    network.mutation()

    #print(" ")
    #print(network.network_architecture())


passing = Passing()
passing.network = network.network

epochs = 5
for epoch in range(epochs):
    average_error = 0

    for i in tqdm(range(len(images))):
        image = images[i:i+1]
        ground_truth = np.zeros((1, 10))
        ground_truth[0][labels[i]] = 1
        error = 0

        value = passing.forward(image)
        error += np.sum((ground_truth - value)**2)

        passing.backward(value - ground_truth)

        average_error += error
    
    print(f"average error {average_error}")
