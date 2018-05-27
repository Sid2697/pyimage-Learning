# importing necessary packages
from neuralnetwork import NeuralNetwork
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# define our 2-2-1 neural network and train it
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, Y, epochs=20000)

# now that our network is trained, loop over the XOR data points
for (x, targets) in zip(X, Y):
    # make the prediction on the data point and display the result
    # to the console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data = {}, ground truth = {}, pred = {:.4f}, step = {}".format(x, targets[0], pred, step))
