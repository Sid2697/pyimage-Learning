# import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    # Compute the sigmoid value of a given input
    return 1.0 / (1.0 + np.exp(-x))


def predict(X, W):
    # Takes the dot product between features and weight matrix
    preds = sigmoid_activation(X.dot(W))
    # apply a step function to threshod the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    # Return the predictions
    return preds


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help='# of epochs')
ap.add_argument("-a", "--alpha", type=float, default=0.01, help='learning rate')
args = vars(ap.parse_args())

# generate a two class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=2)
y = y.reshape(y.shape[0], 1)

# insert a column of 1's as the last entry in the feature
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# Pattition the data into train and test splits
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):
    # take the dot product between the feature 'X' and the weight
    # matrix 'W', then pass this value through sigmoid activation function
    # thereby giving our prediction to the dataset
    preds = sigmoid_activation(trainX.dot(W))

    # now that we have our predictions we need to determine the
    # 'error', which is the difference betweem our predictions and true value
    error = preds - trainY
    loss = np.sum(error**2)
    losses.append(loss)

    # the gradient descent update is the dot product between our
    # features and the error in predictions
    gradient = trainX.T.dot(error)

    # in the update stage, all we need to do is 'nudge' the weight
    # matrix in the negative direction of the gradient
    W += -args['alpha'] * gradient

    # check to see if an update need to be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY.ravel(), s=30)

# construst a figure that plots loss overtime
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Traning loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
