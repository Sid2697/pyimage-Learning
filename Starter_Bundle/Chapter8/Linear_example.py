# importing necessary packages
import numpy as np
import cv2

# initialize the class labels
labels = ['dog', 'cat', 'panda']
np.random.seed(1)

# Randomly initialize weight matrix and bias vector
# For trail we'll use some random value
W = np.random.randn(3, 3072)
b = np.random.randn(3)

orig = cv2.imread('/Users/siddhantbansal/Desktop/dog.png')
image = cv2.resize(orig, (32, 32)).flatten()

# Compute the output score
scores = np.dot(W, image) + b

# Loop over the label + scores and display them
for (label, score) in zip(labels, scores):
    print('[INFO] {}:{:.2f}'.format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our image
cv2.imshow("Image", orig)
cv2.waitKey(0)
