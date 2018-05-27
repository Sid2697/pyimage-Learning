# Importing necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from simpledatasetloader import SimpleDatasetLoader
from simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the input dataset')
args = vars(ap.parse_args())

# grab the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load the dataset from the disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sd1 = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sd1.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# splitting the data into train and test set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

# loop over our set of regulizers
for r in (None, 'l1', 'l2', 'elasticnet'):
    # train the SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss='log', penalty=r, max_iter=1000, learning_rate='constant', eta0=0.01, n_jobs=-1, verbose=50)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc * 100))
