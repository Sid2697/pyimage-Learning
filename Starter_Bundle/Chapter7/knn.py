# importing the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder  # A hepler utility to convert labels represented as strings to integers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
import matplotlib.pyplot as plt
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1, help='# of the nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs for k-NN distance (-1 uses all available cores)')
args = ap.parse_args()
# args = vars(ap.parse_args())
# print(args)

# grab the list of images that we'll be describing
print('[INFO] loading images...')
imagesPaths = list(paths.list_images(args.dataset))


# initialze the image processor, load the dataset from disk
# and resape the data matrix
sp = SimplePreprocessor(32, 32)
sd1 = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sd1.load(imagesPaths, verbose=500)  # verbose is used for yielding more information about the on going process.
data = data.reshape((data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))


# show some information on the memory consumption of the images
print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)


# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
# train and evaluate k-NN classifier on the raw pixel intensities
print('[INFO] evaluating k-NN classifier...')
model = KNeighborsClassifier(n_neighbors=args.neighbors, n_jobs=args.jobs, algorithm = 'brute')
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
# print(classification_report(testY, model.predict(testX)))
