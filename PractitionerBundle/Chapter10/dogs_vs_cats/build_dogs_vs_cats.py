# import the necessary packeges
from config import cat_vs_dogs_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from siddhantbansal.preprocessing import AspectAwarePreprocessor
from siddhantbansal.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab path to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[2].split('.')[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the testing split from the training data
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the validation data
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation and testing image paths along with their corresponding labels and output HDG5 files
datasets = [('train', trainPaths, trainPaths, config.TRAIN_HDF5), ('val', valPaths, valLabels, config.VAL_HDF5), ('test', testPaths, testLabels, config.TEST_HDF5)]

# initialize the image preprocessor and the lists of RGB channels averages
aap = AspectAwarePreperocessor(256, 256)
(R, G, B) = ([], [], [])

# loop over the dataset tupels
for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 dataset
    print('[INFO] buiding {}...'.format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    # initialize the progress bar
    widgets = ['Building Dataset: ', progressbar.Pregentage(), " ", progress.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building the training dataset, then compute the mean of
        # each channel in the image, then update the respective lists
        if dType == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label # to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()

# construct a dictionary of averages, then searize the mean to a JSON file
print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()
