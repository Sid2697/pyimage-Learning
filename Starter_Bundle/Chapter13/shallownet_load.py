# import the necessary packages
from siddhantbansal.preprocessing import ImageToArrayPreprocessor
from siddhantbansal.preprocessing import SimplePreprocessor
from siddhantbansal.preprocessing import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# constrct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to pre-trained')
args = vars(ap.parse_args())

# initialize the class lebels
classLabels = ['cat', 'dog', 'panda']

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print('[INFO] sampling images...')
imagePaths = np.array(list(paths.list_images(args['dataset'])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
isp = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the shape [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, isp])
(data, lables) = sdl.load(imagePaths)
data = data.astype('float') / 255.0

# load the pre-trained network
print('[INFO] loading pre-trainied network...')
model = load_model(args['model'])

# make predictions on the images
print('[INFO] predicting...')
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
