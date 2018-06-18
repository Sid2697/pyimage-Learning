# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from siddhantbansal.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from siddhantbansal.datasets import SimpleDatasetLoader
from siddhantbansal.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers import Input
from keras.applications import VGG16
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# grab the list of images that we'll be describing, then extract the class label names from the image paths
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers tp vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a set of FC layers followed by a aoftmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model of the base model -- this will become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all the layers in the base model and freeze them so they will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model (this needs to be done after setting our layers to being non-trainable)
print('[INFO] compiling model...')
opt = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the model, this time fine-tuning *both* the final set of CONV layers along with the set of FC layers
print('[INFO] fine-tuning model...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=100, steps_per_epoch=len(trainX) // 32, verbose=1)

# evalaute the network on the fine-tuned model
print('[INFO] evaluating after fine-tuning...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_name=classNames))

# save the model to disk
print('[INFO] serializing model...')
model.save(args['model'])
