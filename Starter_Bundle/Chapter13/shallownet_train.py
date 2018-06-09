# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from siddhantbansal.preprocessing import ImageToArrayPreprocessor
from siddhantbansal.preprocessing import SimplePreprocessor
from siddhantbansal.preprocessing import SimpleDatasetLoader
from siddhantbansal.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input datset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, label) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

# partitioning the data
(trainX, testX, trainY, testY) = train_test_split(data, label, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# save the network to the disk
print('[INFO] serializing network...')
model.save(args['model'])

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'panda']))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
