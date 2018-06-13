# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from siddhantbansal.nn.conv import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True, help='path to weights directory')
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the range [0, 1]
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initilize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(widht=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([args['weights'], "weights-{epoch:03d}-{val_loss: .4f}.hdf5"])  # constructs filename
checkpoint = ModelCheckpoint(fname, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
