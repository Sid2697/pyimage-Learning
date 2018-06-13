# import the necessary packages
from keras.applications import ResNet50, InceptionV3, VGG16, VGG19, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the input image')
ap.add_argument('-model', '--model', type=str, default='vgg16', help='name of pre-trained network to use')
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes inside Keras
MODELS = {'vgg16': VGG16,
          'vgg19': VGG19,
          'inception': InceptionV3,
          'resnet': ResNet50
          }

# ensure a valid model name was supplied via command line argument
if args['model'] not in MODELS.keys():
    raise AssertionError('The --model command line argument should be a key in the Models dictionary')

# initialize the input image shape (224x224 pixels)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args['model'] in ('inception'):
    inputShape = (299, 299)
    preprocess = preprocess_input

# load out the network weights from disk
print('[INFO] loading {}...', format(args['model']))
Network = MODELS(args['model'])
model = Network(weights='imagenet')

# load the input image using the keras helper utility while ensuring the image is resized to 'inputShape', the required input dimensions for the ImageNet pre-trained network
print('[INFO] loading and pre-processing image...')
image = load_img(args['image'], target_size=inputShape)
image = img_to_array(image)

# expand the dimenstions of the image
image = np.expand_dims(image, axis=0)

# preprocess the image
image = preprocess(image)

# classify the image
print('[INFO] classifying image with {}...'.format(args['model']))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions + probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%", format(i + 1, label, prob * 100))

# load the image via opencv, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args['image'])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, 'Label: {}'.format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX. 0.8, (0, 255, 0), 2)
cv2.imshow('Classification', orig)
cv2.waitKey(0)
