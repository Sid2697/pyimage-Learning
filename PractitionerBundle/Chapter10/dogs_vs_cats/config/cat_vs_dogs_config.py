# define the path to the images directory
path = '/Users/siddhantbansal/Desktop/Python/Personal_Projects/Cats_vs_Dogs/dataset/kaggle_dogs_vs_cats/train'
# since we do not have access to validation data we need to take a number of images from train and test on them
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the file path to output training, validation and testing HDF5 files
TRAIN_HDF5 = '../dataset/kaggle_dogs_vs_cats/hdf5/train.hdf5'
VAL_HDF5 = '../dataset/kaggle_dogs_vs_cats/hdf5/val.hdf5'
TEST_HDF5 = '../dataset/kaggle_dogs_vs_cats/hdf5/test.hdf5'

# path to the output model file
MODEL_PATH = '/Users/siddhantbansal/Desktop/Python/Personal_Projects/Cats_vs_Dogs/dogs_vs_cats/output/alexnet_dogs_vs_cats.model'

# define the path to the dataset mean
DATASET_MEAN = '/Users/siddhantbansal/Desktop/Python/Personal_Projects/Cats_vs_Dogs/dogs_vs_cats/output/dogs_vs_cats_mean.json'

# define the path to the output directory used for storing plots, classification_reports etc.
OUTPUT_PATH = '/Users/siddhantbansal/Desktop/Python/Personal_Projects/Cats_vs_Dogs/dogs_vs_cats/output'
