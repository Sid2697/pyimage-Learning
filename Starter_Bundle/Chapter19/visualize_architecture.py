# importing necessary packages
from siddhantbansal.nn.conv import LeNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file='/Users/siddhantbansal/Desktop/lenet.png', show_shapes=True)
