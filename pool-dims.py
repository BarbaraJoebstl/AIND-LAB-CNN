from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()


'''

Max Pooling Layers in Keras
To create a max pooling layer in Keras, you must first import the necessary module:

from keras.layers import MaxPooling2D
Then, you can create a convolutional layer by using the following format:

MaxPooling2D(pool_size, strides, padding)
Arguments
You must include the following argument:

pool_size - Number specifying the height and width of the pooling window.
There are some additional, optional arguments that you might like to tune:

strides - The vertical and horizontal stride. If you don't specify anything, strides will default to pool_size.
padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.
NOTE: It is possible to represent both pool_size and strides as either a number or a tuple.

You are also encouraged to read the official documentation.

Example
Say I'm constructing a CNN, and I'd like to reduce the dimensionality of a convolutional layer by following it with a max pooling layer. Say the convolutional layer has size (100, 100, 15), and I'd like the max pooling layer to have size (50, 50, 15). I can do this by using a 2x2 window in my max pooling layer, with a stride of 2, which could be constructed in the following line of code:

    MaxPooling2D(pool_size=2, strides=2)
If you'd instead like to use a stride of 1, but still keep the size of the window at 2x2, then you'd use:

    MaxPooling2D(pool_size=2, strides=1)

    more info: http://cs231n.github.io/convolutional-networks/
'''


# As with neural networks, we add layers to the network by using the .add() method:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''
The network begins with a sequence of three convolutional layers, followed by max pooling layers. These first six layers are designed to take the input array of image pixels and convert it to an array where all of the spatial information has been squeezed out, and only information encoding the content of the image remains. The array is then flattened to a vector in the seventh layer of the CNN. It is followed by two dense layers designed to further elucidate the content of the image. The final layer has one entry for each object class in the dataset, and has a softmax activation function, so that it returns probabilities.

NOTE: In the video, you might notice that convolutional layers are specified with Convolution2D instead of Conv2D. Either is fine for Keras 2.0, but Conv2D is preferred.

Things to Remember
Always add a ReLU activation function to the Conv2D layers in your CNN. With the exception of the final layer in the network, Dense layers should also have a ReLU activation function.
When constructing a network for classification, the final layer in the network should be a Dense layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.
'''