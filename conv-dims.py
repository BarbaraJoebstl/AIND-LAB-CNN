from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()

'''
The depth of a convolutional layer is always equal to the number of filters.

Take note of how the number of parameters in the convolutional layer changes. This corresponds to the value under Param # in the printed output. In the figure above, the convolutional layer has 80 parameters.

Also notice how the shape of the convolutional layer changes. This corresponds to the value under Output Shape in the printed output. In the figure above, None corresponds to the batch size, and the convolutional layer has a height of 100, width of 100, and depth of 16.
Formula: Number of Parameters in a Convolutional Layer
The number of parameters in a convolutional layer depends on the supplied values of filters, kernel_size, and input_shape. Let's define a few variables:

K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
D_in - the depth of the previous layer
Notice that K = filters, and F = kernel_size. Likewise, D_in is the last value in the input_shape tuple.

Since there are F*F*D_in weights per filter, and the convolutional layer is composed of K filters, the total number of weights in the convolutional layer is K*F*F*D_in. Since there is one bias term per filter, the convolutional layer has K biases. Thus, the number of parameters in the convolutional layer is given by K*F*F*D_in + K.
Formula: Shape of a Convolutional Layer
The shape of a convolutional layer depends on the supplied values of kernel_size, input_shape, padding, and stride. Let's define a few variables:

K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
S - the stride of the convolution
H_in - the height of the previous layer
W_in - the width of the previous layer
Notice that K = filters, F = kernel_size, and S = stride. Likewise, H_in and W_in are the first and second value of the input_shape tuple, respectively.

The depth of the convolutional layer will always equal the number of filters K.

If padding = 'same', then the spatial dimensions of the convolutional layer are the following:

height = ceil(float(H_in) / float(S))
width = ceil(float(W_in) / float(S))
If padding = 'valid', then the spatial dimensions of the convolutional layer are the following:

height = ceil(float(H_in - F + 1) / float(S))
width = ceil(float(W_in - F + 1) / float(S))

'''