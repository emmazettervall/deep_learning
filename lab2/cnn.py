import numpy as np
from scipy import signal
import skimage
import data_generator

# Different activations functions
def activation(x, activation):
    # specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'softmax':
        exps = np.exp(x - np.max(x))  # Stability trick
        return exps / np.sum(exps)
    else:
        raise Exception("Activation function is not valid", activation)

# 2D convolutional layer
def conv2d_layer(h,     # activations from previous layer, shape = [height, width, channels prev. layer]
                 W,     # conv. kernels, shape = [kernel height, kernel width, channels prev. layer, channels this layer]
                 b,     # bias vector
                 act    # activation function
):
    # TODO: implement the convolutional layer
    # 1. Specify the number of input and output channels
    CI = h.shape[-1] # Number of input channels
    CO = W.shape[-1] # Number of output channels
    output = np.zeros((h.shape[0], h.shape[1], CO))
    # 2. Setup a nested loop over the number of output channels 
    #    and the number of input channels
    for i in range(CO):
        for j in range(CI):
            
    # 3. Get the kernel mapping between channels i and j
            kernel = W[:,:,j,i]
    # 4. Flip the kernel horizontally and vertically (since
    #    We want to perform cross-correlation, not convolution.
    #    You can, e.g., look at np.flipud and np.fliplr
    #    functions in the numpy library)
            kernel = np.flipud(np.fliplr(kernel))
    # 5. Run convolution (you can, e.g., look at the convolve2d
    #    function in the scipy.signal library)
            conv = signal.convolve2d(h[:,:,j], kernel, mode='same')
    # 6. Sum convolutions over input channels, as described in the 
    #    equation for the convolutional layer
            if j == 0:
                conv_sum = conv
            else:
                conv_sum += conv
    # 7. Finally, add the bias and apply activation function
        output[:, :, i] = activation(conv_sum + b[i], act)
    # TODO: return number of input channels, the output and number of output channels
    return output



# 2D max pooling layer
def pool2d_layer(h):  # activations from conv layer, shape = [height, width, channels]
    # TODO: implement the pooling operation
    # 1. Specify the height and width of the output
    sy, sx = 2, 2

    # 2. Specify array to store output
    ho = np.zeros((h.shape[0]//sy, h.shape[1]//sx, h.shape[2]))

    # 3. Perform pooling for each channel.
    #    You can, e.g., look at the measure.block_reduce() function
    #    in the skimage library
    for i in range(h.shape[2]):
        ho[:,:,i] = skimage.measure.block_reduce(h[:,:,i], (sy, sx), np.max)
    
    return ho


# Flattening layer
def flatten_layer(h): # activations from conv/pool layer, shape = [height, width, channels]
    # TODO: Flatten the array to a vector output.
    # You can, e.g., look at the np.ndarray.flatten() function
    # in the numpy library
    return h.flatten()

# Dense (fully-connected) layer
def dense_layer(h,   # Activations from previous layer
                W,   # Weight matrix
                b,   # Bias vector
                act  # Activation function
):
    # TODO: implement the dense layer.
    # You can use the code from your implementation
    # in Lab 1. Make sure that the h vector is a [Kx1] array.
    h = h[:, np.newaxis]
    z = np.matmul(W, h) + b # Shape [M, 1]
    activated_output = activation(z, act)
    return activated_output[:,0]
    
#---------------------------------
# Our own implementation of a CNN
#---------------------------------
class CNN:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset

    # Set up the CNN from provided weights
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        lname,               # List of layer names
        activation='relu'    # Activation function of layers
    ):
        self.activation = activation
        self.lname = lname

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model
        #       (convolutional kernels, weight matrices, and bias vectors)
        self.N = sum(np.prod(np.array(w).shape) for w in self.W) + \
            sum(np.prod(np.array(b).shape) for b in self.b)
        print('Number of model weights: ', self.N)

    # Feedforward through the CNN of one single image
    def feedforward_sample(self, h):

        # Loop over all the model layers
        for l in range(len(self.lname)):
            act = self.activation
            
            if self.lname[l] == 'conv':
                h = conv2d_layer(h, self.W[l], self.b[l], act)
            elif self.lname[l] == 'pool':
                h = pool2d_layer(h)
            elif self.lname[l] == 'flatten':
                h = flatten_layer(h)
            elif self.lname[l] == 'dense':
                if l==(len(self.lname)-1):
                    act = 'softmax'
                h = dense_layer(h, self.W[l], self.b[l], act)
        return h

    # Feedforward through the CNN of a dataset
    def feedforward(self, x):
        # Output array
        y = np.zeros((x.shape[0],self.dataset.K))

        # Go through each image
        for k in range(x.shape[0]):
            if self.verbose and np.mod(k,1000)==0:
                print('sample %d of %d'%(k,x.shape[0]))

            # Apply layers to image
            y[k,:] = self.feedforward_sample(x[k])   
            
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the CNN.
        # Assume the cross-entropy loss.
        # For the accuracy, you can use the implementation from Lab 1.
        # Feedforward on training set
        y_train_pred = self.feedforward(self.dataset.x_train)
        train_loss = - \
            np.mean(
                np.log(y_train_pred[np.arange(len(y_train_pred)), self.dataset.y_train]))
        train_acc = np.mean(np.argmax(y_train_pred, 1) == self.dataset.y_train)
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the CNN
        y_test_pred = self.feedforward(self.dataset.x_test)
        test_loss = - \
            np.mean(
                np.log(y_test_pred[np.arange(len(y_test_pred)), self.dataset.y_test]))
        test_acc = np.mean(np.argmax(y_test_pred, 1) == self.dataset.y_test)
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
