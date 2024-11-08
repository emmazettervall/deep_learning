import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == '':
        # TODO
    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = 

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = 0

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
        y = 

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        # 2. Specify the input layer (2x1 matrix)
        # 3. For each hidden layer, perform the MLP operations
        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function
        # 4. Specify the final layer, with 'softmax' activation
        
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        train_loss = 
        train_acc = 
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        test_loss = 
        test_acc = 
        print("\tTest loss:      %0.4f"%train_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
