import numpy as np
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

        # specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W) - 1

        self.W = W
        self.b = b

        # specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = len(W) * len(b[0])

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # specify a matrix for storing output values
        y = np.zeros((len(x), self.dataset.K))
        # 1. Specify a loop over all the datapoints        
            
        for i, x_i in enumerate(x):
            # Step 2: Specify input layer (2x1 matrix)
            h = np.reshape(x_i, (2, 1))  # Assuming each input is 2-dimensional
            
            # Step 3: Loop over each hidden layer
            for j in range(len(self.W)): 
                # Multiply weight matrix with output from the previous layer, add bias, and apply activation
                h = np.dot(self.W[j], h) + self.b[j]
                h = activation(h, self.activation)
            
            # Step 4: Specify the final layer with softmax activation
            h = activation(h, 'softmax')
            
            # Store result for this data point
            y[i] = h.flatten()

        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # Feedforward on training set
        y_train_pred = self.feedforward(self.dataset.x_train)
        # formulate the training loss and accuracy of the MLP
        train_loss = np.mean((y_train_pred - self.dataset.y_train_oh)**2)
        train_acc = np.mean(np.argmax(y_train_pred, 1) == self.dataset.y_train)

    
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # Feedforward on test set
        y_test_pred = self.feedforward(self.dataset.x_test)
        # formulate the test loss and accuracy of the MLP
        test_loss = np.mean((y_test_pred - self.dataset.y_test_oh)**2)
        test_acc = np.mean(np.argmax(y_test_pred, 1) == self.dataset.y_test)

        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
