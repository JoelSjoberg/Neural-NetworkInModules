
import numpy as cp

import matplotlib.pyplot as plt
import time
from Scripts.loss_functions import *

class Model:
    def __init__(self, layers = None, loss = None):
        
        # Copy pretrained layers if given
        if layers == None:
            self.layers = []
        else:
            self.layers = layers
        
        self.J = loss
        self.history = {
            "error" : [],
            "acc" : []
        }
        
        self.update_iterator = 0
    
    # Return number of parameters in model
    def size(self):
        s = 0
        for l in self.layers:
            print(l, cp.array(l.W).size)
            s += cp.array(l.W).size
        return s
    
    # Add a layer to this model
    def add(self, layer):
        self.layers.append(layer)
        
    
    def back_propagation(self, x, y, regularization = True, lambd = 0.1):

        pred = self.forward_propagation(x)
                
        if regularization:
            reg_sum = 0
            for layer in self.layers:
                reg_sum +=  cp.sum(layer.get_params())

            reg_sum = 1/2 * reg_sum * lambd

            self.J.reg_term = reg_sum

        else:
            self.J.reg_term = 0

        # Compute Error
        error = self.J.compute(y_t = y, y_p = pred)

        # Calculate gradient
        # First step
        gradient = self.J.derivative(pred)
        # Second step, continuing backwards through the complete structure
        for i in reversed(range(0, len(self.layers))):
            # Use specified method for computing the gradient
            gradient = self.layers[i].compute_gradient(gradient, regularization = regularization, lambd = lambd)
            
        return sum(error)/len(error)
        
    def update_layers(self, alpha = 0.0003):
        
        for layer in self.layers:
            layer.update(alpha)
        
    # Train on batch x with labels y, this runs for one epoch. Both inputs must be iterables!
    def batch_update(self, x, y, alpha = 0.0003, regularization = True, lambd = 0.1):
   
        e = self.back_propagation(x, y, regularization=regularization, lambd = lambd)

        # Update the layers once the gradients have been stored
        self.update_layers(alpha)
        
        err = cp.sum(cp.linalg.norm(cp.array(e), axis = 0))
        return err
        
    def train(self, x, y, epochs = 3, batch_size = 32, alpha = 0.0003, shuffle = True, balanced_batch = False, regularization = True, lambd = 0.1):
        
        for e in range(epochs):
            # Measure time per epoch:
            start_time = time.time()
            index = 0
            errors = []
            
            # Shuffle the dataset
            if shuffle:
                
                shuffle_index = cp.arange(len(x))
                cp.random.shuffle(shuffle_index)
                
                x = x[shuffle_index]
                y = y[shuffle_index]


            while index < len(x):

                batch_examples = x[index : index + batch_size]
                batch_labels = y[index : index + batch_size]
                #if not len(cp.unique(batch_labels)) < len(cp.unique(y)):

                errors.append(self.batch_update(batch_examples, batch_labels, alpha, regularization=regularization, lambd=lambd))
                    
                index = index + batch_size

            end_time = time.time()

            # Print the updates in a single print line
            error = cp.mean(errors)
            if e < epochs-1:
                end = "\r"
            else:
                end = "\n"
            print("Epoch:", e, "Loss:", cp.round(error, 5), " Updates per second:", str(cp.round(len(x) / (end_time - start_time), 2)), end=end)
            
            # Save error to history to get statistics
            self.history["error"].append(error)

        # Plot the history of error statistics
        plt.title("Error curve")
        plt.plot(cp.array(self.history["error"]))
        plt.show()

    # Feed forward, save the derivatives and such
    def forward_propagation(self, x):
        
        for layer in self.layers:
            x = layer.feed_forward(x, train = True)
        return x

    def predict(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x, train = False)
        return x