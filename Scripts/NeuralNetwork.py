
try:
    import cupy as cp
except:
    import numpy as cp

import matplotlib.pyplot as plt
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
            print(cp.array(l.W).size)
            s += cp.array(l.W).size
        return s
    
    # Add a layer to this model
    def add(self, layer):
        self.layers.append(layer)
        
    
    def back_propagation(self, x, y):
        
        pred = self.forward_propagation(x)
        
        # Compute Error
        error = self.J.compute(y_t = y, y_p = pred)
        
        # Calculate gradient
        # First step
        # signal is the input signal of the layer, we propagate backwards now
        
        #gradient = error.reshape(error.size, 1)

        # Second step, continuing backwards through the complete structure
        for i in reversed(range(0, len(self.layers))):
            
            # Get the layer
            layer = self.layers[i]
            
            # Use specified method for computing the gradient
            error = layer.compute_gradient(error)
            
        return sum(error)/len(error)
        
    def update_layers(self, alpha = 0.0003):
        
        #gradient_sum = 0
        for layer in self.layers:
            #if not(layer.gradients is None):
            #    gradient_sum += cp.sum(layer.gradients)
            layer.update(alpha)
        
    # Train on batch x with labels y, this runs for one epoch. Both inputs must be iterables!
    def batch_update(self, x, y, alpha = 0.0003):

        errors = []
        
        # For all examples in batch
        for i in range(len(x)):
            
            e = self.back_propagation(x[i], y[i])
            
            # Store statistics of the batch
            errors.append(e)

        
        # Update the layers once the gradients have been stored
        self.update_layers(alpha)
        
        errors = cp.sum(cp.linalg.norm(cp.array(errors), axis = 0))
        return errors
        
    def train(self, x, y, epochs = 3, batch_size = 1, alpha = 0.0003, shuffle = True):
            
        for e in range(epochs):
            index = 0
            errors = []
            
            preds = []
            for point in x:
                preds.append(self.predict(point))

            preds = cp.squeeze(cp.array(preds))
            self.J.set_latent_points(preds)
            self.J.batch_y_t = y
            
            # Shuffle the dataset
            if shuffle:
                
                shuffle_index = cp.arange(len(x))

                cp.random.shuffle(shuffle_index)
                
                x = x[shuffle_index]
                y = y[shuffle_index]
                
            while index < len(x):
                batch_examples = x[index : index + batch_size]
                batch_labels = y[index : index + batch_size]
                
                errors.append(self.batch_update(batch_examples, batch_labels, alpha))
                
                index = index + batch_size
            
            errors = cp.array(errors)
            
            self.history["error"].append(cp.sum(errors))
        print(self.history["error"][-1])
        plt.title("Error curve")
        plt.plot(cp.array(self.history["error"]).get())
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