from Scripts.activations import*
import numpy as np

class Layer:

    def get_params(self):
        pass

    def feed_forward(self, inp, train = True):
        pass

    def compute_gradient(self, err, regularization = False, lambd = 0):
        return err
    
    def update(self, alpha):
        pass

    def get_params(self):
        return []

class Dense(Layer):

    def __init__(self, layer_size, activation = identity,
                  W = None, standardize = False,
                  trainable = True, bias = True):
        
        if W is None:
            if bias:
                size = (layer_size[0] + 1, layer_size[1])
            else:
                size = (layer_size[0], layer_size[1])
            self.W = np.random.normal(loc = 0, scale = 2/(size[0] + size[1]), size = size).astype(np.double)

        else:
            self.W = W
        
        self.activation = activation
        self.gradients = []
        self.standardize = standardize
        self.trainable = trainable
        self.bias = bias

        self.batch_size = None
        self.signal = None

        self.velocity = np.zeros_like(self.W)
        self.mu = 0.9
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True, axis = 0):

        # Append bias
        if self.bias:
            bias = np.expand_dims(np.ones(inp.shape[axis]), -1)
            inp = np.append(inp, bias, axis = 1)


        self.signal = inp

        # Multiply input with relative parameters, then sum them up
        dot = np.dot(self.signal, self.W)

        if train:
            # Derivative of element-wise activation
            self.derivative = get_derivative(self.activation, dot)
        
        # Element-wise activations
        return self.activation(dot)
    
    def compute_gradient(self, err, regularization = False, lambd = 1):

        # w.r.t. x      
        err = np.multiply(err, self.derivative)

        error = err.dot(self.W.T)

        if self.bias:
            error = error[:, :-1]

        # w.r.t W, how should W be changed?
        gradient = err.T.dot(self.signal).T
        self.gradients.append(gradient)

        # If regularization is active
        if regularization:
            self.gradients[-1] += self.W * lambd
        
        return (error)

        
    def update(self, alpha):
        
        
        if self.trainable:
            prev_v = np.copy(self.velocity)

            grads = np.sum(self.gradients, axis=0)/len(self.gradients)
            self.velocity = self.mu * self.velocity - alpha * grads
            
            self.W += -self.mu * prev_v + (1 + self.mu) * self.velocity

            #print(grads)
            #self.W -= alpha * grads

        # Remember to reset gradient after update
        self.gradients = []

    def __str__(self):
        return "Dense Layer"

class FeatureImportance(Layer):

    def __init__(self, size, trainable = True):

        self.W = np.random.random_sample((1, size))
        self.gradients = []
        self.trainable = trainable
        self.velocity = np.zeros_like(self.W)
        self.mu = 0.9

    def feed_forward(self, inp, train=True):
        
        self.signal = inp

        mult = self.signal * self.W

        if train:
            # Derivative of element-wise activation
            self.derivative = np.ones_like(mult)

        return mult
    
    def compute_gradient(self, err, regularization=False, lambd=0):
        # w.r.t. x      
        err = np.multiply(err, self.derivative)
        error = err.dot(self.W.T)

        # w.r.t W, how should W be changed?
        gradient = err.T.dot(self.signal).T
        self.gradients.append(gradient)

        # If regularization is active
        if regularization:
            self.gradients[-1] += self.W * lambd
        
        return (error)

    def update(self, alpha):
        
        if self.trainable:
            prev_v = np.copy(self.velocity)
            grads = np.sum(self.gradients, axis=1)/len(self.gradients)
            self.velocity = self.mu * self.velocity - alpha * grads
            self.W += -self.mu * prev_v + (1 + self.mu) * self.velocity
            self.W = np.clip(self.W, 0, 1)

        # Remember to reset gradient after update
        self.gradients = []

    def __str__(self):
        return "Feature Importance Layer"

class Graph_Layer:

    def __init__(self, layer_size, activation = identity, W = None, standardize = False,
                  trainable = True):
        
        if W is None:

            size = (layer_size[0], layer_size[0])
            self.W = np.random.normal(loc = 0, scale = 2/(size[0] + size[0]), size = size)
        else:
            self.W = W

        self.mask = np.random.randint(0, 2, size = self.W.shape)        
        self.activation = activation
        self.gradients = []
        self.standardize = standardize
        self.trainable = trainable
        self.batch_size = 1
        self.signal = None
    
    def get_params(self):
        return np.multiply(self.W, self.mask)
    
    def feed_forward(self, inp, train = True, axis = 0):
        self.batch_size = inp.shape[0]

        self.signal = inp
        
        dot = np.dot(inp, np.multiply(self.W, self.mask))
        dot += self.signal
        dot *= 0.5
        if train:
            
            self.derivative = get_derivative(self.activation, dot)
        
        return self.activation(dot)
    
    
    def compute_gradient(self, err):

        # w.r.t. x      
        err = np.multiply(err, self.derivative)

        error = err.dot(self.W.T)

        # w.r.t W, how should W be changed?
        gradient = err.T.dot(self.signal).T
        self.gradients = gradient
        
        return (error)
        
    def update(self, alpha):
        
        if self.trainable:
            self.W -= alpha * self.gradients.astype("float64") * 1/self.batch_size

            if self.standardize:
                self.W = (self.W - cp.mean(self.W))/cp.std(self.W)

        # Remember to reset gradient after update
        self.gradients = []

class Noise(Layer):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.W = []

    def get_params(self):
        return []  
    
    def feed_forward(self, inp, train = True):
        
        self.signal = inp
        
        if train:
            self.signal = inp + np.random.normal(self.mean, scale = self.std, size = inp.shape)
        
        return self.signal
    
class Dropout(Layer):
    
    def __init__(self, rate = 0.1):
        
        self.rate = rate
        self.W = []
        self.mask = None
        self.gradients = None
        
    def get_params(self):
        return self.W   
    
    def feed_forward(self, inp, train = True, eps = 0.00000001):
        
        self.signal = inp
        
        norm = np.linalg.norm(self.signal, axis = 0)
        if train == False:
            return self.signal
    
        self.mask = cp.random.random_sample(inp.shape)
        
        self.mask = cp.where(self.mask > self.rate, 1, 0)

        out = cp.multiply(inp, self.mask)
        drop_norm = np.linalg.norm(out, axis = 0)

        out = (out/(drop_norm + eps)) * norm 

        return out

class BatchNormalization:

    def __init__(self):

        # Estimated from the batch for every neuron/channel
        # After training, estimate these using the entire training set
        self.mean = None
        self.var = None

        # Learnable parameters
        self.beta = None
        self.gamma = None

    def get_params(self):

        return (self.beta, self.gamma)
    
    def feed_forward(self, inp, train = True):

        self.signal = inp

        if train:
            self.mean = np.mean(inp, axis = 0)
            self.var = np.var(inp, axis = 0)

        
