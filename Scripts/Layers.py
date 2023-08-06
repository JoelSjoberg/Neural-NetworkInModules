try:
    import cupy as np
except:
    import numpy as np


from Scripts.activations import *
# Super class of layers
class Layer:
    
    def __init__(self, size = None, W = None, activation = None, trainable = None, rate = 0.4, a = 0, standardize = False, iters = 0):
        
        self.size = size
        self.W = W
        self.activation = activation
        self.trainable = trainable
        self.rate = rate
        self.a = a
        self.standardize = standardize
        self.iters = iters
        
        self.signal = None
        self.derivative = None
        self.gradient = []
        
        
    def get_params(self):
        pass

class scaling:
    def __init__(self, size = 3, W = None, const = 10):
        
        if W is None:
            self.W = np.random.random_sample((size, 1)) * (10 - 5) * const
        else:
            self.W = W
        
        self.W = self.W.reshape((self.W.size, 1))
            
        self.gradients = []
        
    def get_params(self):
        return self.W 
    
    def feed_forward(self, inp, train = True):
        
        self.signal = inp.reshape(inp.size, 1)
        
        if train:
            self.derivative = self.W.reshape(self.W.size, 1)
        
        return np.squeeze(np.multiply(self.signal, self.W))

    def compute_gradient(self, err):
        
        error = err.reshape(err.size, 1)
        
        # Derivative w.r.t. W
        derivative = self.signal
        
        # Gradient is used to update alpha value
        gradient = error * derivative.reshape(derivative.size, 1)

        # Gradient w.r.t Input
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        self.gradients.append(gradient)
        
        return (error)
    
    def update(self, alpha):
        for gradient in self.gradients:
            self.W += -alpha * gradient * 1/len(self.gradients)
            
        # Remember to reset gradient after update
        self.gradients = []

class dropout:
    
    def __init__(self, rate = 0.1):
        
        self.rate = rate
        self.W = []
        self.mask = None
        self.gradients = None
        
    def get_params(self):
        return self.W   
    
    def feed_forward(self, inp, train = True):
        
        self.signal = inp
        
        if train == False:
            return self.signal
    
        self.mask = np.random.random_sample(inp.shape)
        
        self.mask = np.where(self.mask > self.rate, 1, 0)
        
        return np.multiply(inp, self.mask)
    
    def compute_gradient(self, err):
        
        return err
    
    def update(self, alpha):
        return None

class leaky_relu:
    def __init__(self, size, a = None):
        
        if a == None:
            self.W = np.random.random_sample((size, 1))
        else:
            self.W = np.ones(size) * a
            
        def leaky_relu(x):
            ind = np.argwhere(x < 0)
            x[ind] = self.W.reshape(x.shape)[ind] * x[ind]
            signal = x
            
            return np.array(signal)
        
        self.gradients = []
        self.signal = None
        self.activation = leaky_relu
        
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
        
        self.signal = np.array(inp)
        if train:
            self.derivative = get_derivative(self.activation, self.signal)
            self.derivative = np.array(self.derivative).T
        
        return self.activation(inp)
    
    
    def compute_gradient(self, err):
        
        error = err.reshape(err.size, 1)
        
        # Derivative w.r.t. W
        derivative = np.where(self.signal > 0, 0, self.signal)
        
        # Gradient is used to update alpha value
        gradient = error * derivative.reshape(derivative.size, 1)

        # Error is use dto update layers after this
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        self.gradients.append(gradient)
        
        return (error)
        
    def update(self, alpha):
        for gradient in self.gradients:
            self.W += -alpha * gradient * 1/len(self.gradients)
            
        # Remember to reset gradient after update
        self.gradients = []


class static_normalization_layer:
    def __init__(self, W= None, learnable = False):
        
        self.W = W
        def activation(x):
            maxim = np.max(np.abs(x))
            minim = np.min(x)
            return (x - minim)/(maxim - minim)
        
        self.gradients = []
        self.signal = None
        self.activation = activation
        self.learnable = learnable
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
    
        self.signal = np.array(inp)
        
        if train:
            self.derivative = get_derivative(self.activation, self.signal)
            self.derivative = np.array(self.derivative).T
        
        return self.activation(inp)
    
    
    def compute_gradient(self, err):
        
        error = err.reshape(err.size, 1)
        
        # Derivative w.r.t. W
        derivative = np.ones(err.size) * self.signal
        
        # Gradient is used to update alpha value
        gradient = error * derivative.reshape(derivative.size, 1)
        gradient = np.mean(gradient)

        # Error is used to update layers after this
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        self.gradients.append(gradient)
        
        return (error)
        
    def update(self, alpha):
        self.gradients = []

class adaptive_normalization_layer:
    def __init__(self, W= None, learnable = True):
        
        if W== None:
            self.W = np.random.random_sample() * 0.01
        else:
            self.W = W
        
        def activation(x):
            return x * self.W
        
        self.gradients = []
        self.signal = None
        self.activation = activation
        self.learnable = learnable
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
        
        self.signal = np.array(inp)
        
        if train:
            self.derivative = get_derivative(self.activation, self.signal)
            self.derivative = np.array(self.derivative).T
        
        return self.activation(inp)
    
    
    def compute_gradient(self, err):
        
        error = err.reshape(err.size, 1)
        
        # Derivative w.r.t. W
        derivative = np.ones(err.size) * self.signal
        
        # Gradient is used to update alpha value
        gradient = error * derivative.reshape(derivative.size, 1)
        gradient = np.mean(gradient)

        # Error is used to update layers after this
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        self.gradients.append(gradient)
        
        return (error)
        
    def update(self, alpha):
        if self.learnable:
            for gradient in self.gradients:
                self.W += -alpha * gradient * 1/len(self.gradients)
            
        # Remember to reset gradient after update
        self.gradients = []

class zscore_layer:
    def __init__(self, W = None, learnable = False):
        
        if learnable == False:
            self.W = [0, 1]
            def activation(x):
                return (x - np.mean(x))/ np.std(x)
            
        elif W is not None:
            self.W = W
            def activation(x):
                return (x - self.W[0])/ self.W[1]
        
        else:
            self.W = np.random.sample(2)
            def activation(x):
                return (x - self.W[0])/ self.W[1]
        
        self.gradients = []
        self.signal = None
        self.activation = activation
        self.learnable = learnable
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
        
        self.signal = np.array(inp)
        
        if train:
            # Derivative must be taken with respect to each element in vector x separately
            # compute derivative manually here
            # Derivative of mean w.r.t any given x_i is always 1/len(x)
            mean = np.mean(self.signal)
            std = np.std(self.signal)
            n = len(self.signal)

            std_der = (1/(2 * std)) * (2/n) * (self.signal - mean) * (1 - (1/n))
            derivative = (-1/(2*(std**2)) * std_der * (self.signal - mean)) + (1/std * (1 - 1/n))

            self.derivative = derivative
        
        return self.activation(inp)
    
    
    def compute_gradient(self, err):
        
        error = err.reshape(err.size, 1)
        
        # Derivative w.r.t mean and std (made independent for simplicity)
        
        der_mean = -1/self.W[1]
        
        der_std = (1/(self.W[1]**2)) * np.mean((self.signal - self.W[0]))
        
        self.gradients.append(np.array([der_mean, der_std]))
        
        # Derivative w.r.t. x
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        return (error)
        
    def update(self, alpha):
        if self.learnable:
            for gradient in self.gradients:
                self.W += -alpha * gradient * 1/len(self.gradients)
            
        # Remember to reset gradient after update
        self.gradients = []

class fullyConnected:
    def __init__(self, layer_size, activation = identity, W = None, standardize = False, trainable = True, bias = True):
        
        if W == None:
            if bias:
                size = (layer_size[0], layer_size[1]+1)
            else:
                size = (layer_size[0], layer_size[1])
            self.W = np.random.normal(loc = 0, scale = 2/(size[0] + size[1]), size = size)
        else:
            self.W = W
        
        self.activation = activation
        
        self.gradients = []
        self.standardize = standardize
        self.trainable = trainable
        self.bias = bias
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
        
        # Append bias
        if self.bias:
            inp = np.append(inp, 1)
        
        self.signal = inp
        
        if train:
            self.derivative = get_derivative(self.activation, self.W.dot(inp))
            self.derivative = np.array(self.derivative).T
        
        return self.activation(self.W.dot(inp))
    
    
    def compute_gradient(self, err):

        # w.r.t. x
        error = err.reshape(err.size, 1)
        
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        error = error.T.dot(self.W).T
        
        # remove the bias if present
        if self.bias:
            error = error[:-1]
        
        # w.r.t W
        gradient = err.reshape(err.size, 1)
        
        gradient = gradient * self.derivative.reshape(self.derivative.size, 1)
        
        gradient = gradient * self.signal.reshape(1, self.signal.size)
        
        self.gradients.append(gradient)
        
        return (error)
        
    def update(self, alpha):
        
        if self.trainable:
            for gradient in self.gradients:
                self.W -= alpha * gradient.astype("float64") * 1/len(self.gradients)

            if self.standardize:
                self.W = (self.W - np.mean(self.W))/np.std(self.W)

        # Remember to reset gradient after update
        self.gradients = []


class sporadic_activations:
    def __init__(self, size, W = None, activations = None, trainable = True):
        
        # exp_relu seems to be unstable at the moment, debug it and add it later
        activation_functions = [sine, cos, tanh, sigmoid, l_relu, relu, natural_l_relu, identity, cont_sigmoid]
        self.W = []
        self.gradients = []
        self.trainable = True
        if activations == None:
            self.activations = []
            for i in range(size):
                self.activations.append(activation_functions[np.random.randint(len(activation_functions))])
        else:
            self.activations = activations

    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
        
        self.signal = inp
        
        out_signals = []
        out_derivatives = []
        for i in range(len(self.activations)):
            out_signals.append(self.activations[i](self.signal[i]))
            out_derivatives.append(get_derivative(self.activations[i], self.signal[i]))
            
        if train:
            self.derivative = np.array(out_derivatives)
        
        return np.array(out_signals)
    
    
    def compute_gradient(self, err):

        # w.r.t. x
        error = err.reshape(err.size, 1)
        
        error = error * self.derivative.reshape(self.derivative.size, 1)

        return (error)
        
    def update(self, alpha):
        
        return None


class partiallyConnected:
    def __init__(self, size, activation = identity, W = None, standardize = False, rate = 0.4, trainable = True):
        
        if W == None:
            size = (size[0], size[1]+1)
            self.W = np.random.normal(loc = 0, scale = 2/(size[0] + size[1]+1), size = size)
        else:
            self.W = W
        
        
        mask = np.random.random_sample(self.W.shape)
        mask = np.where(mask > rate, 1, 0)
        self.mask = mask
        self.W = np.multiply(self.W, self.mask)
        
        self.activation = activation
        
        self.gradients = []
        self.standardize = standardize
        
        self.trainable = trainable
    
    def get_params(self):
        return self.W
    
    def feed_forward(self, inp, train = True):
        
        # Append bias
        icp = np.append(inp, 1)
        
        self.signal = inp
        
        if train:
            self.derivative = get_derivative(self.activation, self.W.dot(inp))
            self.derivative = np.array(self.derivative).T
        
        return self.activation(self.W.dot(inp))
    
    
    def compute_gradient(self, err):

        # w.r.t. x
        error = err.reshape(err.size, 1)
        
        error = error * self.derivative.reshape(self.derivative.size, 1)
        
        error = error.T.dot(self.W).T
        
        # remove the bias
        error = error[:-1]
        
        # w.r.t W
        gradient = err.reshape(err.size, 1)
        
        gradient = gradient * self.derivative.reshape(self.derivative.size, 1)
        
        gradient = gradient * self.signal.reshape(1, self.signal.size)
        
        self.gradients.append(gradient)
        
        return (error)
        
    def update(self, alpha):
        
        if self.trainable:
            for gradient in self.gradients:
                self.W -= alpha * gradient.astype("float64") * 1/len(self.gradients)
        
        if self.standardize:
            self.W = (self.W - np.mean(self.W[self.W != 0]))/np.std(self.W[self.W != 0])
            
        self.W = np.multiply(self.W, self.mask)
        # Remember to reset gradient after update
        self.gradients = []