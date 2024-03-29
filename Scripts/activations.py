
import numpy as cp

def identity(x):
    return x

def relu(x):
    # One line return statement is dope!
    return cp.where(x > 0, x, 0)

def l_relu(x, a = 0.01):
    
    # One line return statement is dope!
    return cp.where(x > 0, x, a*x)

def natural_l_relu(x, a = 0.01, b = 1):
    return cp.where(x > b, cp.log(cp.abs(x)), a * x)

def power_relu(x, a = 0.6):
    return cp.where(x >= 0, cp.power(x, a), -cp.power(-x, a))

def exp_relu(x, a = 0.1):
    return cp.where(x >= 0, cp.exp(x), x * a)

def sech(x):
    return 2 * cp.exp(x)/(cp.exp(2*x) + 1)

def sine(x):
    
    return cp.sin(x)

def cos(x):
    
    return cp.cos(x)

def tanh(x):
    return cp.tanh(x)

def sigmoid(x):
    return 1/(1+cp.exp(-x) + 1e-7)

def cont_sigmoid(x):
    return sigmoid(x)*x+(x*0.1)

def exp_div(x):
    return 1/cp.exp(x)

# The method for producing generalaized fourier waves!
def randomFourier(x, period = 10, limit1 = 100, limit2 = 100, bias = 0.0007, coef1 = 0.001, coef2 = 0.0001):
    #period = cp.random.normal(1000, 1000)
    #coef1 = cp.random.normal(1000, 1000)
    #coef2 = cp.random.normal(1000, 1000)
    
    #limit1 = cp.random.randint(1000)
    #limit2 = cp.random.randint(1000)
    #bias = cp.random.normal(1000, 1000)
    
    x = cp.matrix(cp.array(x).flatten()).T
    r1 = cp.matrix(cp.linspace(0, limit1, num = limit1)) +1
    
    r2 = cp.matrix(cp.linspace(0, limit2, num = limit2)) +1
    
    x_expanded1 = cp.multiply(r1, x)
    x_expanded2 = cp.multiply(r2, x)
    sine_part = coef1 * cp.sum(cp.sin(x_expanded1) * cp.pi/(period/2), axis = 1)
    cos_part = coef2 * cp.sum(cp.cos(x_expanded2) * cp.pi/(period/2), axis = 1)
    return cp.squeeze(cp.array(sine_part + cos_part + bias))

def relu_sig(x):
    return sigmoid(x) * relu(x)

def relu_tanh(x):
    return tanh(x) * l_relu(x)

def radial_basis_function(x, sig = 0.01, y_t = 0):
    return cp.exp(-cp.power(cp.abs(x - y_t), 2) * sig)

# Experimental: Yield the approximation of the derivative of function f at point x
def get_derivative(f, x, offset = 1e-8):
    
    # Get change between the surrounding space
    der = (f(x + offset) - f(x - offset))/(2*offset)

    return der




