import numpy as np
from sklearn import datasets

def get_mnist():
    data = datasets.load_digits()
    X = data.images
    y = data.target
    print(y[100])
    X = X.reshape(len(X), -1)/255
    y = np.expand_dims(y, -1)

    return X, y
# Generate densly connected data
def generate_dense(num_per_class, dim = 2, classes = 3, seed = None):
    
    if not seed is None:
        np.random.seed(seed)
        
    # Set destination list
    data = []
    labels = []
    # For every class
    for c in range(classes):
        
        # Make an empty list
        category_class = []
        
        # Generate the initial random point for the class
        category_class.append((np.random.random_sample(dim) - 0.5) * 10)
        
        # Generate desired amount of points, dependent on random previously generated points
        for i in range(1, num_per_class):
            new_point = (category_class[-1] + (np.random.random_sample(dim)-0.5))
            category_class.append(new_point)
            
        category_class = np.array(category_class)
        
        # Add the point into a single list
        [data.append(point) for point in category_class]
        
        labels = np.append(labels, [c] * num_per_class)
        
    shuffle_ix = np.arange(len(data))
    np.random.shuffle(shuffle_ix)
        
    data = np.array(data)[shuffle_ix]
    labels = np.expand_dims(labels[shuffle_ix],axis=-1)
        
    return np.array(data), labels