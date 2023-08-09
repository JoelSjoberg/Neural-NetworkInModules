try:
    import cupy as cp
except:
    import numpy as cp

# Generate densly connected data
def generate_dense(num_per_class, dim = 2, classes = 3, seed = None):
    
    if not seed is None:
        cp.random.seed(seed)
        
    # Set destination list
    data = []
    labels = []
    # For every class
    for c in range(classes):
        
        # Make an empty list
        category_class = []
        
        # Generate the initial random point for the class
        category_class.append((cp.random.random_sample(dim) - 0.5) * 10)
        
        # Generate desired amount of points, dependent on random previously generated points
        for i in range(1, num_per_class):
            new_point = (category_class[-1] + (cp.random.random_sample(dim)-0.5))
            category_class.append(new_point)
            
        category_class = cp.array(category_class)
        
        # Add the point into a single list
        [data.append(point) for point in category_class]
        
        labels = cp.append(labels, [c] * num_per_class)
        
    shuffle_ix = cp.arange(len(data))
    cp.random.shuffle(shuffle_ix)
        
    data = cp.array(data)[shuffle_ix]
    labels = labels[shuffle_ix]
        
    return cp.array(data), labels