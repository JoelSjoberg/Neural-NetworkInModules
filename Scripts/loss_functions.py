try:
    import cupy as cp
except:
    import numpy as cp

from Scripts.activations import get_derivative
import matplotlib.pyplot as plt

class Loss:
    def __init__(self):
        self.y_t = None
        self.batch_t = None

class MSE:
    def __init__(self):
        self.y_t = None
        self.batch_y_t = None
    
    def set_yt(self, y_t):
        self.y_t = y_t
    
    def J(self, y_p):

        error = cp.square(self.y_t - y_p) * 1/2
        return cp.nan_to_num(error)
        
    def compute(self, y_p, y_t):
        self.y_t = y_t
            
        return self.J(y_p)
    
    def derivative(self, x):
        x = x.ravel()
        return get_derivative(self.J, x)
            
    def set_latent_points(self, points):
        pass

class Cluster:
    def __init__(self, points, n = 3):
        self.points = points
        self.connections = []
        self.connection_labels = []
        self.num_neighbours = n
        
        self.y_t = None
        self.batch_y_t = None
        self.current_label = None
        
    def set_latent_points(self, points):
        self.points = points
    
    def set_y(self, y):
        self.y_t = y
        
    def compute_min_connections(self):
        connections = []
        connection_labels = []

        # prims algorithm for min spann tree
        tree = [0]
        visited = cp.zeros(len(self.points)).astype("bool")
        visited[tree[0]] = True
        
        for i, point in enumerate(self.points):
            dists = []
            for node in tree:
                
                dists.append(cp.linalg.norm(cp.subtract(self.points, self.points[node]), axis = 1))
                dists[-1][visited] = cp.inf
            
            min_dist = cp.argmin(dists)
            
            y = int(min_dist/len(self.points))
            x = min_dist % len(self.points)
            tree.append(x)
            visited[x] = True
            connections.append([self.points[x], self.points[tree[y]]])
            connection_labels.append([self.batch_y_t [x].astype("int32"), self.batch_y_t [tree[y]].astype("int32")])
        
        self.connections = connections
        self.connection_labels = connection_labels
        
    def draw_space(self):
        colors = cp.array(["blue", "yellow", "green", "red", "cyan", "magenta"])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i, conn in enumerate(self.connections):
            conn = cp.array(conn)
            ax.plot(conn.T[0], conn.T[1], conn.T[2], "black")
            
            ax.scatter(conn.T[0], conn.T[1], conn.T[2], c = colors[self.connection_labels[i]], s = 100)
            
        plt.show()
        
    def loss(self, y_p):
        
        # flat vector
        y_p = y_p.flatten()
        
        # get the n nearest neighbours
        
        # The 0 will be the distance to self
        dists = cp.linalg.norm((self.points - y_p), axis = 1)
        sorted_indices = cp.argsort(dists)
        n = self.num_neighbours + 1
        n_dists = dists[sorted_indices[1:n]]
        
        neighbour_labels = self.batch_y_t [sorted_indices].astype("int")[1:n]
        
        error_dists = n_dists[neighbour_labels != self.current_label]
        
        if len(error_dists) < 1:
            error = cp.zeros_like(y_p)
        
        else:
            class_points = self.points[sorted_indices[1 : n]]
            class_points = class_points[neighbour_labels == self.current_label]
            error = cp.sum(cp.subtract(class_points, y_p), axis = 0)
            
        error = error.reshape(len(error), )
        return cp.nan_to_num(error)
        
    def compute(self, y_p, y_t):
        self.current_label = y_t
        return self.loss(y_p)
    
    def derivative(self, x):
        x = x.ravel()
        return get_derivative(self.loss, x)
    
    def evaluate(self, y_p):
        dists = cp.linalg.norm((self.points - y_p), axis = 1)
        
        sorted_indices = cp.argsort(dists)
        n = self.num_neighbours + 1
        
        neighbours = self.batch_y_t [sorted_indices].astype("int")[1:n]
        counts = cp.bincount(self.batch_y_t)
        
        rate = []
        for i in range(len(counts)):
            
            rate.append(len(neighbours[neighbours == i]))
        
        rate = cp.array(rate)
        rate = rate/ cp.sum(rate)
        
        return rate
