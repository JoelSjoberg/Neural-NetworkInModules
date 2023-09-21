import numpy as np
from Scripts.BatchLayers import Dense, Graph_Layer, Noise, Dropout
from Scripts.activations import *
from Scripts.loss_functions import *
from Scripts.NeuralNetwork import *

from Scripts.data_generator import *


np.random.seed(0)
def make_model():
    np.random.seed(0)
    
    inp_size = 64
    out_size = 1

    loss = MSE()
    model = Model(loss = loss)
    #model.add(Dropout(rate = 0.1))
    model.add(Noise())
    model.add(Dense((inp_size, 16), activation = l_relu, bias = True))
    model.add(Dense((16, out_size), activation = relu, bias = True))

    return model

# Get data
#X, y = generate_dense(100000, classes = 5, dim = 64)
X, y = get_mnist()

batch_size = 64

nums = int(len(X) * 0.7)
train_X = X[:nums]
train_y = y[:nums]

test_X = X[nums:]
test_y = y[nums:]

print(train_X.shape)

# Train the model
model = make_model()

model.train(train_X, train_y, batch_size = batch_size, epochs = 100000, alpha = 0.0003, balanced_batch=False)

# Accuracy for the regular loss methods
true_positives = np.sum((np.round(model.predict(train_X)).astype(np.int64) == train_y))
accuracy = true_positives/len(train_y)
print("Train:", accuracy)

# Measure accuracy on the test set
true_positives = np.sum((np.round(model.predict(test_X)).astype(np.int64) == test_y))
accuracy = true_positives/len(test_y)
print("Test:", accuracy)

"""
preds = model.predict(train_X)
centroids = []
for u in np.unique(train_y):

    class_examples = cp.array(preds)[train_y.flatten() == u]

    centroid = cp.mean(class_examples, axis = 0)

    centroids.append(centroid)

centroids = np.array(centroids)

# Accuracy for the cluster loss method

# Measure accuracy on the train set

pred_ints = []

for p in preds:
    diff = p - centroids

    diff = np.linalg.norm(diff, axis = 1)

    pred_int = np.argmin(diff)

    pred_ints.append(pred_int)

pred_ints = np.expand_dims(np.array(pred_ints), -1)

true_positives = np.sum((pred_ints == train_y))
accuracy = true_positives/len(train_y)

#for i in np.unique(pred_ints.flatten()):
#    points = preds[pred_ints.flatten() == i]
#    plt.scatter(points.T[0], points.T[1])
#    plt.scatter(centroids[i][0], centroids[i][1], color = "Black")
#plt.scatter(centroids.T[0], centroids.T[1], color = "Black")
#plt.scatter(prev_centroids.T[0], prev_centroids.T[1], color = "Gray")
#plt.show()

print("Train:", np.round(accuracy * 100, 2) , "%")

# Measure accuracy on the test set
preds = model.predict(test_X)
pred_ints = []
for p in preds:
    diff = p - centroids
    diff = np.linalg.norm(diff, axis = 1)
    pred_int = np.argmax(diff)
    pred_ints.append(pred_int)

pred_ints = np.expand_dims(np.array(pred_ints), -1)

true_positives = np.sum((pred_ints == test_y))
accuracy = true_positives/len(test_y)
print("Test:", np.round(accuracy * 100, 2) , "%")

"""

