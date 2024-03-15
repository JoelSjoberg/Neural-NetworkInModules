import numpy as np
from Scripts.BatchLayers import Dense, Graph_Layer, Noise, Dropout, FeatureImportance
from Scripts.activations import *
from Scripts.loss_functions import *
from Scripts.NeuralNetwork import *

from Scripts.data_generator import *

import matplotlib.pyplot as plt


def make_model(loss = MSE):

    np.random.seed(0)
    inp_size = 4 * 4
    out_size = 1

    last_W = np.arange(10).reshape((10, 1))

    model = Model(loss = loss)

    #model.add(FeatureImportance(inp_size))
    model.add(Dropout(0.1))
    model.add(Dense((inp_size, 24), activation = l_relu, bias = True))

    model.add(Dense((24, 32), activation = l_relu, bias = True))

    model.add(Dense((32, 48), activation = tanh, bias = True))

    model.add(Dense((48, 52), activation = tanh, bias = True))

    model.add(Dense((52, 60), activation = tanh, bias = True))

    model.add(Dense((60, 64), activation = sigmoid, bias = True))
    #model.add(Dense((10, out_size), activation = l_relu, bias = False, W=last_W, trainable=False))

    return model

np.random.seed(0)
# Get data
#X, y = generate_dense(100000, classes = 5, dim = 64)
X, y = get_mnist()

# Down sample the images
X_m = []

for x in X:
    x_m = []
    for i in range(0, 64, 4):
        x_m.append(np.mean(x[i: i+2]))

    x_m = np.array(x_m)

    X_m.append(x_m)

X_m = np.array(X_m)



# Ohe
#eye = np.eye(10)
#y = np.squeeze(eye[y])

nums = int(len(X) * 0.7)
#train_X = X[:nums]
#train_y = y[:nums]
#test_X = X[nums:]
#test_y = y[nums:]

train_X = X_m[:nums]
train_y = X[:nums]
test_X = X_m[nums:]
test_y = X[nums:]

print("# training points:", train_X.shape)
# Train the model
model = make_model(loss = MSE())
print(model.size())

batch_size = 32

model.train(train_X, train_y, 
            batch_size = batch_size, 
            epochs = 10000, 
            alpha = 0.00001, 
            balanced_batch = True, 
            regularization = True, 
            lambd = 0.000001)

preds = model.predict(test_X[0:5])
plt.subplot(2, 2, 1)
plt.imshow(test_X[0].reshape(4, 4))

plt.subplot(2, 2, 2)
plt.imshow(preds[0].reshape(8, 8))

plt.subplot(2, 2, 3)
plt.imshow(test_X[1].reshape(4, 4))

plt.subplot(2, 2, 4)
plt.imshow(preds[1].reshape(8, 8))
plt.show()

#print(np.round(model.layers[0].W, 2).reshape((8, 8)))
#plt.imshow(model.layers[0].W.reshape((8, 8)))
#plt.show()

"""
# Accuracy for the regular loss methods
true_positives = np.sum((np.round(model.predict(train_X)).astype(np.int64) == train_y))
accuracy = true_positives/len(train_y)
print("Train:", accuracy)

# Measure accuracy on the test set
true_positives = np.sum((np.round(model.predict(test_X)).astype(np.int64) == test_y))
accuracy = true_positives/len(test_y)
print("Test:", accuracy)
"""

# Training accuracy
"""
pred_labels = np.argmax(model.predict(train_X), axis=1)
true_labels = np.argmax(train_y, axis=1)
true_positives = np.sum(pred_labels == true_labels)
accuracy = true_positives/len(train_y)
print("Train:", accuracy)

# For one hot encoded labels
pred_labels = np.argmax(model.predict(test_X), axis=1)
print(pred_labels[:2])
print(np.argmax(y, axis=1)[:2])
print(np.round(model.predict(test_X)[:2], 2))
true_labels = np.argmax(test_y, axis=1)
true_positives = np.sum(pred_labels == true_labels)
accuracy = true_positives/len(test_y)
print("Test:", accuracy)
"""