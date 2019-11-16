#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import local_binary_pattern
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset



# %%
'''
Define the number of input nodes, the number of nodes in the hidden layer and the number of nodes in the output layer 
'''
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    # We are defining a (2,4,1) architecture.
    n_x = X.shape[0] # size of input layer
    n_h = 25          # size of input layer  
    n_y = Y.shape[0] # size of output layer



    return (n_x, n_h, n_y)

#%%
'''
Parameter Initialization
We will have 2 weight matrices and 2 bias vectors. Their dimension depends on the inputs and the architecture of the network. 
W1 = (HiddenLayerSize, InputSize)   = (3,4)
W2 = (OutputSize, HiddenLayerSize)  = (1,4)
b1 = (HiddenLayerSize, 1)           = (4,1) 
b2 = (OutputSize, 1)                = (1,1)
'''

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    #Initialized weights from random numbers and bias from zero 
    W1 = np.random.randn(n_h,n_x)
    b1 = np.zeros([n_h,1])
    W2 = np.random.randn(n_y,n_h)
    b2 = np.zeros([n_y,1])   

    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#%%
'''
Network Training
Step 1: Calculate the output based given network parameters
Forward_propagation
'''
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    m -- size of training set.
    
    Returns:
    A2 --    The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2", We need these values to calculate the partial derivation of the 
    loss function w.r.t to the parameters. Mainly, we will use them in the chain rule. 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    

    # Implement Forward Propagation to calculate A2 (probabilities)
    # Tangent hyberbolic function is used as the non-linear activation function for the hidden layer. And a sigmoid activation
    # is used for the output layer. 
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    

    assert(A2.shape == (10, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


#%%
'''
Network Training
Step 2: Compute the loss/cost
Based on the forward_propagation, calculate the loss. Cross entropy cost is used for this example. 
'''

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost 
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost over the entire training set
    """
    

    m = Y.shape[1] # number of training samples

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = - (1/m) * np.sum(logprobs)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    return cost


#%%
'''
Network Training
Step 3: backward_propagation
Based on the computed loss/cost, calculate the gradient of the loss function w.r.t to the parameters. 
'''


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1] #Size of training set
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
    
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    #These update equations can be derived using chain rule. 
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims= True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))   
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1 ,keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

#%%
'''
Network Training
Step 3:
Based on the computed gradients, update the paramenters using gradient decent 
'''

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule 
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#%%
#Main function for building the Neural Network Model. 
# The size of input and output will be taken from the input and output data. However, the number of nodes in the hidden layer
#needs to be defined. 

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0] #First returned value
    n_y = layer_sizes(X, Y)[2] #Third returned value
    

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)  
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters) 
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y) 
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        
        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters



#%%
# Testing the Neural network through Predict function.

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)

    topPredictions = np.argmax(A2, axis=0)

    return topPredictions

# %%



#%%
parameters = initialize_parameters(2, 1, 1)

#%%
train_samples = 500

#t,z = fetch_openml('CIFAR_10', version=1, return_X_y=True)

#%%
#X_train, X_test, y_train_data, y_test_data = train_test_split(
#    t, z, train_size=train_samples, test_size=10000)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin-1')
    return dict

cifar_data = []
cifar_labels = []

for i in range (1, 6):
    data_dic = unpickle("cifar/data_batch_{}".format(i))
    cifar_data.append(data_dic['data'])
    cifar_labels.append(data_dic['labels'])

cifar_test_dic = unpickle("cifar/test_batch")
cifar_data.append(cifar_test_dic['data'])
cifar_labels.append(cifar_test_dic['labels'])

cifar_data = np.array(cifar_data)
cifar_labels = np.array(cifar_labels)

X_data = cifar_data.reshape(cifar_data.shape[2], cifar_data.shape[0] * cifar_data.shape[1])
cifar_labels = cifar_labels.reshape(cifar_labels.shape[0] * cifar_labels.shape[1])

X_train = X_data[:, :5000]
X_test = X_data [:, 3000:]
y_train_data = cifar_labels[:5000]
y_test_data = cifar_labels[3000:]


#%%
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#%%

u = np.array([2, X_train])

y_train = np.zeros((10, len(y_train_data)))
y_test = np.zeros((10, len(y_test_data)))
print(y_train.shape, y_test.shape)

for i in range(len(y_train_data)):
    y_train[y_train_data[i], i] = 1
for i in range(len(y_test_data)):
    y_test[y_test_data[i], i] = 1

print(X_train.shape)
print(y_train.shape)

#%%
parameters = nn_model(X_train, y_train, n_h = 100, num_iterations = 5000, print_cost=True)

#%%
print(parameters)

#%%
predictions = predict(parameters, X_test)

arr =  []

#Finds diffrence on y_test_data and predictions
for i in range(len(y_test_data)):
    if predictions[i] == y_test_data[i]:
        arr.append(1)

#Calculate and print precent of correct predictions
print("accuracy: ", len(arr) / len(predictions) * 100, "%")

#%%
X_data = cifar_data.reshape(cifar_data.shape[0] * cifar_data.shape[1], 32, 32, 3)

#%%
imagesHoG = []
for i in range(0, len(X_data)):
    # Made size of gradients 8x8 as 4x4 seemed too small, yet 16x16 would have been too few axis
    fd, hog_image = hog(X_data[i], orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    imagesHoG.append(hog_image_rescaled)
imagesHoG = np.array(imagesHoG)

#%%
X_data = imagesHoG.reshape(imagesHoG.shape[1] * imagesHoG.shape[2], imagesHoG.shape[0])

print(imagesHoG.shape)
print(X_data.shape)
#%%
X_train_HoG = X_data[:, :5000]
X_test_HoG = X_data [:, 3000:]

parameters = nn_model(X_train_HoG, y_train, n_h = 100, num_iterations = 5000, print_cost=True)


predictions = predict(parameters, X_test_HoG)

arr =  []

#Finds diffrence on y_test_data and predictions
for i in range(len(y_test_data)):
    if predictions[i] == y_test_data[i]:
        arr.append(1)

#Calculate and print precent of correct predictions
print("accuracy: ", len(arr) / len(predictions) * 100, "%")

# %%
radius = 3
n_points = 8 * radius

X_data = cifar_data.reshape(cifar_data.shape[2], cifar_data.shape[0] * cifar_data.shape[1])

lbp = local_binary_pattern(X_data, n_points, radius)

lbp *= 1/lbp.max()

X_train_LBP = lbp[:, :5000]
X_test_LBP = lbp [:, 3000:]
#%%
parameters = nn_model(X_train_LBP, y_train, n_h = 100, num_iterations = 5000, print_cost=True)


predictions = predict(parameters, X_test_LBP)

arr =  []

#Finds diffrence on y_test_data and predictions
for i in range(len(y_test_data)):
    if predictions[i] == y_test_data[i]:
        arr.append(1)

#Calculate and print precent of correct predictions
print("accuracy: ", len(arr) / len(predictions) * 100, "%")

# %%
