import numpy as onp
import jax.numpy as np 
from jax import grad, jit, vmap, value_and_grad
from jax import random
import pdb
import time
from tqdm import tqdm
# Generate key which is used to generate random numbers
key = random.PRNGKey(1)

x = random.uniform(key, (50, 50))


def timeexec(func):
    """ Decorator which reports the execution time of a function."""

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(func.__name__, end-start)
        return result
    return wrap

# JIT (Just-in-time compilation) - lies at the core of speeding up the code. 
# In practice, this is done by wrapping ('jit()') or 
# decorating (@jit) functions.

#@timeexec
def ReLU(x):
    """ Rectified Linear Unit activation function."""
    return np.maximum(0, x)

jit_ReLU = jit(ReLU)

out = ReLU(x).block_until_ready()

jit_ReLU(x).block_until_ready()

out = jit_ReLU(x).block_until_ready()

# grad
# grad is the autodiff backbone of JAX. Wrapping function with grad and 
# evaluating it returns the gradient.
#@timeexec
def FiniteDiffGrad(x):
    """ Compute the finite difference derivative approximation for the ReLU. """
    return np.array((ReLU(x + 1e-3) - ReLU(x - 1e-3)) / (2 * 1e-3)) 

# Vmap
# vmap gives a tool for batching. vmap allows writing computations for a 
# single sample case and then vmap wraps this computation to be 
# batch-compatible. 

# Example case: Say we have a 100 dimensional feature vector and want
# to process it by a linear layer with 512 hidden units & your ReLU activation.
# Say we want to compute the layer activations for a batch with size 32.


BATCH_DIM = 32
FEATURE_DIM = 100
HIDDEN_DIM = 512

# Generate a batch of vectors to process.
X = random.normal(key, (BATCH_DIM, FEATURE_DIM))

# dim(X) = (32, 100) 

# Generate Gaussian weights and biases.
params = [random.normal(key, (HIDDEN_DIM, FEATURE_DIM)),
        random.normal(key, (HIDDEN_DIM, FEATURE_DIM))
        ]

# dim(params) = (2, 512, 100)


def relu_layer(params, x):
    """ Simple ReLU layer for a single sample."""
    return ReLU(np.dot(params[0], x) + params[1])

def batch_version_relu_layer(params, x):
    """ Error prone batch version."""
    return ReLU(np.dot(x, params[0].T) + params[1])

def vmap_relu_layer(params, x):
    """ vmap version of ReLU layer. """
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))(params, x)

# Consider vmap(relu_layer, in_axes=(None,0), out_axes=0) - what is this doing?

# Debug broadcasting issues
#out = np.stack([relu_layer(params, X[i, :]) for i in range(X.shape[0])])
#out = batch_version_relu_layer(params, X)
#out = vmap_relu_layer(params, X)

# Some simple examples using vmap
# See:https://jiayiwu.me/blog/2021/04/05/learning-about-jax-axes-in-vmap.html
a = np.array(([1, 3],
    [23, -5]
    ))
b = np.array(([11, 7],
    [19, 13]
    ))



c = np.add(a, b)


c1 = vmap(np.add, in_axes=(0,0), out_axes=0)(a,b)
# Here (0,0) => row a + row b, then stack by row. 

c2 = vmap(np.add, in_axes=(0,0), out_axes=1)(a,b)
# (0,0), 1 => row a + row b, stack by col.

c3 = vmap(np.add, in_axes=(1,0), out_axes=1)(a,b)
# (1, 0), 1 => col a + row b, stack by col.

c4 = vmap(np.add, in_axes=(0,1), out_axes=1)(a,b)
# (0,1), 1 => row a + col b, stack by col.

# in_axes can take more than just integers (though at least one should be an integer).
# Let us look at what happens when we supply None.

c5 = vmap(np.add, in_axes=(None, 0), out_axes=0)(a,b)

########################################################### 
# Example: MNIST
# See:https://github.com/RobertTLange/code-and-blog/blob/master/04_jax_intro/jax_workspace.ipynb 

from jax.scipy.special import logsumexp
from jax.experimental import optimizers

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import time

batch_size = 1

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)


#img, idx = next(iter(train_loader))
#print(np.shape(img))

#plt.imshow(img[0, 0, :, :], cmap='gray')
#plt.show()

def init_mlp(sizes, key) -> list:
    """ Initialise the weights of all the layers of a linear layer network. """
    keys = random.split(key, len(sizes)) # Produces a key for number of specified layers

    # Init a single layer with Gaussian weights

    def init_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key) # Splits the layer-specific key into a weight and bias key
        weights_init = scale * random.normal(w_key, (n,m)) # Produce n*m gaussian sampled matrix for weights init
        bias_init = scale * random.normal(b_key, (n,)) # Produce n * 1 gaussian sampled vector for bias init
        return weights_init, bias_init

    return [init_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)] # Returns array of array consistent with out chained-together layer weights e.g. [[l1, l2, k1], [l2, l3, k2], [l3, l4, k3]]

layer_sizes = [784, 512, 512, 10]
#pdb.set_trace()

params = init_mlp(layer_sizes, key)

# Forward Pass

def forward_pass(params, in_array):
    """ Compute forward pass for a single example. """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:-1]: # Looping over random weights and biases (without keys)
        activations = relu_layer([w, b], activations) 

    # Final transform to logits
    final_w, final_b = params[-1] # Take final weight, bias couple (logits transform)
    logits = np.dot(final_w, activations) + final_b
    
    return logits - logsumexp(logits)

# Batching this for multiple examples
# The above is all designed for passing one image through our network
# We use vmap to adjust this for operating on batches
batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)

def one_hot(x, k, dtype=np.float32):
    """ Create a one-hot encoding of x of size k. """
    return np.array(x[:, None] == np.arange(k), dtype) # Returns a true/false array

def loss(params, in_arrays, targets):
    """ Compute the multi-class cross-entropy loss. """
    preds = batch_forward(params, in_arrays)
    return -np.sum(preds * targets)

def accuracy(params, data_loader):
    """ Compute accuracy for a provided dataloader. """
    acc_total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data).reshape(data.size(0), 28*28)
        targets = one_hot(np.array(target), num_classes)

        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(batch_forward(params, images), axis=1)
        acc_total += np.sum(predicted_class == target_class)
    return acc_total/len(data_loader.dataset)


@jit
def update(params,x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters (weights and biases) """
    value, grads = value_and_grad(loss)(params, x, y) # Returns a function that evaluates loss and grad(loss) at (params, x, y) and returns (loss(params, x, y), grad(loss)(params, x, y))
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

# Defining an optimiser in JAX
step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

num_epochs = 3
num_classes = 10

# Training Loop

def run_mnist_training_loop(num_epochs, opt_state, net_type="MLP"):
    """ Implements a learning loop over epochs. """
    # Initialize placeholder for loggin
    log_acc_train, log_acc_test, train_loss = [], [], []
    
    # Get the initial set of parameters 
    params = get_params(opt_state)
    
    # Get initial accuracy after random init
    train_acc = accuracy(params, train_loader)
    test_acc = accuracy(params, test_loader)
    log_acc_train.append(train_acc)
    log_acc_test.append(test_acc)
    
    # Loop over the training epochs
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if net_type == "MLP":
                # Flatten the image into 784 vectors for the MLP
                x = np.array(data).reshape(data.size(0), 28*28)
            elif net_type == "CNN":
                # No flattening of the input required for the CNN
                x = np.array(data)
            y = one_hot(np.array(target), num_classes)
            params, opt_state, loss = update(params, x, y, opt_state)
            train_loss.append(loss)

        epoch_time = time.time() - start_time
        train_acc = accuracy(params, train_loader)
        test_acc = accuracy(params, test_loader)
        log_acc_train.append(train_acc)
        log_acc_test.append(test_acc)
        print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(epoch+1, epoch_time,
                                                                    train_acc, test_acc))
    
    return train_loss, log_acc_train, log_acc_test


train_loss, train_log, test_log = run_mnist_training_loop(num_epochs,
                                                          opt_state,
                                                          net_type="MLP")

