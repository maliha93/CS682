import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    loss += np.log(np.sum(np.exp(scores))) - correct_class_score
    for j in range(num_class):
        temp = np.exp(scores[j]) / np.sum(np.exp(scores))
        if j == y[i]:
            dW[:,j] += (temp-1) * X[i,:]
        else:
            dW[:,j] += temp * X[i,:]
   
  loss /= num_train
  loss += reg * np.sum(W * W)
   
  dW /= num_train
  dW += 2*reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores)
  exp = np.exp(scores)
  sum = np.sum(exp, axis=1)
  correct_class_score = scores[np.arange(num_train), y]
  loss = np.log(sum) - correct_class_score
  loss = np.mean(loss) + reg * np.sum(W * W)
  
  temp = np.divide(exp, sum.reshape(X.shape[0], 1))
  temp[np.arange(X.shape[0]), y] = -(sum - exp[np.arange(X.shape[0]), y]) / sum
  dW = X.T.dot(temp)
  dW /= num_train
  dW += 2*reg*W
  
  return loss, dW

