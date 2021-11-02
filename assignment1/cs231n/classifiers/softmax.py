from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(X.shape[0]):
        score = np.dot(X[i], W)
        loss += (-score[y[i]] + np.log(np.sum(np.exp(score))))
        for j in range(10):
            if j == y[i]:
                dW[:, j] += (1.0 / np.sum(np.sum(np.exp(score)))) * np.exp(score[j]) * X[i] - X[i]
            else:
                dW[:, j] += (1.0 / np.sum(np.sum(np.exp(score)))) * np.exp(score[j]) * X[i]
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    dW /= X.shape[0]
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    score = np.dot(X, W) # n * 10 matrix
    exp_score = np.exp(score)
    exp_sum = np.sum(exp_score, axis=1).reshape(-1,1)
    log_score = np.log(exp_sum)
    correct_score = score[range(num_train), y].reshape(-1,1)
    loss = np.sum(log_score - correct_score)
    loss /= num_train
    loss += reg * np.sum(W * W)
    exp_score /= exp_sum
    exp_score[range(num_train), y] -= 1
    dW = np.dot(X.T, exp_score)
    dW /= num_train
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
