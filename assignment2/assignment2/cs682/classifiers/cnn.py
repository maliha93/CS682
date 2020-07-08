from builtins import object
import numpy as np

from cs682.layers import *
from cs682.fast_layers import *
from cs682.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
     
        C, H, W = input_dim
        dim = H // 2 * W // 2
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * dim, hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        cache = {}
       
        conv_layer, cache['conv'] = conv_forward_fast(X, W1, b1, conv_param)
        relu_layer1, cache['relu_layer1'] = relu_forward(conv_layer)
        pool_layer, cache['pool_layer'] = max_pool_forward_fast(relu_layer1, pool_param)
        nn_1, cache['nn_1'] = affine_forward(pool_layer, W2, b2)
        relu_layer2, cache['relu_layer2'] = relu_forward(nn_1)
        nn_2, cache['nn_2'] = affine_forward(relu_layer2, W3, b3)
        scores = nn_2

        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dout = softmax_loss(scores, y)
        
        loss += 0.5 * self.reg*(np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))        
        dnn_2, grads['W3'], grads['b3'] = affine_backward(dout, cache['nn_2'])
        drelu_layer2 = relu_backward(dnn_2, cache['relu_layer2'])
        dnn_1, grads['W2'], grads['b2'] = affine_backward(drelu_layer2, cache['nn_1'])
        dpool_layer = max_pool_backward_fast(dnn_1, cache['pool_layer'])
        drelu_layer1 = relu_backward(dpool_layer, cache['relu_layer1'])
        dconv_layer, grads['W1'], grads['b1'] = conv_backward_fast(drelu_layer1, cache['conv'])
        
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        
        return loss, grads
