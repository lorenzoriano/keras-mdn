import numpy as np
import math
from keras import backend as K
from keras.engine.topology import Layer


def __softmax_keras(x):
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s


def __softmax_numpy(x):
    e = np.exp(x - np.max(x, axis=1).reshape((x.shape[0], 1)))
    s = np.sum(e, axis=1).reshape((x.shape[0], 1))
    return e/s


def softmax(x):
    # there has to be a better way, but I just can't find it :(
    if isinstance(x, np.ndarray):
        return __softmax_numpy(x)
    else:
        return __softmax_keras(x)


class MixtureDensity(Layer):
    def __init__(self, dimension, num_components, **kwargs):
        self.dimensions = dimension
        self.num_components = num_components
        self.input_dim = None
        self.output_dim = None
        self.weights_matrix = None
        self.bias = None
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.output_dim = self.num_components * (2 + self.dimensions)
        self.weights_matrix = K.variable(np.random.normal(scale=0.5, size=(self.input_dim, self.output_dim)))
        self.bias = K.variable(np.random.normal(scale=0.5, size=self.output_dim))

        self.trainable_weights = [self.weights_matrix, self.bias]

    def call(self, x, mask=None):
        output = K.dot(x, self.weights_matrix) + self.bias
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class MixtureDensityLoss(object):

    def __init__(self, dimensions, num_components, epsilon=1e-12):
        self.dimensions = dimensions
        self.num_components = num_components
        self.epsilon = epsilon
        # otherwise keras can't use it as a callable loss function
        self.__name__ = 'MixtureDensityLoss'

    def gaussian_loss(self, x, mu, sigma):
        # add an epsilon to sigma to avoid divide by zeros
        sigma = sigma + self.epsilon

        # right
        result = x - mu
        result = K.permute_dimensions(result, [2, 1, 0])
        result = result / sigma
        result = -K.square(result) / 2
        # left
        result = K.exp(result) * (1 / sigma) / math.sqrt(2 * math.pi)
        result = K.prod(result, axis=[0])

        return result

    def parse_output(self, output):
        """
        Given this layer output, extract the components weights, the means and the
        sigmas. The weighs are processed using softmax to make sure they add up to 1
        The sigmas are exponentiated to make sure they are positive
        :param output:
        :return:
        """
        components_weights = softmax(output[:, :self.num_components])
        sigmas = K.exp(output[:, self.num_components: 2*self.num_components])
        means = output[:, 2*self.num_components:]
        means = K.reshape(means, [-1, self.num_components, self.dimensions])
        means = K.permute_dimensions(means, [1, 0, 2])

        return components_weights, sigmas, means

    def __call__(self, y, output):
        components_weights, sigmas, means = self.parse_output(output)

        normal_loss = self.gaussian_loss(y, means, sigmas)
        weighted_loss = normal_loss * components_weights
        summed_loss = K.sum(weighted_loss, axis=1, keepdims=True)
        log_loss = -K.log(summed_loss + self.epsilon)
        return K.mean(log_loss)


def sample_network_output(output, num_samples, loss_object):
    assert isinstance(loss_object, MixtureDensityLoss)
    components, sigmas, means = loss_object.parse_output(output)

    # ugly but necessary
    if not isinstance(components, np.ndarray):
        components = K.eval(components)
    if not isinstance(sigmas, np.ndarray):
        sigmas = K.eval(sigmas)
    if not isinstance(means, np.ndarray):
        means = K.eval(means)

    result = np.empty((num_samples, loss_object.dimensions))

    for i in range(0, num_samples):
        for j in range(0, loss_object.dimensions):
            # pick the component
            index = np.random.choice(loss_object.num_components,
                                     p=components[i].flatten())
            mean = means[index, i, j]
            std = sigmas[i, index]
            result[i, j] = mean + np.random.randn() * std

    return result
