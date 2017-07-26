import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from mixture_density import MixtureDensityLoss, MixtureDensity, sample_network_output


def test_osaro():
    NSAMPLE = 5000
    NGAUSSIAN = 20
    NOUTPUTS = 1
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_data = np.float32(np.random.normal(size=(NSAMPLE, 1)))
    y_data = np.float32(np.sin(0.75 * x_data) * 7.0 + x_data * 0.5 + r_data * 1.0)
    # invert training data
    temp_data = x_data
    x_data = y_data
    y_data = temp_data

    model = Sequential()
    model.add(Dense(128, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(MixtureDensity(NOUTPUTS, NGAUSSIAN))

    opt = Adam(lr=0.001)
    loss_object = MixtureDensityLoss(NOUTPUTS, NGAUSSIAN)
    model.compile(loss=loss_object, optimizer=opt)
    iterations = 3000
    model.fit(x_data, y_data, batch_size=x_data.size, nb_epoch=iterations,
              verbose=1)

    x_test = np.arange(-15, 15, 0.05)
    model_output = model.predict(x_test)
    y_test = sample_network_output(model_output, x_test.size, loss_object)

    plt.figure(figsize=(10, 10))
    plt.plot(x_data, y_data, 'ro', label='data', alpha=0.1)
    plt.plot(x_test, y_test, 'bo', label='output', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


def test_multidimensional():
    NSAMPLE = 5000
    NGAUSSIAN = 20
    NOUTPUTS = 3

    first = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    second = np.float32(np.random.normal(size=(NSAMPLE, 1)))
    third = np.float32(np.random.uniform(-5, 5, (1, NSAMPLE))).T

    x_first = np.float32(np.sin(0.75 * first) * 7.0 + second * 0.5 + third * 1.0)
    x_second = np.float32(np.cos(2 * first) * 3.5 + second * 1.0 + third * 2.0)
    x_third = np.float32(np.tan(1.5 * first) * 1.5 + second * 2.0 + third * 5.0)
    x_data = np.dstack((x_first, x_second, x_third))[:, 0, :]

    model = Sequential()
    model.add(Dense(256, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(36))
    model.add(MixtureDensity(NOUTPUTS, NGAUSSIAN))

    opt = Adam(lr=0.001)
    loss_object = MixtureDensityLoss(NOUTPUTS, NGAUSSIAN)
    model.compile(loss=loss_object, optimizer=opt)
    iterations = 3000
    model.fit(third, x_data, batch_size=x_data.size, nb_epoch=iterations, verbose=1)

    # a bit hard to visualize here, just checking that it's stable and doesn't give NaNs

# test_osaro()
test_multidimensional()