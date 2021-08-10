'''
Last update: 20.09.23. by KS.Kwon
'''
import tensorflow as tf
import numpy as np

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

def uniform(shape, scale=0.05, name=None):
    """ Uniform init. """
    initial = tf.random.uniform(
        shape, minval=-scale, maxval=scale, dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def binomial(shape, scale=0.5, name=None):
    """ Binomial init. """
    initial = tf.keras.backend.random_bernoulli(
        shape, p=scale, dtype=tf.float32, seed=213
    )
    return tf.Variable(initial, name=name)
    
    
def glorot(shape, name=None):
    """ Glorot & Bengio (AISTATS 2010) init. """
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(
        shape, minval=-init_range, maxval=init_range, dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def He(shape, name=None):
    """ He init. """
    init_range = np.sqrt(6.0/shape[0])
    initial = tf.random.uniform(
        shape, minval=-init_range, maxval=init_range, dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """ All zeros. """
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """ All ones. """
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)