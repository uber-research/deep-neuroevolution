__copyright__ = """
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from .dqn import Model
import tensorflow as tf


class ModelBN(Model):
    def __init__(self):
        super(ModelBN, self).__init__()
        self.nonlin = lambda x: tf.nn.relu(self.batchnorm(x))
    def batchnorm(self, x):
        with tf.variable_scope(None, default_name='BatchNorm'):
            ret = tf.layers.batch_normalization(x, center=False, scale=False, training=True)

            if len(x.get_shape()) == 4:
                b = self.create_bias_variable('b', (1, 1, ret.get_shape()[-1].value))
            else:
                b = self.create_bias_variable('b', (1, ret.get_shape()[-1].value))
            if self.indices is not None:
                b = tf.gather(b, self.indices)

            ret = ret + b
            return ret

    def _make_net(self, x, num_actions):
        x = self.nonlin(self.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4, bias=False))
        x = self.nonlin(self.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2, bias=False))
        x = self.flattenallbut0(x)
        x = self.nonlin(self.dense(x, 256, 'fc', bias=False))

        ret = self.dense(x, num_actions, 'out', std=0.1)
        return ret


class ModelVirtualBN(Model):
    def __init__(self):
        super(ModelVirtualBN, self).__init__()
        self.is_ref_batch = False
        self.nonlin = lambda x: tf.nn.relu(self.batchnorm(x))
        self.device = None

    @property
    def requires_ref_batch(self):
        return True

    # This got a little out of hand, but it maintains a set of mean/var variables that are updated on load and used during inference.
    def batchnorm(self, x):
        with tf.variable_scope('BatchNorm'):
            if len(x.get_shape()) == 5:
                vbn_mean = tf.get_variable('mean', shape=(self.batch_size, x.get_shape()[-1].value), trainable=False)
                vbn_var = tf.get_variable('var', shape=(self.batch_size, x.get_shape()[-1].value), trainable=False)
            else:
                vbn_mean = tf.get_variable('mean', shape=(self.batch_size, x.get_shape()[-1].value), trainable=False)
                vbn_var = tf.get_variable('var', shape=(self.batch_size, x.get_shape()[-1].value), trainable=False)

            if self.is_ref_batch:
                mean, var = tf.nn.moments(x, list(range(1, len(x.get_shape())-1)))
                var = 1 / tf.sqrt(var + 1e-3)
                mean, var = tf.scatter_update(vbn_mean, self.indices, mean), tf.scatter_update(vbn_var, self.indices, var)
            else:
                mean, var = vbn_mean, vbn_var
            while len(mean.get_shape()) < len(x.get_shape()):
                mean, var = tf.expand_dims(mean, 1), tf.expand_dims(var, 1)

            if self.indices is not None:
                mean, var = tf.gather(mean, self.indices), tf.gather(var, self.indices)

            ret = (x-mean) * var

            if len(x.get_shape()) == 5:
                b = self.create_bias_variable('b', (1, 1, 1, ret.get_shape()[-1].value))
            else:
                b = self.create_bias_variable('b', (1, ret.get_shape()[-1].value))
            if self.indices is not None:
                b = tf.gather(b, self.indices)
            return ret + b

    def _make_net(self, x, num_actions, ):
        with tf.variable_scope('layer1'):
            x = self.nonlin(self.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4, bias=False))
        with tf.variable_scope('layer2'):
            x = self.nonlin(self.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2, bias=False))
        x = self.flattenallbut0(x)
        with tf.variable_scope('layer3'):
            x = self.nonlin(self.dense(x, 256, 'fc', bias=False))

        with tf.variable_scope('layer4'):
            return self.dense(x, num_actions, 'out')

    def make_weights(self):
        super(ModelVirtualBN, self).make_weights()
        self.ref_batch_idx = tf.placeholder(tf.int32, ())
        tmp = self.indices
        self.indices = [self.ref_batch_idx]
        with tf.device(self.device):
            with tf.variable_scope(self.scope, reuse=True):
                ref_batch = tf.stack([self.ref_batch])
                self.is_ref_batch = True
                self.ref_batch_assign = self._make_net(ref_batch, self.num_actions)
                self.is_ref_batch = False
        self.indices = tmp

    def load(self, sess, i, *args, **kwargs):
        ret = super(ModelVirtualBN, self).load(sess, i, *args, **kwargs)
        sess.run(self.ref_batch_assign, {self.ref_batch_idx: i})
        return ret
