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
import numpy as np
import os

import tensorflow as tf

gym_tensorflow_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'gym_tensorflow.so'))


class TensorFlowEnv(object):
    pass


class PythonEnv(TensorFlowEnv):
    def step(self, action, indices=None, name=None):

        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.variable_scope(name, default_name='PythonStep'):

            with tf.device('/cpu:0'):
                reward, done = tf.py_func(self._step, [action, indices], [tf.float32, tf.bool])
                reward.set_shape(indices.shape)
                done.set_shape(indices.shape)
                return reward, done

    def _reset(self, indices):
        raise NotImplementedError()

    def reset(self, indices=None, max_frames=None, name=None):
        
        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.variable_scope(name, default_name='PythonReset'):
            return tf.py_func(self._reset, [indices], tf.int64).op

    def _step(self, action, indices):
        raise NotImplementedError()

    def _obs(self, indices):
        raise NotImplementedError()

    def observation(self, indices=None, name=None):

        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.variable_scope(name, default_name='PythonObservation'):

            with tf.device('/cpu:0'):
                obs = tf.py_func(self._obs, [indices], tf.float32)
                obs.set_shape(tuple(indices.shape) + self.observation_space)
            return tf.expand_dims(obs, axis=1)

    def final_state(self, indices, name=None):
        with tf.variable_scope(name, default_name='PythonFinalState'):
            return tf.zeros([tf.shape(indices)[0], 2], dtype=tf.float32)

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass


class GymEnv(PythonEnv):
    def __init__(self, name, batch_size):
        import gym
        self.env = [gym.make(name) for _ in range(batch_size)]
        self.obs = [None] * batch_size
        self.is_discrete_action = isinstance( self.env[0].action_space , gym.spaces.Discrete ) 
        self.batch_size = batch_size

    @property
    def action_space(self):
        #return np.prod(self.env[0].action_space.shape, dtype=np.int32)
        return self.env[0].action_space.n

    @property
    def observation_space(self):
        return self.env[0].observation_space.shape

    @property
    def discrete_action(self):
        return self.is_discrete_action

    @property
    def env_default_timestep_cutoff(self):
        return 1000

    def _step(self, action, indices):
        assert self.discrete_action == True 
        results = map(lambda i: self.env[indices[i]].step(action[i]), range(len(indices)))
        obs, reward, done, _ = zip(*results)
        for i in range(len(indices)):
            self.obs[indices[i]] = obs[i].astype(np.float32)

        return np.array(reward, dtype=np.float32), np.array(done, dtype=np.bool)

    def _reset(self, indices):
        for i in indices:
            self.obs[i] = self.env[i].reset().astype(np.float32)
        return 0

    def _obs(self, indices):
        return np.array([self.obs[i] for i in indices]).astype(np.float32)
