import os
import tensorflow as tf
import numpy as np

from gym_tensorflow.tf_env import TensorFlowEnv

class StackFramesWrapper(TensorFlowEnv):
    def __init__(self, env, num_stacked_frames=4):
        self.env = env
        self.num_stacked_frames = num_stacked_frames
        self.obs_variable = tf.Variable(tf.zeros(shape=self.observation_space, dtype=tf.float32), trainable=False)

    @property
    def batch_size(self):
        return self.env.batch_size

    @property
    def env_default_timestep_cutoff(self):
        return self.env.env_default_timestep_cutoff

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space[:-1] + (self.env.observation_space[-1] * self.num_stacked_frames, )

    @property
    def discrete_action(self):
        return self.env.discrete_action

    def stack_observation(self, indices, reset=False):
        obs = self.env.observation(indices)

        if reset:
            obs_batch = tf.zeros((tf.shape(indices)[0],) +self.env.observation_space[1:-1] + (self.env.observation_space[-1] * self.num_stacked_frames-1, ), dtype=tf.float32)
            obs_batch = tf.concat([obs_batch, obs], axis=-1)
        else:
            obs_batch = tf.gather(self.obs_variable, indices)
            obs_batch = tf.slice(obs_batch, (0, 0, 0, 1), (-1, -1, -1, -1))
            obs_batch = tf.concat([obs_batch, obs], axis=-1)
        return tf.scatter_update(self.obs_variable, indices, obs_batch)

    def step(self, action, indices=None, name=None):
        if indices is None:
            indices = np.arange(self.batch_size)
        rew, done = self.env.step(action=action, indices=indices, name=name)
        with tf.control_dependencies([rew, done]):
            with tf.control_dependencies([self.stack_observation(indices)]):
                return tf.identity(rew), tf.identity(done)

    def reset(self, indices=None, max_frames=None, name=None):
        '''Resets Atari instances with a random noop start (1-30) and set the maximum number of frames for the episode (default 100,000 * frameskip)
        '''
        if indices is None:
            indices = np.arange(self.batch_size)
        reset_op = self.env.reset(indices=indices, max_frames=max_frames, name=name)
        with tf.control_dependencies([reset_op]):
            return self.stack_observation(indices, reset=True).op

    def observation(self, indices=None, name=None):
        '''Returns current observation after preprocessing (skip, grayscale, warp, stack).\nMust be called ONCE each time step is called if num_stacked_frames > 1
        '''
        if indices is None:
            indices = np.arange(self.batch_size)
        return tf.gather(self.obs_variable, indices)

    def final_state(self, indices, name=None):
        return self.env.final_state(indices, name)

    @property
    def unwrapped(self):
        return self.env

    def close(self):
        return self.env.close()
