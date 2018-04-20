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
import tensorflow as tf
from gym_tensorflow.tf_env import TensorFlowEnv, gym_tensorflow_module


class MazeEnv(TensorFlowEnv):
    def __init__(self, batch_size, name=None):
        self.batch_size = batch_size
        self.obs_variable = None
        with tf.variable_scope(name, default_name='MazeInstance'):
            self.instances = gym_tensorflow_module.maze_make(batch_size=batch_size, filename='hard_maze.txt')

    @property
    def env_default_timestep_cutoff(self):
        return 400

    @property
    def action_space(self):
        return 2

    @property
    def discrete_action(self):
        return False

    def step(self, action, indices=None, name=None):
        with tf.variable_scope(name, default_name='MazeStep'):
            #action = tf.Print(action, [action], 'action=')
            return gym_tensorflow_module.environment_step(self.instances, indices=indices, action=action)

    def reset(self, indices=None, max_frames=None, name=None):
        '''Resets Atari instances with a random noop start (1-30) and set the maximum number of frames for the episode (default 100,000 * frameskip)
        '''
        with tf.variable_scope(name, default_name='MazeReset'):
            noops = tf.random_uniform(tf.shape(indices), minval=1, maxval=31, dtype=tf.int32)
            if max_frames is None:
                max_frames = self.env_default_timestep_cutoff
            return gym_tensorflow_module.environment_reset(self.instances, indices, noops=noops, max_frames=max_frames)

    def observation(self, indices=None, name=None):
        with tf.variable_scope(name, default_name='MazeObservation'):
            with tf.device('/cpu:0'):
                obs = gym_tensorflow_module.environment_observation(self.instances, indices, T=tf.float32)
                obs.set_shape((None,) + (11,))
                #obs = tf.Print(obs, [obs], "obs=")
                return tf.expand_dims(obs, axis=1)

    def final_state(self, indices, name=None):
        with tf.variable_scope(name, default_name='MazeFinalState'):
            return gym_tensorflow_module.maze_final_state(self.instances, indices)

    def close(self):
        pass