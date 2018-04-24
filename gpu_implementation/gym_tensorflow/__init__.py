import tensorflow as tf
import numpy as np
from .tf_env import GymEnv
from.import atari, maze
from .wrappers import StackFramesWrapper

def make(game, batch_size, *args, **kwargs):
    if game == 'maze':
        return maze.MazeEnv(batch_size)
    if game in atari.games:
        return StackFramesWrapper(atari.AtariEnv(game, batch_size, *args, **kwargs))
    if game.startswith('gym.'):
        return GymEnv(game[4:], batch_size, *args, **kwargs)
    raise NotImplementedError(game)


def get_ref_batch(make_env_f, sess, batch_size):
    env = make_env_f(1)
    assert env.discrete_action
    actions = tf.random_uniform((1,), minval=0, maxval=env.action_space, dtype=tf.int32)

    reset_op = env.reset()
    obs_op = env.observation()
    rew_op, done_op=env.step(actions)

    sess.run(tf.global_variables_initializer())

    sess.run(reset_op)

    ref_batch = []
    while len(ref_batch) < batch_size:
        obs, done = sess.run([obs_op, done_op])
        ref_batch.append(obs)
        if done.any():
            sess.run(reset_op)

    return np.concatenate(ref_batch)
