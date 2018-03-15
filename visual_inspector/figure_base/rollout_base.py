"""Rollout Base"""
import tensorflow as tf
import figure_base.settings as gs
import numpy as np
import sys
sys.path.append("..")
from es_distributed.es import SharedNoiseTable
from gym import wrappers


class RolloutBase():
    @classmethod
    def make_env(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def setup_policy(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def print_info(cls, *args, **kwargs):
        pass

    @classmethod
    def post_process(cls, *args, **kwargs):
        return True

    @classmethod
    def setup_and_rollout_policy(cls, policy_file, thisData, *,
                                 noise_stdev=0, num_rollouts=1, fixed_seed=None,
                                 render=False, path=None, record=None, bc_choice=None):
        if gs.noise is None:
            gs.noise = SharedNoiseTable()

        env = cls.make_env()
        env.reset()
        tf.reset_default_graph()
        if path and record:
            env = wrappers.Monitor(env, path + record, force=True)


        result = []

        with tf.Session():
            pi = cls.setup_policy(policy_file, thisData, noise_stdev, path)
            for _ in range(0, num_rollouts):
                if fixed_seed:
                    seed = fixed_seed
                else:
                    seed = np.random.randint(2**31-1)

                if bc_choice:
                    rews, t, novelty_vec = pi.rollout(env, render=render,
                                                      random_stream=np.random.RandomState(), policy_seed=seed, bc_choice=bc_choice)
                else:
                    rews, t, novelty_vec = pi.rollout(env, render=render,
                                                      random_stream=np.random.RandomState(), policy_seed=seed)
                cls.print_info(seed, rews, t, novelty_vec)
                result.append((seed, rews, t, novelty_vec))
                env.close()
        return cls.post_process(env, result)

