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

import argparse
import json
import time
import sys
import base64
import pickle
import tempfile
import os
from shutil import copyfile
import tensorflow as tf
import numpy as np
from neuroevolution.tf_util import get_available_gpus, WorkerSession
from neuroevolution.helper import SharedNoiseTable, make_schedule
from neuroevolution.concurrent_worker import ConcurrentWorkers
from neuroevolution.optimizers import SGD, Adam
import neuroevolution.models
import tabular_logger as tlogger
from threading import Lock
import gym_tensorflow


class TrainingState(object):
    def __init__(self, exp):
        self.num_frames = 0
        self.timesteps_so_far = 0
        self.time_elapsed = 0
        self.validation_timesteps_so_far = 0
        self.it = 0
        self.mutation_power = make_schedule(exp['mutation_power'])
        self.exp = exp

        self.theta = None
        self.optimizer = None

        if isinstance(exp['episode_cutoff_mode'], int):
            self.tslimit = exp['episode_cutoff_mode']
            self.incr_tslimit_threshold = None
            self.tslimit_incr_ratio = None
            self.adaptive_tslimit = False
        elif exp['episode_cutoff_mode'].startswith('adaptive:'):
            _, args = exp['episode_cutoff_mode'].split(':')
            arg0, arg1, arg2, arg3 = args.split(',')
            self.tslimit, self.incr_tslimit_threshold, self.tslimit_incr_ratio, self.tslimit_max = int(arg0), float(arg1), float(arg2), float(arg3)
            self.adaptive_tslimit = True
            tlogger.info(
                'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'.format(
                    self.tslimit, self.incr_tslimit_threshold * 100, self.tslimit_incr_ratio))
        elif exp['episode_cutoff_mode'] == 'env_default':
            self.tslimit, self.incr_tslimit_threshold, self.tslimit_incr_ratio = None, None, None
            self.adaptive_tslimit = False
        else:
            raise NotImplementedError(exp['episode_cutoff_mode'])

    def initialize(self, rs, noise, model):
        theta, _ = model.randomize(rs, noise)
        self.set_theta(theta)

    def set_theta(self, theta):
        self.theta = theta
        self.optimizer = {'sgd': SGD, 'adam': Adam}[self.exp['optimizer']['type']](self.theta, **self.exp['optimizer']['args'])

    def sample(self, schedule):
        return schedule.value(iteration=self.it, timesteps_so_far=self.timesteps_so_far)

class Offspring(object):
    def __init__(self, seeds, rewards, ep_len, validation_rewards=[], validation_ep_len=[]):
        self.seeds = seeds
        self.rewards = rewards
        self.ep_len = ep_len
        self.validation_rewards = validation_rewards
        self.validation_ep_len = validation_ep_len

    @property
    def fitness(self):
        return np.mean(self.rewards)

    @property
    def training_steps(self):
        return np.sum(self.ep_len)


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def main(**exp):
    log_dir = tlogger.log_dir()

    tlogger.info(json.dumps(exp, indent=4, sort_keys=True))
    tlogger.info('Logging to: {}'.format(log_dir))
    Model = neuroevolution.models.__dict__[exp['model']]
    all_tstart = time.time()
    def make_env(b):
        return gym_tensorflow.make(game=exp["game"], batch_size=b)
    worker = ConcurrentWorkers(make_env, Model, batch_size=64)
    with WorkerSession(worker) as sess:
        noise = SharedNoiseTable()
        rs = np.random.RandomState()
        tlogger.info('Start timing')
        tstart = time.time()

        try:
            load_file = os.path.join(log_dir, 'snapshot.pkl')
            with open(load_file, 'rb+') as file:
                state = pickle.load(file)
            tlogger.info("Loaded iteration {} from {}".format(state.it, load_file))
        except FileNotFoundError:
            tlogger.info('Failed to load snapshot')
            state = TrainingState(exp)

            if 'load_from' in exp:
                dirname = os.path.join(os.path.dirname(__file__), '..', 'neuroevolution', 'ga_legacy.py')
                load_from = exp['load_from'].format(**exp)
                os.system('python {} {} seeds.pkl'.format(dirname, load_from))
                with open('seeds.pkl', 'rb+') as file:
                    seeds = pickle.load(file)
                    state.set_theta(worker.model.compute_weights_from_seeds(noise, seeds))
                tlogger.info('Loaded initial theta from {}'.format(load_from))
            else:
                state.initialize(rs, noise, worker.model)
        def make_offspring(state):
            for i in range(exp['population_size'] // 2):
                idx = noise.sample_index(rs, worker.model.num_params)
                mutation_power = state.sample(state.mutation_power)
                pos_theta = worker.model.compute_mutation(noise, state.theta, idx, mutation_power)

                yield (pos_theta, idx)
                neg_theta = worker.model.compute_mutation(noise, state.theta, idx, -mutation_power)
                diff = (np.max(np.abs((pos_theta + neg_theta)/2 - state.theta)))
                assert diff < 1e-5, 'Diff too large: {}'.format(diff)

                yield (neg_theta, idx)

        tlogger.info('Start training')
        _, initial_performance, _ = worker.monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes'])[0]
        while True:
            tstart_iteration = time.time()
            if state.timesteps_so_far >= exp['timesteps']:
                tlogger.info('Training terminated after {} timesteps'.format(state.timesteps_so_far))
                break
            frames_computed_so_far = sess.run(worker.steps_counter)

            tlogger.info('Evaluating perturbations')
            iterator = iter(worker.monitor_eval(make_offspring(state), max_frames=state.tslimit * 4))
            results = []
            for pos_seeds, pos_reward, pos_length in iterator:
                neg_seeds, neg_reward, neg_length = next(iterator)
                assert pos_seeds == neg_seeds
                results.append(Offspring(pos_seeds, [pos_reward, neg_reward], [pos_length, neg_length]))
            state.num_frames += sess.run(worker.steps_counter) - frames_computed_so_far

            state.it += 1
            tlogger.record_tabular('Iteration', state.it)
            tlogger.record_tabular('MutationPower', state.sample(state.mutation_power))
            tlogger.record_tabular('TimestepLimitPerEpisode', state.tslimit)

            # Trim unwanted results
            results = results[:exp['population_size']//2]
            assert len(results) == exp['population_size']//2
            rewards = np.array([b for a in results for b in a.rewards])

            results_timesteps = np.array([a.training_steps for a in results])
            timesteps_this_iter = sum([a.training_steps for a in results])
            state.timesteps_so_far += timesteps_this_iter

            tlogger.record_tabular('PopulationEpRewMax', np.max(rewards))
            tlogger.record_tabular('PopulationEpRewMean', np.mean(rewards))
            tlogger.record_tabular('PopulationEpRewMedian', np.median(rewards))
            tlogger.record_tabular('PopulationEpCount', len(rewards))
            tlogger.record_tabular('PopulationTimesteps', timesteps_this_iter)


            # Update Theta
            returns_n2 = np.array([a.rewards for a in results])
            noise_inds_n = [a.seeds for a in results]

            if exp['return_proc_mode'] == 'centered_rank':
                proc_returns_n2 = compute_centered_ranks(returns_n2)
            else:
                raise NotImplementedError(exp['return_proc_mode'])
            # Compute and take step
            g, count = batched_weighted_sum(
                proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
                (noise.get(idx, worker.model.num_params) for idx in noise_inds_n),
                batch_size=500
            )
            # NOTE: gradients are scaled by \theta
            g /= returns_n2.size

            assert g.shape == (worker.model.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
            update_ratio, state.theta = state.optimizer.update(-g + exp['l2coeff'] * state.theta)

            time_elapsed_this_iter = time.time() - tstart_iteration
            state.time_elapsed += time_elapsed_this_iter
            tlogger.info('Evaluate elite')
            _, test_evals, test_timesteps = worker.monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes'])[0]
            test_timesteps = sum(test_timesteps)
            # Log Results
            tlogger.record_tabular('TestRewMean', np.mean(test_evals))
            tlogger.record_tabular('TestRewMedian', np.median(test_evals))
            tlogger.record_tabular('TestEpCount', len(test_evals))
            tlogger.record_tabular('TestEpLenSum', test_timesteps)
            tlogger.record_tabular('InitialRewMax', np.max(initial_performance))
            tlogger.record_tabular('InitialRewMean', np.mean(initial_performance))
            tlogger.record_tabular('InitialRewMedian', np.median(initial_performance))

            tlogger.record_tabular('TimestepsThisIter', timesteps_this_iter)
            tlogger.record_tabular('TimestepsPerSecondThisIter', timesteps_this_iter/(time.time()-tstart_iteration))
            tlogger.record_tabular('TimestepsComputed', state.num_frames)
            tlogger.record_tabular('TimestepsSoFar', state.timesteps_so_far)
            tlogger.record_tabular('TimeElapsedThisIter', time_elapsed_this_iter)
            tlogger.record_tabular('TimeElapsedThisIterTotal', time.time()-tstart_iteration)
            tlogger.record_tabular('TimeElapsed', state.time_elapsed)
            tlogger.record_tabular('TimeElapsedTotal', time.time()-all_tstart)

            tlogger.dump_tabular()
            fps = state.timesteps_so_far/(time.time() - tstart)
            tlogger.info('Timesteps Per Second: {:.0f}. Elapsed: {:.2f}h ETA {:.2f}h'.format(fps, (time.time() - all_tstart) / 3600, (exp['timesteps'] - state.timesteps_so_far) / fps / 3600))

            if state.adaptive_tslimit:
                if np.mean([a.training_steps >= state.tslimit for a in results]) > state.incr_tslimit_threshold:
                    state.tslimit = min(state.tslimit * state.tslimit_incr_ratio, state.tslimit_max)
                    tlogger.info('Increased threshold to {}'.format(state.tslimit))

            os.makedirs(log_dir, exist_ok=True)
            save_file = os.path.join(log_dir, 'snapshot.pkl')
            with open(save_file, 'wb+') as file:
                pickle.dump(state, file)
            #copyfile(save_file, os.path.join(log_dir, 'snapshot_gen{:04d}.pkl'.format(state.it)))
            tlogger.info("Saved iteration {} to {}".format(state.it, save_file))

            if state.timesteps_so_far >= exp['timesteps']:
                tlogger.info('Training terminated after {} timesteps'.format(state.timesteps_so_far))
                break
            results.clear()

if __name__ == "__main__":
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
