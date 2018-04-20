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
import neuroevolution.models
import gym_tensorflow
import tabular_logger as tlogger


class TrainingState(object):
    def __init__(self, exp):
        self.num_frames = 0
        self.population = []
        self.timesteps_so_far = 0
        self.time_elapsed = 0
        self.validation_timesteps_so_far = 0
        self.elite = None
        self.it = 0
        self.mutation_power = make_schedule(exp['mutation_power'])
        self.curr_solution = None
        self.curr_solution_val = float('-inf')
        self.curr_solution_test = float('-inf')

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

    def sample(self, schedule):
        return schedule.value(iteration=self.it, timesteps_so_far=self.timesteps_so_far)

    def copy_population(self, filename):
        with open(filename, 'rb+') as file:
            state = pickle.load(file)
            self.population = state.population

            # Back-compatibility
            for offspring in self.population:
                offspring.seeds = (offspring.seeds[0], ) + tuple(s if isinstance(s, tuple) else (s, 0.005) for s in offspring.seeds[1:])


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

class OffspringCached(object):
    def __init__(self, seeds):
        self.seeds = seeds

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

        cached_parents = []
        results = []


        def make_offspring():
            if len(cached_parents) == 0:
                return worker.model.randomize(rs, noise)
            else:
                assert len(cached_parents) == exp['selection_threshold']
                parent = cached_parents[rs.randint(len(cached_parents))]
                return worker.model.mutate(parent, rs, noise, mutation_power=state.sample(state.mutation_power))

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

        if 'load_population' in exp:
            state.copy_population(exp['load_population'])

        # Cache first population if needed (on restart)
        if state.population and exp['selection_threshold'] > 0:
            tlogger.info("Caching parents")
            cached_parents.clear()
            if state.elite in state.population[:exp['selection_threshold']]:
                cached_parents.extend([(worker.model.compute_weights_from_seeds(noise, o.seeds), o.seeds) for o in state.population[:exp['selection_threshold']]])
            else:
                cached_parents.append((worker.model.compute_weights_from_seeds(noise, state.elite.seeds), state.elite.seeds))
                cached_parents.extend([(worker.model.compute_weights_from_seeds(noise, o.seeds), o.seeds) for o in state.population[:exp['selection_threshold']-1]])
            tlogger.info("Done caching parents")

        while True:
            tstart_iteration = time.time()
            if state.timesteps_so_far >= exp['timesteps']:
                tlogger.info('Training terminated after {} timesteps'.format(state.timesteps_so_far))
                break
            frames_computed_so_far = sess.run(worker.steps_counter)
            assert (len(cached_parents) == 0 and state.it == 0) or len(cached_parents) == exp['selection_threshold']

            tasks = [make_offspring() for _ in range(exp['population_size'])]
            for seeds, episode_reward, episode_length in worker.monitor_eval(tasks, max_frames=state.tslimit * 4):
                results.append(Offspring(seeds, [episode_reward], [episode_length]))
            state.num_frames += sess.run(worker.steps_counter) - frames_computed_so_far

            state.it += 1
            tlogger.record_tabular('Iteration', state.it)
            tlogger.record_tabular('MutationPower', state.sample(state.mutation_power))

            # Trim unwanted results
            results = results[:exp['population_size']]
            assert len(results) == exp['population_size']
            rewards = np.array([a.fitness for a in results])
            population_timesteps = sum([a.training_steps for a in results])

            state.population = sorted(results, key=lambda x:x.fitness, reverse=True)
            tlogger.record_tabular('PopulationEpRewMax', np.max(rewards))
            tlogger.record_tabular('PopulationEpRewMean', np.mean(rewards))
            tlogger.record_tabular('PopulationEpCount', len(rewards))
            tlogger.record_tabular('PopulationTimesteps', population_timesteps)
            tlogger.record_tabular('NumSelectedIndividuals', exp['selection_threshold'])

            tlogger.info('Evaluate population')
            validation_population = state.population[:exp['validation_threshold']]
            if state.elite is not None:
                validation_population = [state.elite] + validation_population[:-1]

            validation_tasks = [(worker.model.compute_weights_from_seeds(noise, validation_population[x].seeds, cache=cached_parents), validation_population[x].seeds)
                                           for x in range(exp['validation_threshold'])]
            _, population_validation, population_validation_len = zip(*worker.monitor_eval_repeated(validation_tasks, max_frames=state.tslimit * 4, num_episodes=exp['num_validation_episodes']))
            population_validation = [np.mean(x) for x in population_validation]
            population_validation_len = [np.sum(x) for x in population_validation_len]

            time_elapsed_this_iter = time.time() - tstart_iteration
            state.time_elapsed += time_elapsed_this_iter

            population_elite_idx = np.argmax(population_validation)
            state.elite = validation_population[population_elite_idx]
            elite_theta = worker.model.compute_weights_from_seeds(noise, state.elite.seeds, cache=cached_parents)
            _, population_elite_evals, population_elite_evals_timesteps = worker.monitor_eval_repeated([(elite_theta, state.elite.seeds)], max_frames=None, num_episodes=exp['num_test_episodes'])[0]

            # Log Results
            validation_timesteps = sum(population_validation_len)
            timesteps_this_iter = population_timesteps + validation_timesteps
            state.timesteps_so_far += timesteps_this_iter
            state.validation_timesteps_so_far += validation_timesteps

            # Log
            tlogger.record_tabular('TruncatedPopulationRewMean', np.mean([a.fitness for a in validation_population]))
            tlogger.record_tabular('TruncatedPopulationValidationRewMean', np.mean(population_validation))
            tlogger.record_tabular('TruncatedPopulationEliteValidationRewMean', np.max(population_validation))
            tlogger.record_tabular("TruncatedPopulationEliteIndex", population_elite_idx)
            tlogger.record_tabular('TruncatedPopulationEliteSeeds', state.elite.seeds)
            tlogger.record_tabular('TruncatedPopulationEliteTestRewMean', np.mean(population_elite_evals))
            tlogger.record_tabular('TruncatedPopulationEliteTestEpCount', len(population_elite_evals))
            tlogger.record_tabular('TruncatedPopulationEliteTestEpLenSum', np.sum(population_elite_evals_timesteps))

            if np.mean(population_validation) > state.curr_solution_val:
                state.curr_solution = state.elite.seeds
                state.curr_solution_val = np.mean(population_validation)
                state.curr_solution_test = np.mean(population_elite_evals)

            tlogger.record_tabular('ValidationTimestepsThisIter', validation_timesteps)
            tlogger.record_tabular('ValidationTimestepsSoFar', state.validation_timesteps_so_far)
            tlogger.record_tabular('TimestepsThisIter', timesteps_this_iter)
            tlogger.record_tabular('TimestepsPerSecondThisIter', timesteps_this_iter/(time.time()-tstart_iteration))
            tlogger.record_tabular('TimestepsComputed', state.num_frames)
            tlogger.record_tabular('TimestepsSoFar', state.timesteps_so_far)
            tlogger.record_tabular('TimeElapsedThisIter', time_elapsed_this_iter)
            tlogger.record_tabular('TimeElapsedThisIterTotal', time.time()-tstart_iteration)
            tlogger.record_tabular('TimeElapsed', state.time_elapsed)
            tlogger.record_tabular('TimeElapsedTotal', time.time()-all_tstart)

            tlogger.dump_tabular()
            tlogger.info('Current elite: {}'.format(state.elite.seeds))
            fps = state.timesteps_so_far/(time.time() - tstart)
            tlogger.info('Timesteps Per Second: {:.0f}. Elapsed: {:.2f}h ETA {:.2f}h'.format(fps, (time.time()-all_tstart)/3600, (exp['timesteps'] - state.timesteps_so_far)/fps/3600))

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

            if exp['selection_threshold'] > 0:
                tlogger.info("Caching parents")
                new_parents = []
                if state.elite in state.population[:exp['selection_threshold']]:
                    new_parents.extend([(worker.model.compute_weights_from_seeds(noise, o.seeds, cache=cached_parents), o.seeds) for o in state.population[:exp['selection_threshold']]])
                else:
                    new_parents.append((worker.model.compute_weights_from_seeds(noise, state.elite.seeds, cache=cached_parents), state.elite.seeds))
                    new_parents.extend([(worker.model.compute_weights_from_seeds(noise, o.seeds, cache=cached_parents), o.seeds) for o in state.population[:exp['selection_threshold']-1]])

                cached_parents.clear()
                cached_parents.extend(new_parents)
                tlogger.info("Done caching parents")
    return float(state.curr_solution_test), {'val': float(state.curr_solution_val)}

if __name__ == "__main__":
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
