import logging
import time
from collections import namedtuple


import numpy as np

from .dist import MasterClient, WorkerClient

logger = logging.getLogger(__name__)

Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode'
])
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'])
Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count', 'bc_vectors'
])


class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


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

def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


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

def get_ref_batch(env, batch_size=32):
    ref_batch = []
    ob = env.reset()
    while len(ref_batch) < batch_size:
        ob, rew, done, info = env.step(env.action_space.sample())
        ref_batch.append(ob)
        if done:
            ob = env.reset()
    return ref_batch

def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    if exp['policy']['type'] == "ESAtariPolicy":
        from .atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()
    return config, env, sess, policy

def master_extract_parent(eval_bc_vecs, eval_rets, iteration, policy, ref_batch):
    import csv
    import os
    import pickle

    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    h5_filename = path+"snapshot_parent_{:04d}.h5".format(iteration)
    policy.save(h5_filename)

    rb_filename = path+"snapshot_parent_{:04d}_rb.p".format(iteration)
    pickle.dump(ref_batch, open(rb_filename, "wb"))

    if not eval_rets:
        return
    else:
        print("eval_rets size = ", len(eval_rets))
        print(eval_rets)
        print("eval_rets mean = ", int(np.mean(eval_rets)))

    filename = "snapshot_parent_{:04}.dat".format(int(iteration))

    eval_rets_arr = np.array(eval_rets)
    target = int(np.mean(eval_rets_arr))
    idx = (np.abs(eval_rets_arr - target)).argmin()
    point = eval_bc_vecs[idx]

    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        bc_vec = point[0]
        fitness = point[1]
        length = point[2]
        seed = point[3]
        noise_stdev = point[4]
        row = np.hstack((bc_vec[-1], fitness, length, seed, noise_stdev))
        writer.writerow(row)

def master_extract_cloud(curr_task_results, iteration):
    import csv
    import os

    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(iteration))
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        for result in curr_task_results:
            for point in result.bc_vectors:
                bc_vec = point[0]
                fitness = point[1]
                length = point[2]
                noise_idx = point[3]
                policy_seed = point[4]
                sign = point[5]
                row = np.hstack((bc_vec[-1], fitness, length, noise_idx, policy_seed, sign))
                writer.writerow(row)

def run_master(master_redis_cfg, log_dir, exp):
    logger.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env, sess, policy = setup(exp, single_threaded=False)
    master = MasterClient(master_redis_cfg)
    theta = policy.get_trainable_flat()
    optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](theta, **exp['optimizer']['args'])
    noise = SharedNoiseTable()
    rs = np.random.RandomState()

    if policy.needs_ob_stat:
        ob_stat = RunningStat(
            env.observation_space.shape,
            eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
        )

    if policy.needs_ref_batch:
        ref_batch = get_ref_batch(env, batch_size=128)
        policy.set_ref_batch(ref_batch)


    if 'init_from' in exp['policy']:
        logger.info('Initializing weights from {}'.format(exp['policy']['init_from']))
        policy.initialize_from(exp['policy']['init_from'], ob_stat)

    if isinstance(config.episode_cutoff_mode, int):
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = config.episode_cutoff_mode, None, None, config.episode_cutoff_mode
        adaptive_tslimit = False

    elif config.episode_cutoff_mode.startswith('adaptive:'):
        _, args = config.episode_cutoff_mode.split(':')
        arg0, arg1, arg2, arg3 = args.split(',')
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = int(arg0), float(arg1), float(arg2), float(arg3)
        adaptive_tslimit = True
        logger.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}. The maximum timestep limit is {}'.format(
                tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio, tslimit_max))

    elif config.episode_cutoff_mode == 'env_default':
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = None, None, None, None
        adaptive_tslimit = False
    else:
        raise NotImplementedError(config.episode_cutoff_mode)

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)

    while True:
        step_tstart = time.time()
        theta = policy.get_trainable_flat()
        assert theta.dtype == np.float32

        curr_task_id = master.declare_task(Task(
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            ref_batch=ref_batch if policy.needs_ref_batch else None,
            timestep_limit=tslimit
        ))
        master.flush_results()
        new_task_received = False
        while not new_task_received:
            for _ in range(10):
                temp_task_id, temp_result = master.pop_result()
                if temp_task_id == curr_task_id:
                    new_task_received = True
                    break
            if not new_task_received:
                master.task_counter -= 1
                curr_task_id = master.declare_task(Task(
                    params=theta,
                    ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
                    ob_std=ob_stat.std if policy.needs_ob_stat else None,
                    ref_batch=ref_batch if policy.needs_ref_batch else None,
                    timestep_limit=tslimit
                ))
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))

        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids, eval_bc_vecs = [], [], [], [], []
        num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
        while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:
            # Wait for a result
            task_id, result = master.pop_result()
            assert isinstance(task_id, int) and isinstance(result, Result)
            assert (result.eval_return is None) == (result.eval_length is None)
            worker_ids.append(result.worker_id)

            if result.eval_length is not None:
                # This was an eval job
                episodes_so_far += 1
                timesteps_so_far += result.eval_length
                # Store the result only for current tasks
                if task_id == curr_task_id:
                    eval_rets.append(result.eval_return)
                    eval_lens.append(result.eval_length)
                    eval_bc_vecs.append(result.bc_vectors[0])
            else:
                assert (result.noise_inds_n.ndim == 1 and
                        result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
                assert result.returns_n2.dtype == np.float32
                # Store results only for current tasks
                if task_id == curr_task_id:
                    # Update counts
                    result_num_eps = result.lengths_n2.size
                    result_num_timesteps = result.lengths_n2.sum()
                    episodes_so_far += result_num_eps
                    timesteps_so_far += result_num_timesteps

                    curr_task_results.append(result)
                    num_episodes_popped += result_num_eps
                    num_timesteps_popped += result_num_timesteps
                    # Update ob stats
                    if policy.needs_ob_stat and result.ob_count > 0:
                        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        ob_count_this_batch += result.ob_count
                else:
                    num_results_skipped += 1

        # Compute skip fraction
        frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
        if num_results_skipped > 0:
            logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                num_results_skipped, 100. * frac_results_skipped))

        # Assemble results
        master_extract_parent(eval_bc_vecs, eval_rets, curr_task_id, policy, ref_batch)
        master_extract_cloud(curr_task_results, curr_task_id)
        noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
        signreturns_n2 = np.concatenate([r.signreturns_n2 for r in curr_task_results])

        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        if config.return_proc_mode == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif config.return_proc_mode == 'sign':
            proc_returns_n2 = signreturns_n2
        elif config.return_proc_mode == 'centered_sign_rank':
            proc_returns_n2 = compute_centered_ranks(signreturns_n2)
        else:
            raise NotImplementedError(config.return_proc_mode)

        # Compute and take step
        g, count = batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (noise.get(idx, policy.num_params) for idx in noise_inds_n),
            batch_size=500
        )
        g /= returns_n2.size
        assert g.shape == (policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio, theta = optimizer.update(-g + config.l2coeff * theta)

        #updating policy
        policy.set_trainable_flat(theta)

        # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
        if policy.needs_ob_stat:
            policy.set_ob_stat(ob_stat.mean, ob_stat.std)

        # Update number of steps to take
        if adaptive_tslimit and (lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
            old_tslimit = tslimit
            tslimit = min(int(tslimit_incr_ratio * tslimit), tslimit_max)
            logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

        step_tend = time.time()
        tlogger.record_tabular("EpRewMean", returns_n2.mean())
        tlogger.record_tabular("EpRewStd", returns_n2.std())
        tlogger.record_tabular("EpLenMean", lengths_n2.mean())

        tlogger.record_tabular("EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
        tlogger.record_tabular("EvalEpRewMedian", np.nan if not eval_rets else np.median(eval_rets))
        tlogger.record_tabular("EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
        tlogger.record_tabular("EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
        tlogger.record_tabular("EvalPopRank", np.nan if not eval_rets else (
            np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))
        tlogger.record_tabular("EvalEpCount", len(eval_rets))

        tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))
        tlogger.record_tabular("GradNorm", float(np.square(g).sum()))
        tlogger.record_tabular("UpdateRatio", float(update_ratio))

        tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
        tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
        tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
        tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

        num_unique_workers = len(set(worker_ids))
        tlogger.record_tabular("UniqueWorkers", num_unique_workers)
        tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
        tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)
        tlogger.record_tabular("ObCount", ob_count_this_batch)

        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - tstart)
        tlogger.dump_tabular()

        # if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
        if config.snapshot_freq != 0:
            import os.path as osp
            filename = 'snapshot_iter{:05d}_rew{}.h5'.format(
                curr_task_id,
                np.nan if not eval_rets else int(np.mean(eval_rets))
            )
            assert not osp.exists(filename)
            policy.save(filename)
            tlogger.log('Saved snapshot {}'.format(filename))



def rollout_and_update_ob_stat(policy, env, timestep_limit, rs, task_ob_stat, calc_obstat_prob, *, policy_seed=None):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:
        rollout_rews, rollout_len, obs, rollout_nov = policy.rollout(
            env, timestep_limit=timestep_limit, save_obs=True, random_stream=rs, policy_seed=policy_seed)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len, rollout_nov = policy.rollout(env, timestep_limit=timestep_limit, random_stream=rs, policy_seed=policy_seed)
    return rollout_rews, rollout_len, rollout_nov


def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = WorkerClient(relay_redis_cfg, master_redis_cfg)
    exp = worker.get_experiment()
    config, env, sess, policy = setup(exp, single_threaded=True)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)


    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, Task)

        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if policy.needs_ref_batch:
            policy.set_ref_batch(task_data.ref_batch)

        np.random.seed()
        policy_seed_eval, policy_seed_pos, policy_seed_neg = np.random.randint(2**20, size=3)
        policy_seed_eval = int(policy_seed_eval)
        policy_seed_pos = int(policy_seed_pos)
        policy_seed_neg = int(policy_seed_neg)

        if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length, eval_nov_vec = policy.rollout(env, timestep_limit=task_data.timestep_limit, policy_seed=policy_seed_eval)
            eval_return = eval_rews.sum()
            logger.info('Eval result: task={} return={:.3f} length={}'.format(task_id, eval_return, eval_length))
            bc_vectors = []
            bc_vectors.append(
                (eval_nov_vec, eval_return, eval_length, policy_seed_eval, config.noise_stdev)
            )

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=None,
                returns_n2=None,
                signreturns_n2=None,
                lengths_n2=None,
                eval_return=eval_return,
                eval_length=eval_length,
                ob_sum=None,
                ob_sumsq=None,
                ob_count=None,
                bc_vectors=bc_vectors
            ))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths, bc_vectors = [], [], [], [], []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                noise_idx = noise.sample_index(rs, policy.num_params)
                v = config.noise_stdev * noise.get(noise_idx, policy.num_params)

                policy.set_trainable_flat(task_data.params + v)
                rews_pos, len_pos, nov_vec_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob,
                    policy_seed=policy_seed_pos)

                policy.set_trainable_flat(task_data.params - v)
                rews_neg, len_neg, nov_vec_neg = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob,
                    policy_seed=policy_seed_neg)

                signreturns.append([np.sign(rews_pos).sum(), np.sign(rews_neg).sum()])
                noise_inds.append(noise_idx)
                returns.append([rews_pos.sum(), rews_neg.sum()])
                lengths.append([len_pos, len_neg])

            bc_vectors.append(
                (nov_vec_pos, returns[-1][0], len_pos, noise_idx, policy_seed_pos, 1)
            )

            bc_vectors.append(
                (nov_vec_neg, returns[-1][1], len_neg, noise_idx, policy_seed_neg, -1)
            )

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=np.array(noise_inds),
                returns_n2=np.array(returns, dtype=np.float32),
                signreturns_n2=np.array(signreturns, dtype=np.float32),
                lengths_n2=np.array(lengths, dtype=np.int32),
                eval_return=None,
                eval_length=None,
                ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                ob_count=task_ob_stat.count,
                bc_vectors=bc_vectors
            ))
