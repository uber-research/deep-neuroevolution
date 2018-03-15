from .es import *


GATask = namedtuple('GATask', ['params', 'population', 'ob_mean', 'ob_std', 'timestep_limit'])


def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    if exp['env_id'].endswith('NoFrameskip-v4'):
        from .atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()
    return config, env, sess, policy

def master_extract_parent_ga(point, noise_stdev, iteration, policy):
    import csv
    import os

    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    h5_filename = path+"snapshot_parent_{:04d}.h5".format(iteration)
    policy.save(h5_filename)

    filename = "snapshot_parent_{:04}.dat".format(int(iteration))

    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        bc_vec = point[0]
        fitness = point[1]
        length = point[2]
        noise_idx = point[3]
        policy_seed = point[4]
        row = np.hstack((bc_vec, fitness, length, noise_idx, policy_seed, noise_stdev))
        writer.writerow(row)

def master_extract_cloud_ga(curr_task_results, iteration):
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
                row = np.hstack((bc_vec, fitness, length, noise_idx, policy_seed))
                writer.writerow(row)

def run_master(master_redis_cfg, log_dir, exp):
    logger.info('run_master: {}'.format(locals()))
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env, sess, policy = setup(exp, single_threaded=False)
    master = MasterClient(master_redis_cfg)
    noise = SharedNoiseTable()
    rs = np.random.RandomState()

    if isinstance(config.episode_cutoff_mode, int):
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = config.episode_cutoff_mode, None, None
        adaptive_tslimit = False
    elif config.episode_cutoff_mode.startswith('adaptive:'):
        _, args = config.episode_cutoff_mode.split(':')
        arg0, arg1, arg2 = args.split(',')
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = int(arg0), float(arg1), float(arg2)
        adaptive_tslimit = True
        logger.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'.format(
                tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio))
    elif config.episode_cutoff_mode == 'env_default':
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = None, None, None
        adaptive_tslimit = False
    else:
        raise NotImplementedError(config.episode_cutoff_mode)

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)
    best_score = float('-inf')
    population = []
    population_size = exp['population_size']
    num_elites = exp['num_elites']
    population_score = np.array([])
    population_bc_vecs = []

    while True:
        step_tstart = time.time()
        theta = policy.get_trainable_flat()
        assert theta.dtype == np.float32

        if policy.needs_ob_stat:
            ob_stat = RunningStat(
                env.observation_space.shape,
                eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
            )

        curr_task_id = master.declare_task(GATask(
            params=theta,
            population=population,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
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
                curr_task_id = master.declare_task(GATask(
                    params=theta,
                    population=population,
                    ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
                    ob_std=ob_stat.std if policy.needs_ob_stat else None,
                    timestep_limit=tslimit
                ))
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))

        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
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
            else:
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

        # Assemble results + elite
        master_extract_cloud_ga(curr_task_results, curr_task_id)
        noise_inds_n = list(population[:num_elites])
        returns_n2 = list(population_score[:num_elites])
        bc_vectors = list(population_bc_vecs[:num_elites])
        for r in curr_task_results:
            noise_inds_n.extend(r.noise_inds_n)
            returns_n2.extend(r.returns_n2)
            bc_vectors.extend(r.bc_vectors)
        noise_inds_n = np.array(noise_inds_n)
        returns_n2 = np.array(returns_n2)
        bc_vectors = np.array(bc_vectors)
        lengths_n2 = np.array([r.lengths_n2 for r in curr_task_results])
        # Process returns
        idx = np.argpartition(returns_n2, (-population_size, -1))[-1:-population_size-1:-1]
        population = noise_inds_n[idx]
        population_score = returns_n2[idx]
        population_bc_vecs = bc_vectors[idx]
        assert len(population) == population_size
        assert np.max(returns_n2) == population_score[0]

        print('Elite: {} score: {}'.format(population[0], population_score[0]))
        policy.set_trainable_flat(noise.get(population[0][0], policy.num_params))
        policy.reinitialize()
        v = policy.get_trainable_flat()

        for seed in population[0][1:]:
            v += config.noise_stdev * noise.get(seed, policy.num_params)
        policy.set_trainable_flat(v)

        master_extract_parent_ga(population_bc_vecs[0], config.noise_stdev, curr_task_id, policy)
        # Update number of steps to take
        if adaptive_tslimit and (lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
            old_tslimit = tslimit
            tslimit = int(tslimit_incr_ratio * tslimit)
            logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

        step_tend = time.time()
        tlogger.record_tabular("EpRewMax", returns_n2.max())
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


def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = WorkerClient(master_redis_cfg, relay_redis_cfg)
    exp = worker.get_experiment()
    config, env, sess, policy = setup(exp, single_threaded=True)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, GATask)
        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        np.random.seed()
        policy_seed_pos = np.random.randint(2**20)
        policy_seed_pos = int(policy_seed_pos)

        if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length = policy.rollout(env)  # eval rollouts don't obey task_data.timestep_limit
            eval_return = eval_rews.sum()
            logger.info('Eval result: task={} return={:.3f} length={}'.format(task_id, eval_return, eval_length))
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
                bc_vectors=None
            ))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths, bc_vectors = [], [], [], [], []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                if len(task_data.population) > 0:
                    seeds = list(task_data.population[rs.randint(len(task_data.population))]) + [noise.sample_index(rs, policy.num_params)]
                else:
                    seeds = [noise.sample_index(rs, policy.num_params)]

                v = noise.get(seeds[0], policy.num_params)

                policy.set_trainable_flat(v)
                policy.reinitialize()
                v = policy.get_trainable_flat()

                for seed in seeds[1:]:
                    v += config.noise_stdev * noise.get(seed, policy.num_params)
                policy.set_trainable_flat(v)

                rews_pos, len_pos, nov_vec_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob,
                    policy_seed=policy_seed_pos)
                noise_inds.append(seeds)
                returns.append(rews_pos.sum())
                signreturns.append(np.sign(rews_pos).sum())
                lengths.append(len_pos)

            bc_vectors.append(
                (nov_vec_pos, returns[-1], len_pos, seeds, policy_seed_pos)
            )

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=noise_inds,
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
