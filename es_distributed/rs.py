from .ga import *


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

    while True:
        step_tstart = time.time()
        theta = policy.get_trainable_flat()
        assert theta.dtype == np.float32

        if policy.needs_ob_stat:
            ob_stat = RunningStat(
                env.observation_space.shape,
                eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
            )
        if policy.needs_ref_batch:
            ref_batch = get_ref_batch(env, batch_size=128)
            policy.set_ref_batch(ref_batch)

        curr_task_id = master.declare_task(Task(
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            ref_batch=ref_batch if policy.needs_ref_batch else None,
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
                assert (result.noise_inds_n.ndim == 1 and
                        result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 1))
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
        noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        idx = np.argmax(returns_n2)
        if returns_n2[idx] > best_score:
            policy.set_trainable_flat(noise.get(noise_inds_n[idx], policy.num_params))
            policy.reinitialize()
            best_score = returns_n2[idx]
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
        assert isinstance(task_id, int) and isinstance(task_data, Task)
        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if policy.needs_ref_batch:
            policy.set_ref_batch(task_data.ref_batch)

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
                ob_count=None
            ))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths = [], [], [], []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                noise_idx = noise.sample_index(rs, policy.num_params)
                v = noise.get(noise_idx, policy.num_params)

                policy.set_trainable_flat(v)
                policy.reinitialize()
                rews_pos, len_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                noise_inds.append(noise_idx)
                returns.append([rews_pos.sum()])
                signreturns.append([np.sign(rews_pos).sum()])
                lengths.append([len_pos])

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
                ob_count=task_ob_stat.count
            ))
