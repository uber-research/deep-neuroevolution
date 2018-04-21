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
import time
import threading
import tensorflow as tf
import numpy as np
from .tf_util import get_available_gpus
import tabular_logger as tlogger
from queue import Queue
from multiprocessing.pool import ApplyResult
from .distributed_helpers import AsyncWorker, WorkerHub, AsyncTaskHub

class RLEvalutionWorker(AsyncWorker):
    def __init__(self, make_env_f, model, batch_size, device='/cpu:0', ref_batch=None):
        self.batch_size = batch_size
        self.make_env_f = make_env_f
        self.sample_callback = [None] * self.batch_size

        self.queue = Queue()

        self._input_handler = threading.Thread(
            target=RLEvalutionWorker._loop,
            args=(self,)
            )
        self._input_handler._state = 0
        self._input_handler.start()

        self.make_net(model, device=device, ref_batch=ref_batch)
        self.num_frames = 0

    def make_net(self, model_constructor, device, ref_batch=None):
        self.model = model_constructor()

        with tf.variable_scope(None, default_name='model'):
            with tf.device('/cpu:0'):
                self.env = self.make_env_f(self.batch_size)

                self.placeholder_indices = tf.placeholder(tf.int32, shape=(None, ))
                self.placeholder_max_frames = tf.placeholder(tf.int32, shape=(None, ))
                self.reset_op = self.env.reset(indices=self.placeholder_indices, max_frames=self.placeholder_max_frames)

                with tf.device(device):
                    self.obs_op = self.env.observation(indices=self.placeholder_indices)
                    obs = tf.expand_dims(self.obs_op, axis=1)
                    self.action_op = self.model.make_net(obs, self.env.action_space, indices=self.placeholder_indices, batch_size=self.batch_size, ref_batch=ref_batch)
                self.model.initialize()

                if self.env.discrete_action:
                    self.action_op = tf.argmax(self.action_op[:tf.shape(self.placeholder_indices)[0]], axis=-1, output_type=tf.int32)
                with tf.device(device):
                    self.rew_op, self.done_op = self.env.step(self.action_op, indices=self.placeholder_indices)

                self.steps_counter = tf.Variable(np.zeros((), dtype=np.int64))
                self.incr_counter = tf.assign_add(self.steps_counter, tf.cast(tf.reduce_prod(tf.shape(self.placeholder_indices)), dtype=tf.int64))

    def _loop(self):
        info = [None] * self.batch_size
        running = np.zeros((self.batch_size,), dtype=np.bool)
        cumrews = np.zeros((self.batch_size, ), dtype=np.float32)
        cumlen = np.zeros((self.batch_size, ), dtype=np.int32)

        tf_indices = tf.placeholder(dtype=tf.int32, shape=(None,), name='input_indices')

        while True:
            # nothing loaded, block
            if not any(running):
                idx = self.queue.get()
                if idx is None:
                    break
                running[idx] = True
            while not self.queue.empty():
                idx = self.queue.get()
                if idx is None:
                    break
                running[idx] = True

            indices = np.nonzero(running)[0]
            rews, is_done, _ = self.sess.run([self.rew_op, self.done_op, self.incr_counter], {self.placeholder_indices: indices})
            cumrews[running] += rews
            cumlen[running] += 1
            if any(is_done):
                for idx in indices[is_done]:
                    self.sample_callback[idx](self, idx, (self.model.seeds[idx], cumrews[idx], cumlen[idx]))
                cumrews[indices[is_done]] = 0.
                cumlen[indices[is_done]] = 0.
                running[indices[is_done]] = False

    def initialize(self, sess):
        self.sess = sess

    def close(self):
        self.queue.put(None)
        try:
            self.env.close()
        except:
            tlogger.exception('failed at calling env.close()')

    @property
    def concurrent_tasks(self):
        return range(self.batch_size)

    def run_async(self, task_id, task, callback):
        theta, extras, max_frames=task
        self.model.load(self.sess, task_id, theta, extras)
        if max_frames is None:
            max_frames = self.env.env_default_timestep_cutoff
        self.sess.run(self.reset_op, {self.placeholder_indices:[task_id], self.placeholder_max_frames:[max_frames]})
        self.sample_callback[task_id] = callback
        self.queue.put(task_id)


class ConcurrentWorkers(object):
    def __init__(self, make_env_f, *args, gpus=get_available_gpus() * 4, input_queue=None, done_queue=None, **kwargs):
        self.sess = None
        if not gpus:
            gpus = ['/cpu:0']
        with tf.Session() as sess:
            import gym_tensorflow
            ref_batch = gym_tensorflow.get_ref_batch(make_env_f, sess, 128)
            ref_batch=ref_batch[:, ...]
        if input_queue is None and done_queue is None:
            self.workers = [RLEvalutionWorker(make_env_f, *args, ref_batch=ref_batch, **dict(kwargs, device=gpus[i])) for i in range(len(gpus))]
            self.model = self.workers[0].model
            self.steps_counter = sum([w.steps_counter for w in self.workers])
            self.async_hub = AsyncTaskHub()
            self.hub = WorkerHub(self.workers, self.async_hub.input_queue, self.async_hub)
        else:
            fake_worker = RLEvalutionWorker( * args, ** dict(kwargs, device=gpus[0]))
            self.model = fake_worker.model
            self.workers = []
            self.hub = None
            self.steps_counter = tf.constant(0)
            self.async_hub = AsyncTaskHub(input_queue, done_queue)


    def eval_async(self, theta, extras, max_frames=None, callback=None, error_callback=None):
        return self.async_hub.run_async((theta, extras, max_frames), callback=callback, error_callback=error_callback)

    def eval(self, theta, extras, max_frames=None):
        return self.eval_async(theta, extras, max_frames).get()

    def monitor_eval(self, it, max_frames):
        logging_interval = 5
        last_timesteps = self.sess.run(self.steps_counter)
        tstart_all = time.time()
        tstart = time.time()

        tasks = []
        for t in it:
            tasks.append(self.eval_async(*t, max_frames=max_frames))
            if time.time() - tstart > logging_interval:
                cur_timesteps = self.sess.run(self.steps_counter)
                tlogger.info('Num timesteps:', cur_timesteps, 'per second:', (cur_timesteps-last_timesteps)//(time.time()-tstart), 'num episodes finished: {}/{}'.format(sum([1 if t.ready() else 0 for t in tasks]), len(tasks)))
                tstart = time.time()
                last_timesteps = cur_timesteps

        while not all([t.ready() for t in tasks]):
            if time.time() - tstart > logging_interval:
                cur_timesteps = self.sess.run(self.steps_counter)
                tlogger.info('Num timesteps:', cur_timesteps, 'per second:', (cur_timesteps-last_timesteps)//(time.time()-tstart), 'num episodes:', sum([1 if t.ready() else 0 for t in tasks]))
                tstart = time.time()
                last_timesteps = cur_timesteps
            time.sleep(0.1)
        tlogger.info('Done evaluating {} episodes in {:.2f} seconds'.format(len(tasks), time.time()-tstart_all))

        return [t.get() for t in tasks]

    def monitor_eval_repeated(self, it, max_frames, num_episodes):
        logging_interval = 30
        last_timesteps = self.sess.run(self.steps_counter)
        tstart_all = time.time()
        tstart = time.time()

        tasks = []
        for t in it:
            for _ in range(num_episodes):
                tasks.append(self.eval_async(*t, max_frames=max_frames))
                if time.time() - tstart > logging_interval:
                    cur_timesteps = self.sess.run(self.steps_counter)
                    tlogger.info('Num timesteps:', cur_timesteps, 'per second:', (cur_timesteps-last_timesteps)//(time.time()-tstart), 'num episodes finished: {}/{}'.format(sum([1 if task.ready() else 0 for task in tasks]), len(tasks)))
                    tstart = time.time()
                    last_timesteps = cur_timesteps

        while not all([t.ready() for t in tasks]):
            if time.time() - tstart > 5:
                cur_timesteps = self.sess.run(self.steps_counter)
                tlogger.info('Num timesteps:', cur_timesteps, 'per second:', (cur_timesteps-last_timesteps)//(time.time()-tstart), 'num episodes:', sum([1 if t.ready() else 0 for t in tasks]))
                tstart = time.time()
                last_timesteps = cur_timesteps
            time.sleep(0.1)
        tlogger.info('Done evaluating {} episodes in {:.2f} seconds'.format(len(tasks), time.time()-tstart_all))

        results = [t.get() for t in tasks]

        # Group episodes
        results = zip(*[iter(results)] * num_episodes)

        l = []
        for evals in results:
            seeds, rews, length = zip(*evals)
            for s in seeds[1:]:
                assert s == seeds[0]
            l.append((seeds[0], np.array(rews), np.array(length)))
        return l

    def initialize(self, sess):
        for worker in self.workers:
            worker.initialize(sess)
        self.sess = sess
        if self.hub:
            self.hub.initialize()

    def close(self):
        if self.hub:
            self.hub.close()
        for worker in self.workers:
            worker.close()
