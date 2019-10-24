
import threading
from queue import Queue
from multiprocessing.pool import ApplyResult

import tabular_logger as tlogger

class AsyncWorker(object):
    @property
    def concurrent_tasks(self):
        raise NotImplementedError()

    def run_async(self, task_id, task, callback):
        raise NotImplementedError()


class WorkerHub(object):
    def __init__(self, workers, input_queue, done_queue):
        self.done_buffer = Queue()
        self.workers = workers
        self.available_workers = Queue()
        self.done_queue = done_queue
        self._cache = {}
        self.input_queue = input_queue

        for w in workers:
            for t in w.concurrent_tasks:
                self.available_workers.put((w, t))

        self.__initialize_handlers()

    def __initialize_handlers(self):
        self._input_handler = threading.Thread(
            target=WorkerHub._handle_input,
            args=(self,)
            )
        self._input_handler._state = 0

        self._output_handler = threading.Thread(
            target=WorkerHub._handle_output,
            args=(self,)
            )
        self._output_handler._state = 0

    def worker_callback(self, worker, subworker, result):
        worker_task = (worker, subworker)
        task_id = self._cache[worker_task]
        del self._cache[worker_task]
        self.done_buffer.put((task_id, result))
        # Return back to queue only after cleaning the cache to avoid
        # possible race conditions. Otherwise the worker can be reused and
        # come back to this method while cache become cleared from previous 
        # visit and this will result in error.
        self.available_workers.put(worker_task)

    @staticmethod
    def _handle_input(self):
        try:
            while True:
                worker_task = self.available_workers.get()
                if worker_task is None:
                    tlogger.info('WorkerHub._handle_input done')
                    break
                worker, subworker = worker_task

                task = self.input_queue.get()
                if task is None:
                    tlogger.info('WorkerHub._handle_input done')
                    break
                task_id, task = task
                self._cache[worker_task] = task_id

                worker.run_async(subworker, task, self.worker_callback)
        except:
            tlogger.exception('WorkerHub._handle_input exception thrown')
            raise

    @staticmethod
    def _handle_output(self):
        try:
            while True:
                result = self.done_buffer.get()
                if result is None:
                    tlogger.info('WorkerHub._handle_output done')
                    break
                self.done_queue.put(result)
        except:
            tlogger.exception('WorkerHub._handle_output exception thrown')
            raise

    def initialize(self):
        self._input_handler.start()
        self._output_handler.start()

    def close(self):
        self.available_workers.put(None)
        self.input_queue.put(None)
        self.done_buffer.put(None)

class AsyncTaskHub(object):
    def __init__(self, input_queue=None, results_queue=None):
        if input_queue is None:
            input_queue = Queue(64)
        self.input_queue = input_queue
        self._cache = {}
        self.results_queue = None
        if results_queue is not None:
            self.results_queue = results_queue

            self._output_handler = threading.Thread(
                target=AsyncTaskHub._handle_output,
                args=(self,)
                )
            self._output_handler.daemon = True
            self._output_handler._state = 0
            self._output_handler.start()

    @staticmethod
    def _handle_output(self):
        try:
            while True:
                result = self.results_queue.get()
                if result is None:
                    tlogger.info('AsyncTaskHub._handle_output done')
                    break
                self.put(result)
        except:
            tlogger.exception('AsyncTaskHub._handle_output exception thrown')
            raise

    def run_async(self, task, callback=None, error_callback=None):
        result = ApplyResult(self._cache, callback, error_callback)
        self.input_queue.put((result._job, task))
        return result

    def put(self, result):
        job, result=result
        self._cache[job]._set(0, (True, result))

