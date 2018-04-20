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
import numpy as np

import tabular_logger as tlogger

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class WorkerSession(object):
    def __init__(self, worker):
        self._worker = worker
    def __enter__(self, *args, **kwargs):
        self._sess = tf.Session(*args, **kwargs)
        self._sess.run(tf.global_variables_initializer())
        self._worker.initialize(self._sess)

        tlogger.info(self._worker.model.description)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self._sess, self.coord, start=True)

        return self._sess

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type in [tf.errors.OutOfRangeError, StopIteration]:
            exception_type = None
        try:
            self._worker.close()
            self.coord.request_stop()
            self.coord.join(self.threads)
            if self._sess is None:
                raise RuntimeError('Session is already closed.')
            self._sess.close()
        finally:
            self._sess = None
        return exception_type is None
