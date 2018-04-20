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

import numpy as np


class Optimizer(object):
    def __init__(self, theta):
        self.theta = theta
        self.dim = len(self.theta)
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.theta
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        new_theta = self.theta + step
        self.theta = new_theta
        return ratio, new_theta

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, theta, stepsize, momentum=0.9):
        Optimizer.__init__(self, theta)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        # NOTE: different from Open AI to match more common momentum implementations (e.g. Tensorflow)
        # original from OpenAI: self.v = self.momentum * self.v + (1 - self.momentum) * globalg
        self.v = self.momentum * self.v + globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

