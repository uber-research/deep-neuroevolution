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
from .base import BaseModel


class SmallDQN(BaseModel):
    def _make_net(self, x, num_actions):
        x = self.nonlin(self.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4))
        x = self.nonlin(self.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2))
        x = self.flattenallbut0(x)
        x = self.nonlin(self.dense(x, 256, 'fc'))

        return self.dense(x, num_actions, 'out', std=0.1)


class LargeDQN(BaseModel):
    def _make_net(self, x, num_actions):
        x = self.nonlin(self.conv(x, name='conv1', num_outputs=32, kernel_size=8, stride=4, std=1.0))
        x = self.nonlin(self.conv(x, name='conv2', num_outputs=64, kernel_size=4, stride=2, std=1.0))
        x = self.nonlin(self.conv(x, name='conv3', num_outputs=64, kernel_size=3, stride=1, std=1.0))
        x = self.flattenallbut0(x)
        x = self.nonlin(self.dense(x, 512, 'fc'))

        return self.dense(x, num_actions, 'out', std=0.1)
