import tensorflow as tf
from ..tf_env import gym_tensorflow_module

try:
    indexed_matmul = gym_tensorflow_module.indexed_batch_mat_mul
except:
    import time
    print('Index MatMul implementation not available. This significantly affects performance')
    time.sleep(5)
    def indexed_matmul(a, b, idx):
        return tf.matmul(a, tf.gather(b, idx))