from..import tf_env

from .tf_atari import *

if not hasattr(tf_env.gym_tensorflow_module, 'atari_make'):
    class AtariEnv(TensorFlowEnv):
        def __init__(self, * args, ** kwargs):
            raise NotImplementedError("gym_tensorflow was not compiled with ALE support.")
