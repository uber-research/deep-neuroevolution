import gym
import figure_base.settings as gs
from figure_base.rollout_base import RolloutBase


class RolloutAtari(RolloutBase):
    @classmethod
    def make_env(self):
        from es_distributed.atari_wrappers import wrap_deepmind
        env_id = "FrostbiteNoFrameskip-v4"
        env = gym.make(env_id)
        env = wrap_deepmind(env)
        return env

    @classmethod
    def setup_policy(cls, policy_file, thisData, noise_stdev, path):
        from es_distributed.policies import ESAtariPolicy
        import pickle

        pi = ESAtariPolicy.Load(policy_file, extra_kwargs=None)

        iteration = thisData.gen
        rb_file = path+"/snapshots/snapshot_gen_{:04}/snapshot_parent_{:04d}_rb.p".format(iteration, iteration)
        rb_saved = pickle.load( open( rb_file, "rb" ) )
        pi.set_ref_batch(rb_saved)

        if not thisData.parentOrNot:
            noiseIdx, seed, noiseSign = thisData.child_op_data[-3:].astype(int)
            print(noiseIdx, seed, noiseSign)
            theta = pi.get_trainable_flat() + noiseSign * noise_stdev * gs.noise.get(noiseIdx, pi.num_params)
            pi.set_trainable_flat(theta)

        return pi

    @classmethod
    def print_info(cls, seed, rews, t, novelty_vec):
        print('return={:.4f} len={}'.format(rews.sum(), t))

    @classmethod
    def post_process(cls, env, result):
        ram = env.unwrapped._get_ram()
        print(ram)
        return True

class RolloutMujoco(RolloutBase):
    @classmethod
    def make_env(cls):
        env = gym.make('Humanoid-v1')
        return env

    @classmethod
    def setup_policy(cls, policy_file, thisData, noise_stdev, path):
        from es_distributed.policies import MujocoPolicy
        pi = MujocoPolicy.Load(policy_file, extra_kwargs=None)
        if not thisData.parentOrNot:
            noiseIdx, noiseSign = thisData.child_op_data[1:3].astype(int)
            theta = pi.get_trainable_flat() + noiseSign * noise_stdev * gs.noise.get(noiseIdx, pi.num_params)
            pi.set_trainable_flat(theta)
        return pi

    @classmethod
    def get_x_y_death_from_humanoid_bc(cls, bc):
        idx_last_x, idx_last_y = int(len(bc) / 2 - 1), -1
        x_coord, y_coord = bc[idx_last_x], bc[idx_last_y]
        return x_coord, y_coord

    @classmethod
    def print_info(cls, seed, rews, t, novelty_vec):
        x_coord, y_coord = cls.get_x_y_death_from_humanoid_bc(novelty_vec)
        print('seed={} x = {:.6f}  y = {:.6f} reward={:.8f} len={}'.format(
              seed, x_coord, y_coord, rews.sum(), t)
             )


    @classmethod
    def post_process(cls, env, result):
        xs, ys, ts, scores, seeds = [], [], [], [], []
        for r in result:
            seed, rews, _, novelty_vec = r
            x_coord, y_coord = cls.get_x_y_death_from_humanoid_bc(novelty_vec)
            xs.append(x_coord)
            ys.append(y_coord)
            ts.append(novelty_vec)
            scores.append(rews.sum())
            seeds.append(seed)
        return xs, ys, ts, scores, seeds
