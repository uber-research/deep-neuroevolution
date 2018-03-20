import click


@click.command()
@click.argument('env_id')
@click.argument('policy_file')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policy_file, record, stochastic, extra_kwargs):
    import gym
    from gym import wrappers
    import tensorflow as tf
    from es_distributed.policies import MujocoPolicy, ESAtariPolicy
    from es_distributed.atari_wrappers import ScaledFloatFrame, wrap_deepmind
    from es_distributed.es import get_ref_batch
    import numpy as np

    is_atari_policy = "NoFrameskip" in env_id

    env = gym.make(env_id)
    if is_atari_policy:
        env = wrap_deepmind(env)

    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    with tf.Session():
        if is_atari_policy:
            pi = ESAtariPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
            pi.set_ref_batch(get_ref_batch(env, batch_size=128))
        else:
            pi = MujocoPolicy.Load(policy_file, extra_kwargs=extra_kwargs)
            
        while True:
            if is_atari_policy:
                rews, t, novelty_vector = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)
            print('return={:.4f} len={}'.format(rews.sum(), t))

            if record:
                env.close()
                return


if __name__ == '__main__':
    main()
