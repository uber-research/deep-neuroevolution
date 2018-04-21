import os
import tensorflow as tf
import numpy as np

from gym_tensorflow.tf_env import TensorFlowEnv, gym_tensorflow_module

games = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
        'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
        'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
        'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
        'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
        'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
        'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
        'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
        'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']

screen_size = {'name_this_game': (210, 160), 'hero': (210, 160), 'space_invaders': (210, 160), 'qbert': (210, 160), 'solaris': (210, 160), 'asteroids': (210, 160), 'pooyan': (250, 160), 'gopher': (250, 160), 'video_pinball': (250, 160), 'alien': (210, 160), 'krull': (210, 160), 'beam_rider': (210, 160), 'battle_zone': (210, 160), 'double_dunk': (250, 160), 'skiing': (250, 160), 'frostbite': (210, 160), 'pong': (210, 160), 'freeway': (210, 160), 'jamesbond': (210, 160), 'tutankham': (250, 160), 'asterix': (210, 160), 'boxing': (210, 160), 'seaquest': (210, 160), 'riverraid': (210, 160), 'elevator_action': (210, 160), 'breakout': (210, 160), 'bank_heist': (250, 160), 'journey_escape': (230, 160), 'pitfall': (210, 160), 'ice_hockey': (210, 160), 'atlantis': (210, 160), 'amidar': (250, 160), 'zaxxon': (210, 160), 'tennis': (250, 160), 'kangaroo': (210, 160), 'robotank': (210, 160), 'kung_fu_master': (210, 160), 'chopper_command': (210, 160), 'assault': (250, 160), 'demon_attack': (210, 160), 'berzerk': (210, 160), 'crazy_climber': (210, 160), 'up_n_down': (210, 160), 'wizard_of_wor': (250, 160), 'yars_revenge': (210, 160), 'carnival': (250, 160), 'montezuma_revenge': (210, 160), 'road_runner': (210, 160), 'ms_pacman': (210, 160), 'gravitar': (210, 160), 'star_gunner': (210, 160), 'fishing_derby': (210, 160), 'private_eye': (210, 160), 'centipede': (250, 160), 'venture': (210, 160), 'bowling': (210, 160), 'phoenix': (210, 160), 'time_pilot': (210, 160), 'air_raid': (250, 160), 'enduro': (210, 160)}


class AtariEnv(TensorFlowEnv):
    def __init__(self, game, batch_size, warp_size=(84, 84), color_pallete=None, frameskip=4, name=None):
        assert game in games, "{} is not part of the available Atari suite".format(game)
        if color_pallete is None:
            color_pallete = grayscale_palette

        self.game = game
        self.batch_size = batch_size
        self.obs_variable = None
        self.warp_size = warp_size
        self.color_pallete = color_pallete
        self.frameskip = frameskip

        rom = os.path.join(os.path.dirname(__file__), '..', 'atari-py/atari_py/atari_roms/{}.bin'.format(game))
        with tf.variable_scope(name, default_name='AtariInstance'):
            self.instances = gym_tensorflow_module.atari_make(batch_size=batch_size, game=rom)

    @property
    def env_default_timestep_cutoff(self):
        return 100000 * self.frameskip

    @property
    def action_space(self):
        return game_actions[self.game]

    @property
    def observation_space(self):
        return (self.batch_size, ) + self.warp_size + (self.color_pallete.shape[1],)

    @property
    def discrete_action(self):
        return True

    def step(self, action, indices=None, name=None):
        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.variable_scope(name, default_name='AtariStep'):
            rew, done = gym_tensorflow_module.environment_step(self.instances, indices, action)
            return rew, done

    def reset(self, indices=None, max_frames=None, name=None):
        '''Resets Atari instances with a random noop start (1-30) and set the maximum number of frames for the episode (default 100,000 * frameskip)
        '''
        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.variable_scope(name, default_name='AtariReset'):
            noops = tf.random_uniform(tf.shape(indices), minval=1, maxval=31, dtype=tf.int32)
            if max_frames is None:
                max_frames = tf.ones_like(indices, dtype=tf.int32) * (100000 * self.frameskip)
            import collections
            if not isinstance(max_frames, collections.Sequence):
                max_frames = tf.ones_like(indices, dtype=tf.int32) * max_frames
            return gym_tensorflow_module.environment_reset(self.instances, indices, noops=noops, max_frames=max_frames)

    def render(self, indices=None):
        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.device('/cpu:0'):
            obs = gym_tensorflow_module.environment_observation(self.instances, indices, T=tf.uint8)
            return tf.gather(tf.constant(rgb_palette_uint8), tf.cast(obs, tf.int32))

    def observation(self, indices=None, name=None):
        '''Returns current observation after preprocessing (skip, grayscale, warp, stack).\nMust be called ONCE each time step is called if num_stacked_frames > 1
        '''
        if indices is None:
            indices = np.arange(self.batch_size)
        with tf.variable_scope(name, default_name='AtariObservation'):
            with tf.device('/cpu:0'):
                obs = gym_tensorflow_module.environment_observation(self.instances, indices, T=tf.uint8)

            obs = tf.gather(tf.constant(self.color_pallete), tf.cast(obs, tf.int32))
            obs = tf.reduce_max(obs, axis=1)
            obs = tf.image.resize_bilinear(obs, self.warp_size, align_corners=True)
            obs.set_shape((None,) + self.warp_size + (1,))
            return obs


    def close(self):
        pass

def get_game_obs(game):
    return screen_size[game]

def get_gym_env(game):
    import gym
    name = ''.join([g.capitalize() for g in game.split('_')])
    return gym.make('{}NoFrameskip-v4'.format(name))

ntsc_to_rgb_palette = [
    0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
    0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
    0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
    0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
    0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
    0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
    0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
    0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
    0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
    0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
    0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
    0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
    0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
    0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
    0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
    0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
    0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
    0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
    0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
    0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
    0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
    0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
    0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
    0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
    0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
    0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
    0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
    0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
    0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
    0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
    0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
    0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0]

rgb_palette_uint8 = np.empty((256, 3), np.uint8)
for i in range(256):
    zrgb = ntsc_to_rgb_palette[i]
    rgb_palette_uint8[i, 0] = (zrgb >> 16) & 0xff
    rgb_palette_uint8[i, 1] = (zrgb >> 8) & 0xff
    rgb_palette_uint8[i, 2] = (zrgb >> 0) & 0xff
rgb_palette = rgb_palette_uint8.astype(np.float32)  * (1.0 / 255.0)
grayscale_palette = np.expand_dims(np.dot(rgb_palette, np.array([0.299, 0.587, 0.114], np.float32)), axis=-1)
ntsc_palette_uint8 = np.empty((1, 1), np.uint8)

game_actions = {
    "air_raid": 6,
    "alien": 18,
    "amidar": 10,
    "assault": 7,
    "asterix": 9,
    "asteroids": 14,
    "atlantis": 4,
    "bank_heist": 18,
    "battle_zone": 18,
    "beam_rider": 9,
    "berzerk": 18,
    "bowling": 6,
    "boxing": 18,
    "breakout": 4,
    "carnival": 6,
    "centipede": 18,
    "chopper_command": 18,
    "crazy_climber": 9,
    "demon_attack": 6,
    "double_dunk": 18,
    "elevator_action": 18,
    "enduro": 9,
    "fishing_derby": 18,
    "freeway": 3,
    "frostbite": 18,
    "gopher": 8,
    "gravitar": 18,
    "hero": 18,
    "ice_hockey": 18,
    "jamesbond": 18,
    "journey_escape": 16,
    "kangaroo": 18,
    "krull": 18,
    "kung_fu_master": 14,
    "montezuma_revenge": 18,
    "ms_pacman": 9,
    "name_this_game": 6,
    "phoenix": 8,
    "pitfall": 18,
    "pong": 6,
    "pooyan": 6,
    "private_eye": 18,
    "qbert": 6,
    "riverraid": 18,
    "road_runner": 18,
    "robotank": 18,
    "seaquest": 18,
    "skiing": 3,
    "solaris": 18,
    "space_invaders": 6,
    "star_gunner": 18,
    "tennis": 18,
    "time_pilot": 10,
    "tutankham": 8,
    "up_n_down": 6,
    "venture": 18,
    "video_pinball": 9,
    "wizard_of_wor": 10,
    "yars_revenge": 18,
    "zaxxon": 18
}
