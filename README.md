## AI Labs Neuroevolution Algorithms

This repo contains distributed implementations of the algorithms described in:

[1] [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567)

[2] [Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/abs/1712.06560)

Our code is based off of code from OpenAI, who we thank. The original code and related paper from OpenAI can be found [here](https://github.com/openai/evolution-strategies-starter). The repo has been modified to run both ES and our algorithms, including our Deep Genetic Algorithm (DeepGA) locally and on AWS.

Note: The Humanoid experiment depends on [Mujoco](http://www.mujoco.org/). Please provide your own Mujoco license and binary

The article describing these papers can be found [here](https://eng.uber.com/deep-neuroevolution/)

## Visual Inspector for NeuroEvolution (VINE)
The folder `./visual_inspector` contains implementations of VINE, i.e., Visual Inspector for NeuroEvolution, an interactive data visualization tool for neuroevolution. Refer to `README.md` in that folder for further instructions on running and customizing your visualization. An article describing this visualization tool can be found [here](https://eng.uber.com/vine/).

## How to run locally

clone repo

```
git clone https://github.com/uber-common/deep-neuroevolution.git
```

create python3 virtual env

```
python3 -m venv env
. env/bin/activate
```

install requirements
```
pip install -r requirements.txt
```
If you plan to use the mujoco env, make sure to follow [mujoco-py](https://github.com/openai/mujoco-py)'s readme about how to install mujoco correctly

launch redis
```
. scripts/local_run_redis.sh
```

launch sample ES experiment
```
. scripts/local_run_exp.sh es configurations/frostbite_es.json  # For the Atari game Frostbite
. scripts/local_run_exp.sh es configurations/humanoid.json  # For the MuJoCo Humanoid-v1 environment
```

launch sample NS-ES experiment
```
. scripts/local_run_exp.sh ns-es configurations/frostbite_nses.json
. scripts/local_run_exp.sh ns-es configurations/humanoid_nses.json
```

launch sample NSR-ES experiment
```
. scripts/local_run_exp.sh nsr-es configurations/frostbite_nsres.json
. scripts/local_run_exp.sh nsr-es configurations/humanoid_nsres.json
```

launch sample GA experiment
```
. scripts/local_run_exp.sh ga configurations/frostbite_ga.json  # For the Atari game Frostbite
```

launch sample Random Search experiment
```
. scripts/local_run_exp.sh rs configurations/frostbite_ga.json  # For the Atari game Frostbite
```


visualize results by running a policy file
```
python -m scripts.viz 'FrostbiteNoFrameskip-v4' <YOUR_H5_FILE>
python -m scripts.viz 'Humanoid-v1' <YOUR_H5_FILE>
```

### extra folder
The extra folder holds the XML specification file for the  Humanoid
Locomotion with Deceptive Trap domain used in https://arxiv.org/abs/1712.06560. Use this XML file in gym to recreate the environment.
