## AI Labs - GPU Neuroevolution
This folder contains preliminary work done to implement GPU-based deep neuroevolution.
For problems like Atari where the policy evaluation takes a considerable amount of time it is advantageous to make use of GPUs to evaluate the Neural Networks. This code shows how it is possible to run Atari simulations in parallel using the GPU in a way where we can evaluate neural networks in batches and have both CPU and GPU operating at the same time.

This folder has code in prototype stage and still requires a lot of changes to optimize performance, maintanability, and testing. We welcome pull requests to this repo and have plans to improve it in the future. Although it can run on CPU-only, it is slower than our original implementation due to overhead. Once this implementation has matured we plan on distributing it as a package for easy installation. We included an implementation of the HardMaze, but the GA-NS implementation will be added later on.

## Installation

clone repo

```
git clone https://github.com/uber-common/deep-neuroevolution.git
```

create python3 virtual env

```
python3 -m venv env
. env/bin/activate
```

install tensorflow or tensorflow-gpu > 1.2.
```
pip install tensorflow-gpu
```
Follow instructions under ./gym_tensorflow/README on how to compile the optimized interfaces.

To train GA on Atari just run:
```
python ga.py ga_atari_config.json
```
Random search (It's a special case of GA where 0 individuals become parents):
```
python ga.py rs_atari_config.json
```

Evolution Strategies:
```
python es.py es_atari_config.json
```

Visualizing policies is possible if you install gym with `pip install gym` and run:
```
python -m neuroevolution.display
```
We currently have one example policy but more will be added in the future.

## Breakdown

* gym_tensorflow - Folder containing TensorFlow custom ops for Reinforcement Learning (Atari, Hard Maze).
  * moving away from python-based environments has significant speed ups on a multithreaded environment.
* neuroevolution - folder containing source code to evaluate many policies simultaneously.
  * concurrent_worker.py - Improved implementation where each thread can evaluate a dynamic sized batch of policies at a time. Needs custom Tensorflow ops.
