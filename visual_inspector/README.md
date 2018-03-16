## Visual Inspector for NeuroEvolution (VINE)

This repo contains implementations of VINE, i.e., Visual Inspector for NeuroEvolution, an interactive data visualization tool for neuroevolution. An article describing this visualization tool can be found [here](https://eng.uber.com/vine/).

### Dependencies that need to be downloaded by end-user from a third party

In addition to requirements in `../requirements.txt`:

* [Matplotlib](https://matplotlib.org/) -- version 2.0.2
* [Sklearn](http://scikit-learn.org/stable/) -- version 0.19.1
* [Pandas](https://pandas.pydata.org/) -- version 0.22.0
* [Colour](https://github.com/vaab/colour) -- version 0.1.5

### Visualize the pseudo-offspring clouds

__Example 1__: visualize the sample Mujoco Humanoid 2D BC (i.e., final x-y location) data for Generations 90 to 99
```
python -m main_mujoco 90 99 sample_data/mujoco/final_xy_bc/
```
This will bring up the GUI which consists of two interrelated plots: a pseudo-offspring cloud plot, and a fitness plot, similar to Figure 2 of the [article](https://eng.uber.com/vine/), which is described in detail there.

__Example 2__: click `Movie` button on the GUI to generate a visualization of the moving cloud similar to Figure 3 of the [article](https://eng.uber.com/vine/), which can be saved as a movie clip by checking `save movie` checkbox.

__Example 3__: right click any point of the pseudo-offspring cloud to view videos of the corresponding agent’s deterministic and stochastic behaviors (only available for Generation 97 in `sample_data`). Follow the steps (all "clicks" are right click) illustrated in Figure 5 of the [article](https://eng.uber.com/vine/).


To see HELP for the complete description of all available options (e.g., multiple BCs, and hi-dimensional BCs):
```
python -m main_mujoco --help
```


### Using dimensionality reduction to process high-dimensional BC

Assume you would like to reduce 2000D BCs to 2D for Generations 0 to 99 using PCA:
```
python -m process_bc 0 99 <path_to_hd_bc> 2000 --method pca
```
The reduced BC data is stored at `<path_to_hd_bc>/reduced_pca`

To see HELP for the complete description of all available options:
```
python -m process_bc --help
```

### Create and visualize your own data

1. Choose proper behavior characterizations (BCs) (refer to the [article](https://eng.uber.com/vine/) for examples).
2. Moderately modify your GA or ES code that dump out the BCs during neuroevolution.
   Examples of BC choices and modified version of GA and ES, namely, `es_modified.py`, `ga_modified.py`, are privode in `../es_distributed` for your references.
3. If applicable, using dimensionality reduction (see above) to reduce hi-dimensional BCs to 2D.
4. Create (if necessary) and run main_<XXX>.py file to launch GUI.
   `main_mujoco.py` or `main_atari.py` can be used directly or used as a template for most of your use cases.



