# Multiscale communication in cortico-cortical networks

This repository contains scripts and functions to reproduce the results presented in "[Multiscale communication in cortico-cortical networks](<https://www.biorxiv.org/content/10.1101/2020.10.02.323030v1>)".

## The main script

[main_scripy.py](main_script.py) contains a script that allows anyone to replicate the analysis described in the paper by simply running it. This script uses [load_data.py](load_data.py) to load the required data and stores this data in a python dictionary. It then computes the main results and creates the figures presented in the paper.

Users can load their own data and modify this main script accordingly to reproduce similar analyses using their own dataset.

## The data

The [data](data) folder contains the data that was used to perform the analysis described in the main text of the paper (Lau1000). This data is necessary to generate the figures presented in the paper. It includes:

### Required for main analysis

- 'sc' : Adjacency matrix of the structural connectome [(n, n) ndarray].
- 'fc' : Adjacency matrix of the functional connectome [(n, n) ndarray].
- 'coords' : Coordinates of the parcels of our connectomes. [(n, 3) ndarray].
- 'rsn' : ndarray vector of resting-state network affiliation of the nodes
    of our networks.
- 'rsn_names' : List of names for the 7 resting-state networks.
- 've' : ndarray vector of von Economo class affiliation of the nodes of our
    networks.
- 've_names' : List of names for the 7 von-economo classes.
- 'sc_ci' : Multiscale parcellations of the n nodes of the structural connectomes into communities. Each parcellation can be stored in an individual numpy array.

### Required for plotting results on the surface of the brain

- 'lhannot' and 'rhannot': Freesurfer annotation files for the selected
    parcellation. These parameters are used by the
    netneurotools.plotting.plot_fsaverage function to plot our results on the surface of the brain
    [see <https://netneurotools.readthedocs.io> for more information].
- 'noplot' : List of names in lhannot and rhannot to not plot.
    provided annotation files. Default: None. These parameters are used by
    the netneurotools.plotting.plot_fsaverage function to plot our results on the surface of the brain
    [see <https://netneurotools.readthedocs.io> for more information].
- 'order' :  Order of the hemispheres (either ‘LR’ or ‘RL’). These parameters
    are used by the netneurotools.plotting.plot_fsaverage function to plot our results on the surface of the brain
    [see <https://netneurotools.readthedocs.io> for more information].

## The functions

The functions that have been used to compute the main results described in this paper are stored in the [multiscale_functions.py](multiscale_functions.py) file.

## The figures

[figures.py](figures.py) contains functions to generate the figures that were presented in the paper. Currently this file contains
functions to generate figures 1, 2, 4 and 5. Namely, all of the figures showing brain-related results.

## The requirements

The experiments presented in this repository make use of a certain number of python packages that will be necessary to run the main script. These packages are:

- [numpy](<https://numpy.org/doc/stable/reference/>)
- [scipy](<https://docs.scipy.org/doc/scipy/reference/>)
- [bctpy](<https://github.com/aestrivex/bctpy>) : Brain connectivity toolbox containing functions for brain network analysis. It is used to compute graph theorical measures such as shortest path or clustering coefficient.
- [tqdm](<https://github.com/tqdm/tqdm>)
- [matplotlib](<https://matplotlib.org/>)
- [seaborn](<https://seaborn.pydata.org/index.html>)
- [netneurotools](<https://github.com/netneurolab/netneurotools>) : Netneurotools is a collection of functions written in Python (and some Matlab!) that get frequent usage in the Network Neuroscience Lab.
- [palettable](<https://github.com/jiffyclub/palettable>) : A library of color palettes for python.
