# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:39:12 2020

This python script helps you reproduce the results and figures presented
in the "Multiscale communication in cortico-cortical networks" paper.

IMPORTANT:
    The preliminary step to reproduce the results and generate the results
    is to load the neccessary data. This script will first load this data
    into a dictionary. The necessary entries in this dictionary are:

    'sc' : Adjacency matrix of the structural connectome.
    'fc' : Adjacency matrix of the functional connectome
    'coords' : Coordinates of the parcels of our connectomes.

    'lhannot' and 'rhannot': Freesurfer annotation files for the selected
        parcellation. These parameters are used by the
        netneurotools.plotting.plot_fsaverage function.
        [see https://netneurotools.readthedocs.io for more information]
    'noplot' : List of names in lhannot and rhannot to not plot.
        provided annotation files. Default: None. These parameters are used by
        the netneurotools.plotting.plot_fsaverage function.
        [see https://netneurotools.readthedocs.io for more information]
    'order' :  Order of the hemispheres (either ‘LR’ or ‘RL’). These parameters
        are used by the netneurotools.plotting.plot_fsaverage function.
        [see https://netneurotools.readthedocs.io for more information]

    'rsn' : ndarray vector of resting-state network affiliation of the nodes
        of our networks
    'rsn_names' : List of names for the 7 resting-state networks.
    've' : ndarray vector of von Economo class affiliation of the nodes of our
        networks.
    've_names' : List of names for the 7 von-economo classes.
    'sc_ci' : Multiscale parcellations of the n nodes of the structural
        connectomes into communities. Each parcellation can be stored in an
        individual ndarray (n,).

You can assemble this dictionary with your own data and/or parcellation.
As an example, we will reproduce the results presented in our study using the
data stored in this git-hub repository, which was generated from the Lausanne
dataset [https://zenodo.org/record/2872624#.X5s-G4hKiUk] and the cammoun
parcellation of 1000 nodes, as described in the manuscript.

If some of the data is missing, you will still be able to generate the figures
that do not require the data. For example, panels a and b of Figure 1 can
be created in the absence of the freesurfer annotation files.

@author: Vincent Bazinet
"""

import numpy as np
import os

# Load the example data
import load_data
LAU1000 = load_data.load_dict("data/LAU1000")

'''
RESULT 1:
    Compute the random walker transition probabilities.
'''

# Choose time points at which the transition probabilities will be evaluated.
LAU1000['t_points'] = np.logspace(-0.5, 1.5, 100)

# Compute transition probabilities for every time points.
from multiscale_functions import transition_probabilities
LAU1000['pr'] = transition_probabilities(LAU1000['sc'],
                                         LAU1000['t_points'],
                                         method='rwL')

'''
RESULT 2:
    Compute the multiscale centrality (c_multi) of our network's nodes.
'''

# Compute multiscale closeness centrality for every time points.
from multiscale_functions import multiscale_closeness
LAU1000['cmulti'], LAU1000['sp'] = multiscale_closeness(LAU1000['sc'],
                                                        LAU1000['pr'])

'''
RESULT 3:
    Compute the multiscale neighborhood similarities of our network's nodes.
'''

# Compute neighborhood similarity for every time points.
from multiscale_functions import neighborhood_similarity
LAU1000['nei_sim'] = neighborhood_similarity(LAU1000['sc'],
                                             LAU1000['pr'])

'''
FIGURES:
'''

# Specify individual nodes that will be highlighted in the figures.
specify = {}
specify["ID"] = [195, 266, 484, 499]
specify["colors"] = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]
specify["labels"] = ["posterior cingulate",
                     "superior parietal",
                     "transverse temporal",
                     "insula"]

# Make a "figures/" folder to store the saved figures.
if not os.path.exists("figures"):
    os.makedirs("figures")

# Plot figure 1 and save the generated elements in the 'figures/fig1' folder
from figures import figure_1

if not os.path.exists("figures/fig1"):
    os.makedirs("figures/fig1")

figure_1(LAU1000, specify, [40, 60, 75], [25, 40, 60, 75],
         save=True, show=False, save_path="figures/fig1")

# Plot figure 2 and save the generated elements in the 'figures/fig2' folder
from figures import figure_2

if not os.path.exists("figures/fig2"):
    os.makedirs("figures/fig2")

figure_2(LAU1000, save=True, show=False, save_path='figures/fig2')

# Plot figure 4 and save the generated elements in the 'figures/fig4' folder
from figures import figure_4

if not os.path.exists("figures/fig4"):
    os.makedirs("figures/fig4")

figure_4(LAU1000, specify, [25, 49, 60, 75], save=True, show=False,
         save_path='figures/fig4')

# Plot figure 5 and save the generated elements in the 'figures/fig5' folder
from figures import figure_5

if not os.path.exists("figures/fig5"):
    os.makedirs("figures/fig5")

figure_5(LAU1000, save=True, show=False, save_path='figures/fig5')
