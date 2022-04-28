# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:39:12 2020

This python file contains functions to load data stored in this git-hub
repository and store it in a dictionary.

@author: Vincent Bazinet
"""

import numpy as np
import os


def load_dict(path):
    '''
    Function to load a dictionary containing the required data to generate
    our results and figures. See main_script.py for information about the
    entries in this dictionary.

    Parameters
    ----------
    path : str
        Relative path to the data folder (from the current working dictionary).
        Example: 'data/LAU1000'.

    Returns
    -------
    data : dict
        Dictionary containing the data to generate our results and figures.

    '''

    data = {}
    data['sc'] = np.load(os.path.join(os.getcwd(),
                                      path,
                                      'sc.npy'))

    data['fc'] = np.load(os.path.join(os.getcwd(),
                                      path,
                                      'fc.npy'))

    data['coords'] = np.load(os.path.join(os.getcwd(),
                                          path,
                                          'coords.npy'))

    data['lhannot'] = os.path.join(os.getcwd(),
                                   path,
                                   'lh.annot')

    data['rhannot'] = os.path.join(os.getcwd(),
                                   path,
                                   'rh.annot')

    data['noplot'] = None

    data['order'] = 'RL'

    data['rsn'] = np.load(os.path.join(os.getcwd(),
                                       path,
                                       'rsn.npy'))

    data['rsn_names'] = np.load(os.path.join(os.getcwd(),
                                             path,
                                             'rsn_names.npy')).tolist()

    data['ve'] = np.load(os.path.join(os.getcwd(),
                                      path,
                                      've.npy'))

    data['ve_names'] = np.load(os.path.join(os.getcwd(),
                                            path,
                                            've_names.npy')).tolist()

    data['ci'] = []
    ci_path = os.path.join(os.getcwd(), path, 'sc_ci')
    for ci in os.listdir(os.path.join(ci_path)):
        data['ci'].append(np.load(os.path.join(ci_path, ci)))

    return data
