#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:33:59 2020

@author: oriol
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import matplotlib.font_manager
import matplotlib as mpl
import matplotlib.cm as cm

import sys

##---------------------------------------------------------------------------##
    
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
          
sizetext = 20
plt.rcParams['xtick.major.pad'] = '10'
plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': sizetext})

size_ticksnumber = 25 #plt.tick_params(labelsize=size_ticksnumber)
size_axeslabel = 29 #plt.xlabel("$m$",{'fontsize': size_axeslabel})
size_legend_square = 20 #plt.legend(fancybox=True, shadow=True, ncol = 3, numpoints=1,loc = 'upper center', fontsize = size_legend)
size_text = 30 #plt.text(-0.16,1.05, r'$(A)$', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, fontsize= size_text) 
size_ticksnumber_inset = 20
size_axeslabel_inset = 20

markers = ['o', 's', '^', 'v']

lw = 2.5
ms = 9
transparency = 1.

##---------------------------------------------------------------------------##

def fname_mean(dynamics,num_nodes,pp):
    return "../Results/"+str(dynamics)+"/ER_MeanDist_N%d_p%g.dat" % (num_nodes, pp)

def fname_mean_theor(dynamics,num_nodes,pp):
    return "../Results/"+str(dynamics)+"/ER_MeanDist_N%d_p%g_Theoretical.dat" % (num_nodes, pp)

def fname_dij(dynamics,num_nodes, pp, time):
    return "../Results/"+str(dynamics)+"/ER_Dij_N%d_p%g_T%s.dat" % (num_nodes, pp, str('{:.6f}'.format(time)))

def fname_network(dynamics,num_nodes,pp):
    return "../Results/"+str(dynamics)+"/ER_N%d_p%g_network.dat" % (num_nodes, pp)

def sorting_matrix(d_ij,sorted_degs):
    
    sorted_d_ij = np.zeros((N,N))
    
    for new_index in range(0, len(sorted_degs)):
        old_index = int(sorted_degs[new_index][1])
#        print(new_index, old_index)
    
#        d_ij[:, [new_index, old_index]] = d_ij[:, [old_index, new_index]] #swapping columns
#        d_ij[[new_index, old_index], :] = d_ij[[new_index, old_index], :] #swapping rows
        
        sorted_d_ij[new_index] = d_ij[old_index]
        sorted_d_ij[:,new_index] = d_ij[:,old_index]
        
    return sorted_d_ij

def maximum_value_d(N,pp,times_perturbation):
    
    maxv = 0
    for tt in range(1,len(times_perturbation)):
        d_ij = np.loadtxt(fname_dij(N, pp, times_perturbation[tt]), comments='#', delimiter=',')
        if np.max(d_ij) > maxv:
            maxv = np.max(d_ij)
    print(maxv)
    return maxv

###############################################################################
###############################################################################

N = 100
N_l = [50, 100, 150]
N_l = [50]

num_time_points = 100
times_perturbation = np.logspace(-2, 2, num = num_time_points)

dynamics_list = ['Mutualistic', 'Biochemical', 'Population', 'Regulatory',
                 'Epidemics','Synchronization', 'Neuronal', 'NoisyVM']

#dynamics = dynamics_list[6]

###############################################################################
###############################################################################

fig, ax = plt.subplots()

for dynamics in dynamics_list:
  if dynamics != 'Synchronization':
#  if dynamics == 'Neuronal' or dynamics == 'Regulatory':
    for N in N_l:
        pp = 5./N
        tt = []
        d_mean = []
        with open(fname_mean(dynamics,N,pp),'r') as fp:
            for line in fp:
                columns = line.split()
                tt.append(float(columns[0]))
                d_mean.append(float(columns[1]))
          
        
        
#        plt.loglog(tt, d_mean, 'o', lw = 0, alpha = 0.6, label = r'%s' % dynamics, color = colors[dynamics_list.index(dynamics)])
        plt.semilogx(tt, d_mean, 'o', lw = 0, alpha = 0.6, label = r'%s' % dynamics, color = colors[dynamics_list.index(dynamics)])
    
    plt.tick_params(labelsize=size_ticksnumber)
            
    tt = []
    d_t = []
    with open(fname_mean_theor(dynamics,N,pp), "r") as fp:
        for line in fp:
            columns = line.split()
            tt.append(float(columns[0]))
            d_t.append(float(columns[1]))
            
#    plt.loglog(tt, d_t, color = 'b')
    plt.semilogx(tt, d_t, color = 'k')

plt.xlabel(r'$\tau$', {'fontsize': size_axeslabel})
plt.ylabel(r'$d(\tau)$', {'fontsize': size_axeslabel})
plt.legend(frameon=False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.savefig("FigCombined.pdf", bbox_inches='tight')   

#plt.grid()

#plt.xlim((5,30))
















