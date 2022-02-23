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
import os

##---------------------------------------------------------------------------##
    
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

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

def savevideo(dynamics, N, k_out):
    str_w_vars = "_N%d_kout%g.png" % (N, k_out)
    os.system("ffmpeg -r 5 -i %02d"+str(str_w_vars)+" -vcodec mpeg4 -y "+str(dynamics)+"_video.mp4")
    return

def fname_mean(dynamics,num_nodes,k_out):
    return "../../Results/"+str(dynamics)+"/GN_MeanDist_N%d_kout%g.dat" % (num_nodes, k_out)

def fname_mean_theor(dynamics,num_nodes,k_out):
    return "../../Results/"+str(dynamics)+"/GN_MeanDist_N%d_kout%g_Theoretical.dat" % (num_nodes, k_out)

def fname_dij(dynamics, num_nodes, k_out, time):
    return "../../Results/"+str(dynamics)+"/GN_Dij_N%d_kout%g_T%s.dat" % (num_nodes, k_out, str('{:.6f}'.format(time)))

def fname_network(dynamics,num_nodes,k_out):
    return "../../Results/"+str(dynamics)+"/GN_N%d_kout%g_network.dat" % (num_nodes, k_out)

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

def maximum_value_d(dynamics,N,k_out,times_perturbation):
    
    maxv = 0
    for tt in range(1,len(times_perturbation)):
        d_ij = np.loadtxt(fname_dij(dynamics, N, k_out, times_perturbation[tt]), comments='#', delimiter=',')
        if np.max(d_ij) > maxv:
            maxv = np.max(d_ij)
    print(maxv)
    return maxv

###############################################################################
###############################################################################

num_time_points = 100
times_perturbation = np.logspace(-2, 2, num = num_time_points)

dynamics_list = ['Mutualistic', 'Biochemical', 'Population', 'Regulatory',
                 'Epidemics','Synchronization', 'Neuronal', 'NoisyVM']

dynamics = dynamics_list[-1]

################################################################################
################################################################################
################################################################################

N = 128; k_out = 5
num_groups = 3; nodes_per_group = 16; N = num_groups*nodes_per_group; k_out = 0.1

G = nx.read_edgelist(fname_network(dynamics,N,k_out))

sorted_degs = []
degs = dict(G.degree())
for keys in degs.keys():
    sorted_degs.append((degs[keys], keys))
#    print(keys, degs[keys])
sorted_degs.sort()

max_value_cb = maximum_value_d(dynamics,N,k_out,times_perturbation)

for tt in range(1,len(times_perturbation)):
    fig, ax2 = plt.subplots(figsize=(10,10))
    print(tt)
    d_ij = np.loadtxt(fname_dij(dynamics, N, k_out, times_perturbation[tt]), comments='#', delimiter=',')
    d_ij = np.maximum(d_ij, d_ij.transpose()) #symetrize: https://stackoverflow.com/questions/28904411/making-a-numpy-ndarray-matrix-symmetric

#    sorted_d_ij = sorting_matrix(d_ij, sorted_degs)

#    plt.imshow(d_ij)
#    plt.imshow(sorted_d_ij, cmap=cm.RdYlGn, vmin=0, vmax=max_value_cb)
    plt.imshow(d_ij, cmap=cm.RdYlGn) 
#    plt.xlabel(r'$\longrightarrow k$', {'fontsize': size_axeslabel})
#    plt.ylabel(r'$\longleftarrow k$', {'fontsize': size_axeslabel})
    ax2.set_title(r'$t=%g$'%times_perturbation[tt], {'fontsize': size_axeslabel})
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    plt.tick_params(labelsize=size_ticksnumber)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=size_axeslabel)
    cb.ax.yaxis.offsetText.set(size=size_axeslabel)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    
    plt.savefig("%02d_N%d_kout%g.png" % (tt, N, k_out))
    
    plt.close()
#    break

savevideo(dynamics,N,k_out)
os.system("mv *.png HM_%s/" % dynamics)
os.system("mv *.mp4 HM_%s/" % dynamics)













