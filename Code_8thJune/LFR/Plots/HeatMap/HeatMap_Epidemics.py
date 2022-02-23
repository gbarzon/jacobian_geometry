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

def savevideo(dynamics, N, kmean, mixparm):
    str_w_vars = "_N%d_kmean%g_mixparm%g.png" % (N, kmean, mixparm)
    os.system("ffmpeg -r 5 -i %02d"+str(str_w_vars)+" -vcodec mpeg4 -y "+str(dynamics)+"_video.mp4")
    return

def fname_mean(dynamics, num_nodes, kmean, mixparm):
    return "../../Results/"+str(dynamics)+"/LFR_MeanDist_N%d_kmean%g_mixparm%g.dat" % (num_nodes, kmean, mixparm)

def fname_mean_theor(dynamics, num_nodes, kmean, mixparm):
    return "../../Results/"+str(dynamics)+"/LFR_MeanDist_N%d_kmean%g_mixparm%g_Theoretical.dat" % (num_nodes, kmean, mixparm)

def fname_dij(dynamics, num_nodes, kmean, mixparm, time):
    return "../../Results/"+str(dynamics)+"/LFR_Dij_N%d_kmean%g_mixparm%g_T%s.dat" % (num_nodes, kmean, mixparm, str('{:.6f}'.format(time)))

def fname_network(dynamics, num_nodes, kmean, mixparm):
    return "../../Results/"+str(dynamics)+"/LFR_N%d_kmean%g_mixparm%g_network.dat" % (num_nodes, kmean, mixparm)

def fname_comms(dynamics, num_nodes, kmean, mixparm):
    return "../../Results/%s/LFR_N%d_kmean%g_mixparm%g_CommStruct.dat" % (dynamics, num_nodes, kmean, mixparm)

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

def load_communities(filename):
    comms = []
    with open(filename,'r') as fp:
        for line in fp:
            columns = line.split()
            if columns[0] != '#':
                comms.append([int(node) for node in columns])

    #sort list of list from small to large communities
    comms.sort(key=len)

    return comms

def sorting_matrix_by_communities(d_ij, sorted_degs, communities):

    sorted_d_ij = np.zeros((N,N))

    new_index = 0
    for i in range(0, len(communities)):
        for old_index in communities[i]:
            sorted_d_ij[new_index] = d_ij[old_index]
            sorted_d_ij[:,new_index] = d_ij[:,old_index]
            new_index += 1

    return sorted_d_ij

def maximum_value_d(dynamics, N, kmean, mixparm, times_perturbation):
    
    maxv = 0
    for tt in range(1,len(times_perturbation)):
        d_ij = np.loadtxt(fname_dij(dynamics, N, kmean, mixparm, times_perturbation[tt]), comments='#', delimiter=',')
        if np.max(d_ij) > maxv:
            maxv = np.max(d_ij)
    print(maxv)
    return maxv

###############################################################################
###############################################################################

num_time_points = 100
times_perturbation = np.logspace(-2, 2, num = num_time_points)

dynamics_list = ['Mutualistic', 'Biochemical', 'Population', 'Regulatory',
                 'Epidemics','Synchronization', 'Neuronal']

dynamics = dynamics_list[4]

################################################################################
################################################################################
################################################################################

N = 250; kmean = 10; mixparm = 0.1

G = nx.read_edgelist(fname_network(dynamics,N,kmean,mixparm))

sorted_degs = []
degs = dict(G.degree())
for keys in degs.keys():
    sorted_degs.append((degs[keys], keys))
#    print(keys, degs[keys])
sorted_degs.sort()

max_value_cb = maximum_value_d(dynamics,N,kmean,mixparm,times_perturbation)

communities = load_communities(fname_comms(dynamics, N, kmean, mixparm))
ticks_communities = np.cumsum([0] + [len(comms) for comms in communities])

for tt in range(len(times_perturbation)-1,len(times_perturbation)):
    fig, ax2 = plt.subplots(figsize=(10,10))
    print(tt)
    d_ij = np.loadtxt(fname_dij(dynamics, N, kmean, mixparm, times_perturbation[tt]), comments='#', delimiter=',')
    d_ij = np.maximum(d_ij, d_ij.transpose()) #symetrize: https://stackoverflow.com/questions/28904411/making-a-numpy-ndarray-matrix-symmetric

    
    sorted_d_ij = sorting_matrix_by_communities(d_ij, sorted_degs, communities)

#    plt.imshow(d_ij)
#    plt.imshow(sorted_d_ij, cmap=cm.RdYlGn, vmin=0, vmax=max_value_cb)
    plt.imshow(d_ij, cmap=cm.RdYlGn) 
#    plt.xlabel(r'$\longrightarrow k$', {'fontsize': size_axeslabel})
#    plt.ylabel(r'$\longleftarrow k$', {'fontsize': size_axeslabel})
    ax2.set_title(r'$t=%g$'%times_perturbation[tt], pad=20, fontsize=size_axeslabel)
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.set_xticks(ticks_communities)
    ax2.set_yticks(ticks_communities)
    plt.tick_params(labelsize=size_ticksnumber, length=10, width=2)
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=size_axeslabel)
    cb.ax.yaxis.offsetText.set(size=size_axeslabel)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    
    plt.savefig("%02d_N%d_kmean%g_mixparm%g.png" % (tt, N, kmean, mixparm))

    
#    plt.close()
#    break

#savevideo(dynamics,N,kmean,mixparm)
#os.system("mv *.png HM_%s/" % dynamics)
#os.system("mv *.mp4 HM_%s/" % dynamics)













