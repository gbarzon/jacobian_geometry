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
                 'Epidemics','Synchronization', 'Neuronal']

dynamics = dynamics_list[6]

###############################################################################
###############################################################################

fig, ax = plt.subplots()

for N in N_l:
    pp = 5./N
    tt = []
    d_mean = []
    with open(fname_mean(dynamics,N,pp),'r') as fp:
        for line in fp:
            columns = line.split()
            tt.append(float(columns[0]))
            d_mean.append(float(columns[1]))
      
    
    
#    plt.loglog(tt, d_mean, 'o', lw = 0, alpha = 0.6, label = r'%d' % N,
#               color = colors[N_l.index(N)])
    plt.semilogx(tt, d_mean, 'o', lw = 0, alpha = 0.6, label = r'%d' % N,
               color = colors[N_l.index(N)])

plt.xlabel(r'$\tau$', {'fontsize': size_axeslabel})
plt.ylabel(r'$d(\tau)$', {'fontsize': size_axeslabel})
plt.legend(frameon=False)

plt.tick_params(labelsize=size_ticksnumber)
        
#plt.savefig("Fig6c.pdf", bbox_inches='tight')        

tt = []
d_t = []
with open(fname_mean_theor(dynamics,N,pp), "r") as fp:
    for line in fp:
        columns = line.split()
        tt.append(float(columns[0]))
        d_t.append(float(columns[1]))
        
#plt.loglog(tt, d_t)
plt.semilogx(tt, d_t)

plt.grid()

#plt.xlim((5,30))

################################################################################
################################################################################
################################################################################
#
#N = 150; pp = 5./N
#N = 50; pp = 5./N
#
#G = nx.read_edgelist(fname_network(N,pp))
#
#sorted_degs = []
#degs = dict(G.degree())
#for keys in degs.keys():
#    sorted_degs.append((degs[keys], keys))
##    print(keys, degs[keys])
#sorted_degs.sort()
#
#
#
#max_value_cb = maximum_value_d(N,pp,times_perturbation)
#
#fig, ax2 = plt.subplots(figsize=(10,10))
#
#for tt in range(1,len(times_perturbation)):
#    
#    d_ij = np.loadtxt(fname_dij(N, pp, times_perturbation[tt]), comments='#', delimiter=',')
#    d_ij = np.maximum(d_ij, d_ij.transpose()) #symetrize: https://stackoverflow.com/questions/28904411/making-a-numpy-ndarray-matrix-symmetric
#
#    sorted_d_ij = sorting_matrix(d_ij, sorted_degs)
#
##    plt.imshow(d_ij)
##    plt.imshow(sorted_d_ij, cmap=cm.RdYlGn, vmin=0, vmax=max_value_cb)
#    plt.imshow(sorted_d_ij, cmap=cm.RdYlGn) 
#    plt.xlabel(r'$\longrightarrow k$', {'fontsize': size_axeslabel})
#    plt.ylabel(r'$\longleftarrow k$', {'fontsize': size_axeslabel})
#    ax2.set_title(r'$t=%g$'%times_perturbation[tt], {'fontsize': size_axeslabel})
#    ax2.axes.xaxis.set_ticklabels([])
#    ax2.axes.yaxis.set_ticklabels([])
#    plt.tick_params(labelsize=size_ticksnumber)
#    cb = plt.colorbar(shrink=0.8)
#    cb.ax.tick_params(labelsize=size_axeslabel)
#    
##    plt.show()
#    break















