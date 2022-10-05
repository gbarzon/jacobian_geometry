import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.linalg import expm, eig
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from seaborn import clustermap, hls_palette
import pandas as pd

import networkx as nx

from tqdm.auto import tqdm

import utils.CommonFunctions as CF


def compute_laplacian(mat, args = []):
    if len(args) > 0:
        A, B = args[0], args[1]
    else:
        A, B = 1., 1.
    print(A, B)
    return A * np.eye(len(mat)) - B * mat / np.sum(mat, axis=1)[:,None]

def compute_distance(mat_enr, t):
    num_nodes = len(mat_enr)
    expM = expm(mat_enr*t)

    d_ij = np.zeros((num_nodes,num_nodes))
    
    for i in range(0, num_nodes-1):
        for j in range(i+1, num_nodes):
            d_ij_tmp = expM[i] - expM[j]
            d_ij[i,j] = np.sqrt(d_ij_tmp.dot(d_ij_tmp))

    return d_ij + d_ij.T

def average_distance(mat_enr, tmax=None, display=True, return_snapshot=False):
    N = mat_enr.shape[0]
    if tmax is None:
        tmax = N

    d_t_ij = np.zeros((tmax,N,N))
    
    if display:
        for t in tqdm(range(1, tmax+1)):
            d_t_ij[t-1] = compute_distance(mat_enr, t)
    else:
        for t in range(1, tmax+1):
            d_t_ij[t-1] = compute_distance(mat_enr, t)
    
    # Average along times
    average_mat = np.mean(d_t_ij, axis=0)
    average_dist = np.mean(d_t_ij, axis=2).mean(axis=1)
    
    if return_snapshot:
        return average_mat, average_dist, d_t_ij
    else:
        return average_mat, average_dist

def plot_communities(mat, comms, ax=None):
    n_comms = len(np.unique(comms))
    cmap = plt.cm.get_cmap('plasma', n_comms)
    node_color = [cmap(i) for i in comms]
    #mat = nx.from_numpy_array(mat)
    nx.draw(nx.from_numpy_array(mat), node_color=node_color, with_labels=False, ax=ax)
    #plt.show()
    

def diffusion_distance(mat, show=True, method='ward', args=[], name=None, comms=None, cs=None, title=None):
    '''
    method = single, complete, average, weighted, centroid, median, ward
    '''
    
    print('DIFFUSION DISTANCE')
    N = mat.shape[0]
    
    # Compute laplacian...
    print('- Compute laplacian...')
    laplacian = compute_laplacian(mat, args)
    
    # Compute average diffusion distance
    print('- Compute average distance...')
    avg_mat, avg_dd = average_distance(-laplacian)
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_mat), method=method)
    
    if name is not None:
        np.savetxt('results/diffusion_'+name, avg_mat)
    
    if show:
        plot_results(avg_mat, -laplacian, Z, 'Diffusion', method, comms, cs, title)
    
    return avg_mat, avg_dd, Z, -laplacian

def jacobian_distance(mat, dynamics, norm=False, show=True, method='ward', args=[], name=None, comms=None, cs=None, title=None):
    '''
    method = single, complete, average, weighted, centroid, median, ward
    '''
    
    print('JACOBIAN DISTANCE')
    print('Dynamics: '+dynamics)
    N = mat.shape[0]
    
    # Get steady state
    initial_state = np.random.random(N)
    steady_state = CF.Numerical_Integration(mat, dynamics, initial_state, show=True, args=args)
        
    # Compute jacobian
    jacobian = CF.Jacobian(mat, dynamics, steady_state[-1], norm=norm, args=args)
    
    # Compute average jacobian distance
    print('- Compute average distance...')
    avg_mat, avg_dd = average_distance(jacobian)
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_mat), method=method)
    
    # Save results
    if name is not None:
        np.savetxt('results/'+dynamics+'_'+str(args)+'_'+name, avg_mat)
    
    # Plot results
    if show:
        plot_results(avg_mat, jacobian, Z, dynamics, method, comms, cs, title)
    
    return avg_mat, avg_dd, Z, jacobian

'''
def plot_results(avg_dd, mat_to_exp, Z, dynamics, method):
    #f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1.1, 1, 1]}, figsize=(14,4))
    
    fig = plt.figure(figsize=(14,4), constrained_layout=False)
    gs = GridSpec(2,3, width_ratios=[1, 1.3, 1], height_ratios=[1.4, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax0 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:, 2])
    
    #plot_communities(mat, best_part, ax)
    #nx.draw(nx.from_numpy_array(mat), ax=axs[0,0])
        
    plt.subplot(ax0)
    im = plt.imshow(avg_dd, cmap='cividis')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    #cbar.ax.set_title(r'$\overline{d}_{ij}$')
    cbar.ax.set_ylabel(r'$\overline{d}_{ij}$', rotation=90)
    plt.axis('off')
    #plt.title('Average distance')
    
    # Get eigevalues and eigenvectors
    eigvals, eigl, eigr = eig(mat_to_exp, left=True, right=True)
    # Get order
    order = np.argsort(-eigvals)
    # Get partecipation ratio
    pr = ( np.sum(eigl**2 *eigr**2, axis=0) / np.sum(eigl * eigr, axis=0)**2 )**-1
    
    plt.subplot(ax1)
    plt.plot(np.arange(len(avg_dd))+1, eigvals[order], 'o-')
    #plt.xlabel(r'Rank index $i$')
    plt.ylabel(r'$\lambda_i$')
    plt.xscale('log')
    #plt.title('Eigenvalues')
    ax1.axes.xaxis.set_ticklabels([])
    
    plt.subplot(ax2)
    plt.plot(np.arange(len(avg_dd))+1, pr[order], 'o-', c='red')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'Rank index $i$')
    plt.ylabel(r'PR$_i$')
    
    plt.subplot(ax3)
    dendrogram(Z, color_threshold=0)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    #plt.title(f'Dendrogram (method: {method})')
        
    plt.suptitle(dynamics)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.5)
    plt.show()
'''
    

def plot_results(avg_dd, mat_to_exp, Z, dynamics, method, comms, row_colors, title, figsize=(12,5)):
    N = len(mat_to_exp)
    
    ### Plot clustermap
    # Create linkage
    mylinkage = linkage(squareform(avg_dd), method=method)
    
    # Setup row colors
    if row_colors is None and comms is not None:
        #labels = np.repeat(np.arange(n_comms),N//n_comms)
        #row_colors = [palette[i] for i in labels]
        n_comms = len(np.unique(comms))
        palette = hls_palette(n_comms)
        row_colors = [palette[i] for i in comms]
    
    # Create clustermap
    clust_map = clustermap(avg_dd, row_linkage=mylinkage, col_linkage=mylinkage,
                           cmap='cividis',
                           #cbar_kws = dict(orientation='horizontal'),
                           cbar_pos=None,
                           row_colors=row_colors,
                           tree_kws=dict(linewidths=1.4),
                           figsize=figsize)

    clust_map.ax_col_dendrogram.set_visible(False) # hide dendrogram above columns
    
    # Update position
    clust_map.gs.update(left=0.41)
    
    # Setup colorbar
    cbnorm = Normalize(vmin=np.min(avg_dd),vmax=np.max(avg_dd)) #setting the scale
    cb = plt.colorbar(cm.ScalarMappable(norm=cbnorm, cmap='cividis'),ax=clust_map.ax_heatmap,pad=0.13)
    cb.ax.set_title(r'$\overline{d}_{ij}$')
    #axins1 = inset_axes(clust_map.ax_heatmap, width="50%", height="5%", loc='upper center')

    
    ### Plot eigenvalues and partecipation ratio
    # Add gridspec
    gs = GridSpec(2,1, left=0.0, right=0.39, bottom=0.15, top=.85, height_ratios=[1, 1])
    ax1 = clust_map.fig.add_subplot(gs[0])
    ax2 = clust_map.fig.add_subplot(gs[1])
    
    # Get eigevalues and eigenvectors
    eigvals, eigl, eigr = eig(mat_to_exp, left=True, right=True)
    # Get order
    eigvals = eigvals.real
    order = np.argsort(-eigvals)
    # Get partecipation ratio
    pr = ( np.sum(eigl**2 *eigr**2, axis=0) / np.sum(eigl * eigr, axis=0)**2 )**-1
    
    # Plot eigenvalues
    plt.subplot(ax1)
    plt.plot(np.arange(len(avg_dd))+1, eigvals[order], 'o-')
    #plt.xlabel(r'Rank index $i$')
    plt.ylabel(r'$\lambda_i$')
    plt.xscale('log')
    #plt.title('Eigenvalues')
    ax1.axes.xaxis.set_ticklabels([])
    
    # Plot partecipation ratio
    plt.subplot(ax2)
    plt.plot(np.arange(len(avg_dd))+1, pr[order], 'o-', c='red')
    plt.ylim(-5,N*11/10)
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'Rank index $i$')
    plt.ylabel(r'PR$_i$')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, top=1.1)
    plt.show()