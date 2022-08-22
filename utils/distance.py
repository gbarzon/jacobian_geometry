import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13
from matplotlib.gridspec import GridSpec

from scipy.linalg import expm, eig
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from seaborn import clustermap

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
    average = np.mean(d_t_ij, axis=0)
    
    if return_snapshot:
        return average, d_t_ij
    else:
        return average

def plot_communities(mat, comms, ax=None):
    n_comms = len(np.unique(comms))
    cmap = plt.cm.get_cmap('plasma', n_comms)
    node_color = [cmap(i) for i in comms]
    #mat = nx.from_numpy_array(mat)
    nx.draw(nx.from_numpy_array(mat), node_color=node_color, with_labels=False, ax=ax)
    #plt.show()
    

def diffusion_distance(mat, show=True, method='ward', args=[], name=None):
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
    avg_dd = average_distance(-laplacian)
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_dd), method=method)
    
    if name is not None:
        np.savetxt('results/diffusion_'+name, avg_dd)
    
    if show:
        plot_results(avg_dd, -laplacian, Z, 'Diffusion', method)
    
    return avg_dd, Z

def jacobian_distance(mat, dynamics, norm=False, show=True, method='ward', args=[], name=None):
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
    avg_dd = average_distance(jacobian)
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_dd), method=method)
    
    # Save results
    if name is not None:
        np.savetxt('results/'+dynamics+'_'+str(args)+'_'+name, avg_dd)
    
    # Plot results
    if show:
        plot_results(avg_dd, jacobian, Z, dynamics, method)
    
    return avg_dd, Z

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
    

def plot_results(avg_dd, mat_to_exp, Z, dynamics, method, figsize=(12,6)):
    ### Plot clustermap
    mylinkage = linkage(squareform(avg_dd), method=method)
    
    clust_map = clustermap(avg_dd, row_linkage=mylinkage, col_linkage=mylinkage,
                           cmap='cividis',
                           #cbar_kws = dict(orientation='vertical'),
                           cbar_pos = None,
                           row_colors=None,
                           tree_kws=dict(linewidths=1.4),
                           figsize=figsize)

    clust_map.ax_col_dendrogram.set_visible(False) # hide dendrogram above columns
    #clust_map.set(xlabel='my x label', ylabel='my y label')
    
    # Update position
    clust_map.gs.update(left=0.5)
    
    # Setup colorbar
    cbnorm = Normalize(vmin=0-0.5,vmax=5+0.5,clip=False) #setting the scale
    #cb = plt.colorbar(cm.ScalarMappable(norm=cbnorm, cmap=newcmp),fraction=1,ax=ax_color,ticks=np.arange(6))
    
    ### Plot eigenvalues and partecipation ratio
    # Add gridspec
    gs = GridSpec(2,1, left=0.0, right=0.4, bottom=0.1, top=.8, height_ratios=[1, 1])
    ax1 = clust_map.fig.add_subplot(gs[0])
    ax2 = clust_map.fig.add_subplot(gs[1])
    
    # Get eigevalues and eigenvectors
    eigvals, eigl, eigr = eig(mat_to_exp, left=True, right=True)
    # Get order
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
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'Rank index $i$')
    plt.ylabel(r'PR$_i$')
        
    plt.suptitle(dynamics)
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.5)
    plt.show()
        
    #plot_clustermap(avg_dd)
