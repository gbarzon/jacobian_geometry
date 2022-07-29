import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13
from scipy.linalg import expm, eig
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from utils.plotter import plot_clustermap

import igraph as ig
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
        f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1.1, 1, 1]}, figsize=(14,4))
        #plot_communities(mat, best_part, ax)
        #nx.draw(nx.from_numpy_array(mat), ax=axs[0,0])
        
        plt.subplot(axs[0])
        im = plt.imshow(avg_dd, cmap='cividis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title('Average diffusion distance')
        
        plt.subplot(axs[1])
        eigvals, eigvecs = eig(-laplacian)
        plt.plot(eigvals.real, eigvals.imag, 'o')
        #plt.axvline(0, lw=0.8)
        plt.xlabel(r'$Re(\lambda)$')
        plt.ylabel(r'$Im(\lambda)$')
        plt.title('Eigenvalues')
        
        plt.subplot(axs[2])
        dendrogram(Z, color_threshold=0)
        plt.title(f'Dendrogram (method: {method})')
        
        plt.tight_layout()
        plt.show()
        
        #plot_clustermap(avg_dd)
    
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
    
    if name is not None:
        np.savetxt('results/'+dynamics+'_'+str(args)+'_'+name, avg_dd)
    
    if show:
        f, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1.1, 1, 1]}, figsize=(14,4))
        #plot_communities(mat, best_part, ax)
        #nx.draw(nx.from_numpy_array(mat), ax=axs[0,0])
        
        plt.subplot(axs[0])
        im = plt.imshow(avg_dd, cmap='cividis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title('Average jacobian distance')
        
        plt.subplot(axs[1])
        eigvals, eigvecs = eig(jacobian)
        plt.plot(eigvals.real, eigvals.imag, 'o')
        #plt.axvline(0, lw=0.8)
        plt.xlabel(r'$Re(\lambda)$')
        plt.ylabel(r'$Im(\lambda)$')
        plt.title('Eigenvalues')
        
        plt.subplot(axs[2])
        dendrogram(Z, color_threshold=0)
        plt.title(f'Dendrogram (method: {method})')
        
        plt.tight_layout()
        plt.show()
        
        #plot_clustermap(avg_dd)
    
    return avg_dd, Z