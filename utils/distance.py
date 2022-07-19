import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eig
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import igraph as ig
import networkx as nx

from tqdm.auto import tqdm

import utils.CommonFunctions as CF


def laplacian(mat, A=1., B=1.):
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

def diffusion_distance(mat, t):
    lapl = laplacian(mat)
    return compute_distance(-lapl, t)

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

def clustering_diffusion_distance(mat, precomputed=None, show=True, method='complete'):
    '''
    method = single, complete, average, weighted, centroid, median, ward
    '''
    
    print('DIFFUSION DISTANCE')
    N = mat.shape[0]
    
    # Compute average diffusion distance
    if precomputed is None:
        print('- Compute average distance...')
        avg_dd = average_distance(-laplacian(mat))
    else:
        print('** Distance already computed')
        avg_dd = precomputed
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_dd), method=method)
    
    if show:
        plt.figure(figsize=(10,6))
        
        ax = plt.subplot(2,2,1)
        #plot_communities(mat, best_part, ax)
        nx.draw(nx.from_numpy_array(mat), ax=ax)
        
        plt.subplot(2,2,2)
        eigvals, eigvecs = eig(-laplacian(mat))
        plt.plot(eigvals.real, eigvals.imag, 'o')
        plt.xlabel(r'$Re(\lambda)$')
        plt.ylabel(r'$Im(\lambda)$')
        plt.title('')
        
        plt.subplot(2,2,3)
        im = plt.imshow(avg_dd, cmap='cividis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title('Average')
        
        plt.subplot(2,2,4)
        dendrogram(Z)
        
        plt.tight_layout()
        plt.show()
    
    return n_clusters, mean_dd, best_clust, best_part

def clustering_jacobian_distance(mat, dynamics, clust_max=20, tmax=None, norm=False, norm_ind=False, show=True, method='complete', args=[]):
    '''
    method = single, complete, average, weighted, centroid, median, ward
    '''
    
    print('COMMUNITY DETECTION - JACOBIAN DISTANCE')
    print('Dynamics: '+dynamics)
    N = mat.shape[0]
    
    # Define total number of clusters to check
    n_clusters = np.arange(2,clust_max)
    n_clusters = np.append(n_clusters, [30, 40, 50, 60, 70])
    mean_dd = np.zeros(len(n_clusters))
    
    # Get steady state
    initial_state = np.random.random(N)
    steady_state = CF.Numerical_Integration(mat, dynamics, initial_state, show=True)
    
    # Compute jacobian
    jacobian = CF.Jacobian(mat, dynamics, steady_state[-1], norm=norm, args=args)
    
    # Compute average jacobian distance
    print('- Compute average distance...')
    avg_dd = average_distance(jacobian)
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_dd), method=method)
    
    print('- Loop over induced graphs...')
    for i, n_clust in enumerate(n_clusters):
        # Get communities
        comms = find_comms(Z, n_clust)
        
        # Compute induced graph
        mat_induced = induced_graph(mat, comms)
        
        # Integrate dynamics up to steady state
        initial_state = np.random.random(mat_induced.shape[0])
        steady_state = CF.Numerical_Integration(mat_induced, dynamics+'_norm', initial_state, show=False)
        
        # Coarse-grain steady state
        #steady_state_induced = np.array( [np.mean(steady_state[-1][comms==idx]) for idx in range(n_clust)] )
        
        # Compute jacobian
        jacobian_induced = CF.Jacobian(mat_induced, dynamics+'_norm', steady_state[-1], norm=norm_ind, args=args)
        
        # Compute diff distance on induced graph
        avg_dd_induced = average_distance(jacobian_induced, display=False, tmax=tmax)
        
        #mean_dd[i] = np.mean(avg_dd_induced[np.triu_indices(mat_induced.shape[0],1)])
        mean_dd[i] = np.mean(avg_dd_induced)
        
        #print('[*] n clust = {} - avg diff dist = {}'.format(n_clust, np.round(mean_dd[i],3)))
    
    # Get best partition
    best_clust = n_clusters[np.argmax(mean_dd)]
    best_part = fcluster(Z, t=best_clust, criterion='maxclust') -1
    
    if show:
        plt.figure(figsize=(10,6))
        
        ax = plt.subplot(2,3,1)
        plot_communities(mat, best_part, ax)
        
        plt.subplot(2,3,2)
        eigvals, eigvecs = eig(jacobian)
        plt.plot(eigvals.real, eigvals.imag, 'o')
        plt.xlabel(r'$Re(\lambda)$')
        plt.ylabel(r'$Im(\lambda)$')
        plt.title('')
        
        plt.subplot(2,3,3)
        im = plt.imshow(avg_dd, cmap='cividis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title('Average')
        
        plt.subplot(2,3,5)
        dendrogram(Z)
        
        plt.subplot(2,3,6)
        plt.plot(n_clusters, mean_dd, 'o-')
        plt.xlabel('Modules')
        plt.ylabel('Mean dist')
        plt.plot(n_clusters[np.argmax(mean_dd)], mean_dd[np.argmax(mean_dd)], 'o', c='red', label=f'best={best_clust}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return n_clusters, mean_dd, best_clust, best_part