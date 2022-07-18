import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from matplotlib.ticker import MaxNLocator

import igraph as ig
import networkx as nx

from tqdm.auto import tqdm

import utils.CommonFunctions as CF

def community_detection(mat, method, *args):
    graph = ig.Graph.Weighted_Adjacency(mat.tolist(), mode=ig.ADJ_UNDIRECTED, attr="weight", loops=False)
    
    if method == 'louvain':
        comms = graph.community_multilevel(weights=graph.es['weight'], return_levels=False)
    elif method == 'leiden':
        if len(args)>0:
            resolution_parameter = args[0]
        comms = graph.community_leiden(weights=graph.es['weight'], resolution_parameter=resolution_parameter, n_iterations=-1, objective_function='modularity') #objective_function: Constant Potts Model (CPM) or modularity
    elif method == 'spin_glass':
        comms = graph.community_spinglass(weights=graph.es['weight'], spins=int(1e3))
    elif method == 'infomap':
        comms = graph.community_infomap(edge_weights=graph.es['weight'], trials=10)
    else:
        raise( Exception('Community detection method not defined.\n'))
    
    print(f'Found {len(np.unique(comms.membership))} communities')
    
    return comms.membership

def metrics(comms_true, comms_emp, method):
    if method == 'nmi':
        score = normalized_mutual_info_score(comms_true, comms_emp)
    elif method == 'ami':
        score = adjusted_mutual_info_score(comms_true, comms_emp)
    elif method == 'ari':
        score = adjusted_rand_score(comms_true, comms_emp)
    else:
        raise(Exception('Evaluation method not defined.\n'))
    
    return score

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

def find_comms(Z, n_clust):
    comms = fcluster(Z, t=n_clust, criterion='maxclust') -1
    if n_clust != len(np.unique(comms)):
        print(f'WARNING: found less communities ({len(np.unique(comms))}) than asked ({n_clust})...')
        #n_clust = len(np.unique(comms))
        
    return comms

def induced_graph(mat, comms):
    # Initialize arrays
    n_clust = len(np.unique(comms))
    N = len(comms)
    comms_name = np.arange(n_clust)
    new_adj = np.zeros((n_clust, n_clust))
    
    # Create network of communities
    for node1 in np.arange(N):
        for node2 in np.arange(N):
            new_adj[comms[node1], comms[node2]] += mat[node1,node2]
            
    # Symmetrize and return
    return new_adj

def cluster_within_sum(distance, comms):
    distance = distance**2
    n_comms = len(np.unique(comms))
    avg_dd_cluster = np.zeros(n_comms)
    
    for idx in range(n_comms):
        mask = comms==idx
        mask = mask[:,None] * mask
        
        avg_dd_cluster[idx] = np.sum(distance[mask]) / 2
        
    return avg_dd_cluster

def plot_communities(mat, comms, ax=None):
    n_comms = len(np.unique(comms))
    cmap = plt.cm.get_cmap('plasma', n_comms)
    node_color = [cmap(i) for i in comms]
    #mat = nx.from_numpy_array(mat)
    nx.draw(nx.from_numpy_array(mat), node_color=node_color, with_labels=False, ax=ax)
    #plt.show()
    
def plot_dd_comms(res1, res2=None):
    plt.plot(res1[0], res1[1], 'o-')
    
    plt.plot(res1[0][np.argmax(res1[1])], res1[1][np.argmax(res1[1])], 'o', c='red')
    
    if res2 is not None:
        plt.plot(res2[0], res2[1], 'o-')
        plt.plot(res2[0][np.argmax(res2[1])], res2[1][np.argmax(res2[1])], 'o', c='red')
        
    plt.xlabel('Modules')
    plt.ylabel('Mean diff dist')
        
    plt.show()
    
def gap_statistics(Z, avg_dd, n_clusters, n_rand=100):
    # Init arrays
    dists = np.zeros(len(n_clusters))
    dists_rand = np.zeros((len(n_clusters),n_rand))
    
    for i, nn in enumerate(n_clusters):
        # Compute clusters
        comms = fcluster(Z, t=nn, criterion='maxclust') -1
        
        # Compute real distance
        dists[i] = np.sum(cluster_within_sum(avg_dd, comms))

        # Compute reference distance
        for idx in range(n_rand):
            # Randomized communities
            np.random.shuffle(comms)
            # Compute randomized distance
            dists_rand[i,idx] = np.sum(cluster_within_sum(avg_dd, comms))
    
    # Compute gap statistics
    ll = np.mean(np.log(dists_rand), axis=1)
    gap = ll - np.log(dists)
    sd = np.std(np.log(dists_rand), axis=1)
    #sd = sd * np.sqrt(1 +1/n_clust)
    tmp = np.insert(gap, 0, 0)
    delta = tmp[1:]-tmp[:-1]
    #delta_sd = np.sqrt(sd[1:]**2 + sd[:-1]**2)
    
    return dists, dists_rand, gap, delta, sd

def clustering_diffusion_distance(mat, precomputed=None, clust_max=20, n_rand=100, show=True, method='ward'):
    '''
    method = single, complete, average, weighted, centroid, median, ward
    '''
    
    print('COMMUNITY DETECTION - DIFFUSION DISTANCE')
    N = mat.shape[0]
    
    # Define total number of clusters to check
    n_clusters = np.arange(1,clust_max)
    #n_clusters = np.append(n_clusters, [30, 40, 50, 60, 70])
    
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
    
    # Compute gap statistics
    print('- Compute gap statistics...')
    dists, dists_rand, gap, delta, sd = gap_statistics(Z, avg_dd, n_clusters, n_rand)
    
    # Get best partition
    best_clust = n_clusters[np.argmax(delta)]
    best_part = fcluster(Z, t=best_clust, criterion='maxclust') -1
    
    if show:
        plt.figure(figsize=(10,7))
        
        # Plot average distance matrix
        plt.subplot(2,2,1)
        plt.axis('off')
        im = plt.imshow(avg_dd, cmap='cividis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Average distance')
        
        # Plot network with comunity color
        ax = plt.subplot(2,2,2)
        plot_communities(mat, best_part, ax)
        
        # Plot dendrogram distance
        plt.subplot(2,2,3)
        cmap = plt.cm.get_cmap('plasma', best_clust)
        dendrogram(Z, color_threshold=0)
        plt.ylabel('distance')
        plt.xlabel('node')
        
        # Plot gap statistics
        ax1 = plt.subplot(2,2,4)
        ax1.plot(n_clusters, np.log(dists), 'o-', label='Real', c='k')
        ax1.plot(n_clusters, np.mean(np.log(dists_rand), axis=1), 'o-', label='Randomized', c='blue')

        ax1.set_xlabel('Modules k')
        ax1.set_ylabel(r'$log(W_k)$')
        ax1.legend(loc=1)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2 = ax1.twinx()
        ax2.errorbar(n_clusters, delta, fmt='o-.', yerr=sd, capsize=5, ms=6, c='green', label=r'$\Delta Gap$')
        #ax2.plot(n_clusters, delta, '*-.', c='green', ms=10, label=r'$\Delta Gap$')
        ax2.plot(n_clusters[best_clust-1], delta[best_clust-1], 'o', zorder=10, c='red', ms=6, label=f'best={best_clust}')
        ax2.set_ylabel(r'$\Delta Gap$')
        ax2.legend(loc=7)
        
        plt.tight_layout()
        plt.show()
    
    return best_clust, best_part

def clustering_jacobian_distance(mat, dynamics, clust_max=20, n_rand=100, norm=True, show=True, method='ward', args=[]):
    '''
    method = single, complete, average, weighted, centroid, median, ward
    '''
    
    print('COMMUNITY DETECTION - JACOBIAN DISTANCE')
    print('Dynamics: '+dynamics)
    N = mat.shape[0]
    G = nx.from_numpy_array(mat)
    
    # Define total number of clusters to check
    n_clusters = np.arange(1,clust_max)
    #n_clusters = np.append(n_clusters, [30, 40, 50, 60, 70])
    
    # Get steady state
    initial_state = np.random.random(N)
    steady_state = CF.Numerical_Integration(G, dynamics, initial_state, show=True)
    
    # Compute jacobian
    jacobian = CF.Jacobian(G, dynamics, steady_state[-1], norm=norm)
    
    # Compute average jacobian distance
    print('- Compute average distance...')
    avg_dd = average_distance(jacobian)
    
    # Compute hierarchical clustering
    print('- Compute hierarchical clustering with method {}...'.format(method))
    Z = linkage(squareform(avg_dd), method=method)
    
    # Compute gap statistics
    print('- Compute gap statistics...')
    dists, dists_rand, gap, delta, sd = gap_statistics(Z, avg_dd, n_clusters, n_rand)
    
    # Get best partition
    best_clust = n_clusters[np.argmax(delta)]
    best_part = fcluster(Z, t=best_clust, criterion='maxclust') -1
    
    if show:
        plt.figure(figsize=(10,7))
        
        # Plot average distance matrix
        plt.subplot(2,2,1)
        plt.axis('off')
        im = plt.imshow(avg_dd, cmap='cividis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Average distance')
        
        # Plot network with comunity color
        ax = plt.subplot(2,2,2)
        plot_communities(mat, best_part, ax)
        
        # Plot dendrogram distance
        plt.subplot(2,2,3)
        cmap = plt.cm.get_cmap('plasma', best_clust)
        dendrogram(Z, color_threshold=0)
        plt.ylabel('distance')
        plt.xlabel('node')
        
        # Plot gap statistics
        ax1 = plt.subplot(2,2,4)
        ax1.plot(n_clusters, np.log(dists), 'o-', label='Real', c='k')
        ax1.plot(n_clusters, np.mean(np.log(dists_rand), axis=1), 'o-', label='Randomized', c='blue')

        ax1.set_xlabel('Modules k')
        ax1.set_ylabel(r'$log(W_k)$')
        ax1.legend(loc=1)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2 = ax1.twinx()
        ax2.errorbar(n_clusters, delta, fmt='o-.', yerr=sd, capsize=5, ms=6, c='green', label=r'$\Delta Gap$')
        #ax2.plot(n_clusters, delta, '*-.', c='green', ms=10, label=r'$\Delta Gap$')
        ax2.plot(n_clusters[best_clust-1], delta[best_clust-1], 'o', zorder=10, c='red', ms=6, label=f'best={best_clust}')
        ax2.set_ylabel(r'$\Delta Gap$')
        ax2.legend(loc=7)
        
        plt.tight_layout()
        plt.show()
    
    return best_clust, best_part