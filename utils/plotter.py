import numpy as np
import matplotlib.pyplot as plt
from seaborn import clustermap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from utils.CommonFunctions import get_average_distance_matrix

cmap = 'gist_heat'

### Distance matrix vs time, average distance
def plot_dist_matrix_evol(results, labels, t_print=[1, 5, 10, 20, 'avg'], hspace = 0.0): # removed 'avg_norm'

    plt.figure(figsize=(20,30))

    Y = len(t_print)
    X = len(results)

    for i, t in enumerate(t_print):
        for j, res in enumerate(results):
            plt.subplot(X,Y, i+1 + j*Y)
            
            if t=='avg':
                im = plt.imshow(get_average_distance_matrix(res, norm=False), cmap=cmap)
            elif t=='avg_norm':
                im = plt.imshow(get_average_distance_matrix(res, norm=True), cmap=cmap)
            else:
                im = plt.imshow(res[t], cmap=cmap)
    
            plt.colorbar(im,fraction=0.046, pad=0.03)
            plt.xticks([])
            plt.yticks([])
            if j==0:
                plt.title(r'$\tau = $'+str(t))
            if i==0:
                plt.ylabel(labels[j])
                
    plt.subplots_adjust(wspace=0, hspace=hspace)
    plt.tight_layout()
    plt.show()
    
def plot_average_dist_matrix(results, labels, n_rows=3, n_columns=3, norm = False, hspace = 0.0, tmin=0):
    plt.figure(figsize=(10,20))

    for i, res in enumerate(results):
        plt.subplot(n_rows,n_columns,i+1)
    
        im = plt.imshow(get_average_distance_matrix(res[tmin:], norm), cmap=cmap)
            
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.xticks([])
        plt.yticks([])
        plt.title(labels[i])

    plt.subplots_adjust(wspace=0, hspace=hspace)
    plt.tight_layout()
    plt.show()
    
def plot_average_dist_matrix_square(results, labels_rows, labels_cols, norm = False, hspace = 0.0, tmin=0):
    plt.figure(figsize=(15,30))

    for i, res in enumerate(results):
        plt.subplot(len(labels_rows),len(labels_cols),i+1)
    
        im = plt.imshow(get_average_distance_matrix(res[tmin:], norm), cmap=cmap)
            
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.xticks([])
        plt.yticks([])
        
        if i<len(labels_cols):
            plt.title(labels_cols[i])
        if i%len(labels_cols)==0:
            plt.ylabel(labels_rows[i//len(labels_cols)])

    plt.subplots_adjust(wspace=0, hspace=hspace)
    plt.tight_layout()
    plt.show()
    
### Clustering
# clustermap want 2d array of observation, not 2d symmetric distance matrix
# thus, we have to pre compute the linkage and pass it to clustermap
# https://stackoverflow.com/questions/38705359/how-to-give-sns-clustermap-a-precomputed-distance-matrix
def plot_clustermap(matrix, row_colors=None, method='complete', figsize=(10,10), linewidths=1.5, title='', name_to_save=None, dpi=200):
    
    mylinkage = linkage(squareform(matrix), method=method)
    
    clust_map = clustermap(matrix, row_linkage=mylinkage, col_linkage=mylinkage,
                           cbar_kws = dict(orientation="horizontal"),
                           row_colors=row_colors,
                           tree_kws=dict(linewidths=linewidths),
                           figsize=figsize)

    clust_map.ax_col_dendrogram.set_visible(False) # hide dendrogram above columns
    #clust_map.set(xlabel='my x label', ylabel='my y label')
    
    # Setup colorbar
    x0, _y0, _w, _h = clust_map.cbar_pos
    clust_map.ax_cbar.set_position([x0*20, 0.825, clust_map.ax_row_dendrogram.get_position().width*2, 0.02])
    clust_map.ax_cbar.set_title('distance')
    
    plt.title(title)
    
    if name_to_save is not None:
        plt.savefig(name_to_save+'.png', dpi=dpi)
    plt.show()