import numpy as np
import matplotlib.pyplot as plt

def get_average_norm(dist):
    maxis = np.max(np.max(dist, axis=2), axis=1)
    dist = dist / maxis[:,None,None]
    return np.mean(dist, axis=0)

def plot_dist_matrix_evol(results, labels, t_print=[1, 5, 10, 20, 50], hspace = 0.0):
    t_print = [1, 5, 10, 20, 50]

    plt.figure(figsize=(20,30))

    Y = len(t_print)
    X = len(results)

    for i, t in enumerate(t_print):
        for j, res in enumerate(results):
            plt.subplot(X,Y, i+1 + j*Y)
    
            im = plt.imshow(res[t], cmap='jet')
    
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
    
def plot_average_dist_matrix(results, labels, norm = False, hspace = 0.0):
    plt.figure(figsize=(15,30))

    for i, res in enumerate(results):

        plt.subplot(3,len(results)//2,i+1)
    
        if norm:
            im = plt.imshow(np.mean(res,axis=0), cmap='jet')
        else:
            im = plt.imshow(get_average_norm(res), cmap='jet')
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.xticks([])
        plt.yticks([])
        plt.title(labels[i])

    plt.subplots_adjust(wspace=0, hspace=hspace)
    plt.tight_layout()
    plt.show()