import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from scipy import stats
import random as rn

# sebastiano.bontorin@unitn.it
# ------------------------------------------

def Probability_Matrix(G, adjacency):
	
    '''
    Args:
    G = nx.Graph object
    adjacency = nx.ndarray (2d) weighted adjacency matrix

    Returns:
    probabiliy matrix (or transition matrix)
    '''

    num_nodes = G.number_of_nodes()
    
	# Seb's version; requires fully connected network
#    probability = np.zeros(shape = (num_nodes,num_nodes))
#    for n in range(num_nodes):
#        for m in range(num_nodes):
#			
#            # Probability to jump from n to m is equal to link weight adjacency[m][n]
#			# normalized by the weighted out-degree of node n			
#            F_n = np.sum([adjacency[l][n] for l in G.neighbors(n)])
#            F_m = np.sum([adjacency[k][m] for k in G.neighbors(m)])
#			
#            probability[m][n] = adjacency[m][n]/float(F_n)
#            probability[n][m] = adjacency[n][m]/float(F_m)
#    return probability

    probability2 = np.zeros(shape = (num_nodes,num_nodes))
    for n in range(num_nodes):
        # Probability to jump from n to m is equal to link weight adjacency[m][n]		
        # normalized by the weighted out-degree of node n
        F_n = np.sum([adjacency[l][n] for l in G.neighbors(n)])
        if F_n == 0:
            raise ValueError('node %d is isolated and it should be not' % n)
        for m in range(num_nodes):            
            probability2[m][n] = adjacency[m][n]/float(F_n)

    return probability2

def Brockmann_Distance_Matrix(G, probability):
	'''
	Computes distances between two neighbors A_{ij} == 1 according to equation
	D_{ij} = 1 - np.log10(P_{ij})
	'''

	num_nodes = G.number_of_nodes()
	distance_matrix = np.zeros(shape = (num_nodes,num_nodes))

	for n in range(num_nodes):
		for m in G.neighbors(n):
				distance_matrix[m][n] = 1.0 - np.log10(probability[m][n])
				distance_matrix[n][m] = 1.0 - np.log10(probability[n][m])

	return distance_matrix

def Barzel_and_Brockmann_Distance(G, tau, distance_matrix, node):
    '''
    Args:
    - G = nx.Graph object
    - tau = numpy array calculated via Barzel universal exponent: tau = degrees**theta
    - distance_matrix = numpy ndarray calculated from function: Brockmann_Distance_Matrix()
    - node = source node for the associated distance
    
    Returns:
    - brockmann and barzel distances from: node
    
    ( to obtain the full matrix use: Barzel_and_Brockmann_Matrix() )

    Note:
    Paths between node and the rest of the network are calculated such that:
    I find the length (L) of the shortest path between source and target, and then
    I find the set ensemble of paths connecting them that have at maximum length equal to L + 3 (if L < 5)
    or L + 2 otherwise. The reason is to not search for longer paths that require A LOT more time and
    are likely to not be the ones with higher probability or smallest delay 
    (this assumption has to be checked if valid, otherwise increase to a larger cutoff)

    Computes distance as:
    Barzel:
        For each path sums the cumulative time delay of each node in the path - source node non included
        then select the path with smallest cumulative time delay (and that will be the distance)
        
    Brockmann:
        For each path computes the cumulative sum of each D_{ij} in the path and the selects the path
        with smallest cumulative D (analogous to selecting the path of which the product of transition probabilities is maximum)
    '''

    num_nodes = G.number_of_nodes()
    barzel_distances = np.zeros(num_nodes)
    brockmann_distances = np.zeros(num_nodes)
    Gnodes = sorted(list(G.nodes()))


    #
    # Generation of paths between node and the rest of the network
    #

    #oriol:
    targets = Gnodes[0:node] + Gnodes[node+1:]

    print("Building paths list ..")
    ALL_PATHS = [[] for j in Gnodes] #each element j is a list of paths from node to j

    shortest_lengths = [len(nx.shortest_path(G, source=node, target=n)) - 1 for n in Gnodes]
    max_dist = int(max(shortest_lengths))

    #oriol:
    subsets = {int(k+1):[] for k in range(max_dist)}
    # fill subsets; oriol
    for n in targets:
        subsets[shortest_lengths[n]].append(n)

    # now path gen with each subset
    for L in list(subsets.keys()):
        if L < 4:
            cut = L + 3
        else:
            cut = L + 2
        for path in list(nx.all_simple_paths(G, source=node, target=subsets[L],cutoff = cut)):
            ending = path[-1] #last element is always the target node
            ALL_PATHS[ending].append(path)

	#
	# Calculation of distances
	#
    print(f"Calculation of distances from node {node}")
	
    for j in Gnodes:
        if j != node:

            paths = ALL_PATHS[j]
            lengths, brockmann_lengths = [],[]

            if len(paths) != 0:
                for path in paths:
                    D_jn = 0
                    tau_length = 0

                    #Barzel
                    for p in path[1:]: #do not count source
                        tau_length += tau[p]
                    lengths.append(tau_length)

					# Brockmann
                    for ind,value in enumerate(path):
                        if ind != (len(path)-1):
                            j_ind = path[ind]
                            i_ind = path[ind+1]
                            D_jn = D_jn + distance_matrix[i_ind][j_ind]
                    brockmann_lengths.append(D_jn)
            else:
                print(f"Not found path between node {node} and node {j} - consider bigger cutoff")

            barzel_distances[j] = np.min(lengths)
            brockmann_distances[j] = np.min(brockmann_lengths)

#    raise ValueError
    return barzel_distances, brockmann_distances


def Barzel_and_Brockmann_Matrix(G,tau, distance_matrix):
    '''
    Produces the full matrices of distances
    '''
    
    num_nodes = G.number_of_nodes()
    barzel_matrix = np.zeros(shape=(num_nodes,num_nodes))
    brockmann_matrix = np.zeros(shape=(num_nodes,num_nodes))

    for node in range(num_nodes):
        barzel_distances,brockmann_distances = Barzel_and_Brockmann_Distance(G, tau, distance_matrix, node)
        barzel_matrix[node] = barzel_distances
        brockmann_matrix[node] = brockmann_distances
		
    return barzel_matrix, brockmann_matrix



def Plot_Brockmann_Barzel(G):

	# Adjacency and parameters
    A = nx.to_numpy_matrix(G,nodelist=sorted(G.nodes()),weight = 'weight')
    adjacency = np.squeeze(np.asarray(A))

    degrees = np.array([np.float(d) for (n, d) in sorted(G.degree())]) #Oriol has modified this, because it was not sorted
    tau = degrees**(-1) # for a simple SIS

	# Auxiliary matrices	
    probability_matrix = Probability_Matrix(G, adjacency)	
    distance_matrix =  Brockmann_Distance_Matrix(G, probability_matrix)

	# Distances
    barzel_matrix, brockmann_matrix = Barzel_and_Brockmann_Matrix(G,tau, distance_matrix)

	# Plots 
    plt.imshow(barzel_matrix)
    plt.title('Barzel matrix')
    plt.show()
    
    plt.imshow(brockmann_matrix)
    plt.title('Brockmann matrix')
    plt.show()














