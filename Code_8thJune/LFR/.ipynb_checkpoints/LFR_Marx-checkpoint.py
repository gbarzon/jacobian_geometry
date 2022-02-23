import sys

import networkx as nx
from networkx.generators.community import LFR_benchmark_graph

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve
import networkx as nx
from scipy.linalg import expm
import time


sys.path.append("/home/oriol/Documents/Trento/JacobianGeometry/Code/Dynamics_v2/")
#sys.path.append("/data/oartimevila/JacobGeom/Code/Dynamics/")
import CommonFunctions as cf


def Simu(dynamics, G, infoG):
    """
    Inputs: dynamics (any from  dynamics_list = ['Mutualistic', 'Biochemical', 'Population', 'Regulatory',
                                                 'Epidemics','Synchronization', 'Neuronal', 'NoisyVM'])
            Network topology G
            infoG (list): Info about the network. infoG[0] = network_type; infoG[i>0] = parameters
    """
    start_time = time.time()
    
    num_nodes = G.number_of_nodes()
    num_time_points = 100 # small to decrease computational load
    times = np.linspace(0, 100, num = num_time_points) #time to find the steady state    
    str_tp = "np.logspace(-2, 2., num = num_time_points)" #times at which we look how the perturbation evolves
    times_perturbation = eval(str_tp)

    if dynamics in ['Mutualistic', 'Population', 'Regulatory','Synchronization', 'Neuronal']:
        perturbation_strength = 0.5
    elif dynamics in ['Biochemical', 'Epidemics', 'NoisyVM']:
        perturbation_strength = 0.05
    else:
        print('Dynamics not valid. Manual Exiting')
        exit()
    
    if nx.is_connected(G) == 0:
        raise ValueError('The network should be in a single component. Exit!')
    

    #Integration to get the steady state
    initial_state = np.random.random(len(G.nodes()))   
#    initial_state = np.ones(len(G.nodes()))

    SteadyState_ref = cf.Numerical_Integration(G, dynamics, initial_state, times = times, fixed_node = 1e+6, show = 0)
#        SteadyState_ref = SteadyState_ref[-1]/max(SteadyState_ref[-1])
    SteadyState_ref = SteadyState_ref[-1]

    d_t = cf.Jacobian(G, dynamics, SteadyState_ref, 
                      perturbation_strength, times_perturbation)

    mean_d = np.zeros(num_time_points)
    d_ij = np.zeros((num_time_points,num_nodes,num_nodes))
        
    c = 0
    for node1 in range(0, num_nodes):
      print(node1)
      for node2 in range(node1, num_nodes):
          if node1 != node2:
            Perturbed_SteadyState = cf.Numerical_Integration_perturbation(G,dynamics,SteadyState_ref,
                                                                          node1,node2, perturbation_strength,
                                                                          times = times_perturbation,
                                                                          fixed_node = 1e+6, show = 0)

#                delta1 = abs(Perturbed_SteadyState[:,node1] - SteadyState_ref[node1])
#                delta2 = abs(Perturbed_SteadyState[:,node2] - SteadyState_ref[node2])
#                d_tmp = abs(delta1 - delta2)
#                mean_d += d_tmp

            delta1 = Perturbed_SteadyState[:,node1] - SteadyState_ref[node1]
            delta2 = Perturbed_SteadyState[:,node2] - SteadyState_ref[node2]
            d_tmp = np.sqrt(delta1*delta1 + delta2*delta2)
            
#            if d_tmp != 0:
            c+=1
            mean_d += d_tmp

            if 0 in d_tmp:
                print('0 inside ', c)

            d_ij[:, node1,node2] = d_tmp
            
    d_ij = d_ij/c
    mean_d = mean_d/c
    
    cf.PrintFiles(mean_d, d_t, d_ij, times_perturbation, str_tp, dynamics, G, infoG)

    print("--- %s seconds ---" % (time.time() - start_time))

    return



#Creating LFR benchmark
N = 150
degree_exp = 3.
commsize_exp = 2.
mixing_parm = 0.1
mean_k = 10
min_community = 25

G_lfr = LFR_benchmark_graph(N, degree_exp, commsize_exp, mixing_parm, 
                            average_degree = mean_k, min_community = min_community)

while nx.is_connected(G_lfr) == 0:
    print('Single component required')
    G_lfr = LFR_benchmark_graph(N, degree_exp, commsize_exp, mixing_parm, 
                            average_degree = mean_k, min_community = min_community)

        
infoG = ['LFR',mean_k,mixing_parm]

print('Mean degree', np.mean(list(dict(G_lfr.degree()).values())))



dynamics_list = ['Mutualistic', 'Biochemical', 'Population', 'Regulatory', 
                 'Epidemics','Synchronization', 'Neuronal', 'NoisyVM']
#dyn = 'NoisyVM'
#Simu(dyn,G_lfr,infoG)

for dyn in dynamics_list[::-1]:
    print(dyn)
    Simu(dyn,G_lfr,infoG)    















