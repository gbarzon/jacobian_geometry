import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve
import networkx as nx
from scipy.linalg import expm, eigh

import Mutualistic as mut
import Biochemical as bio
import Population as pop
import Regulatory as reg
import Regulatory2 as reg2
import Epidemics as epi
import Synchronization as syn
import Neuronal as neu 
import NoisyVM as nvm

from tqdm.auto import tqdm

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    #From https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def Check_SteadyState(times,xx,G,epsilon):
	'''
	Function to check if dynamics has reached a steady state: this is evaluated when
	increment = max{ (x[t+1] - x[t])/(x[t]*dt) } < epsilon
	'''
	delta_t = (times[-1]-times[0])/len(times)
	steadystate = False

	for t in range(1,len(times)):
		increments = [abs((xx[:,i][t]-xx[:,i][t-1])/(xx[:,i][t]*delta_t)) for i in G.nodes()]

		if max(increments) < epsilon:
			steadystate = True
			#t_ss = t
			break
		else:
			continue

	if steadystate == False:
		raise ValueError('Did not reach steady state - consider higher epislon or longer integration')

	# return filtered dynamics up to the steady state found
	#return  xx[:t_ss,:]
	return xx

def Check_Char_Time(xx,times,epsilon):
	'''
	Function to check if dynamics has reached a steady state: this is evaluated when
	increment = max{ (x[t+1] - x[t])/(x[t]*dt) } < epsilon
	'''
	delta_t = times[1]-times[0]
	char_time = np.inf

	for t in range(1,len(times)):
		increments = np.sum( np.abs((xx[t]-xx[t-1])/xx[t]/delta_t) )

		if increments < epsilon:
			char_time = times[t]
			break
		else:
			continue

	#if char_time == np.inf:
		#char_time = len(xx[0])
        #raise ValueError('Did not reach steady state - consider higher epislon or longer integration')
        

	# return filtered dynamics up to the steady state found
	return char_time


 
def Numerical_Integration(G, dynamics, initial_state, 
						times = np.linspace(0,20, num = 2000),
						fixed_node = 1e+12, 
						show = False, 
						epsilon = 1e0):
    '''
	Kwargs:
	- fixed_nodes: list of integers of nodes indexes to be kept constant during integrations 
				   e.g.: if fixed_nodes = [n] -> dx_n/dt = 0, default is [1e+12] to not select nodes (unless
				   network has more than 1e+12 nodes ..)
    '''

    if dynamics == 'Mutualistic':
        xx = odeint(mut.Model_Mutualistic, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Biochemical':
        xx = odeint(bio.Model_Biochemical, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Population':
        xx = odeint(pop.Model_Population, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Regulatory':
        xx = odeint(reg.Model_Regulatory, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Regulatory2':
        xx = odeint(reg2.Model_Regulatory2, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Epidemics':
        xx = odeint(epi.Model_Epidemics, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Synchronization':
        xx = odeint(syn.Model_Synchronization, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'Neuronal':
        xx = odeint(neu.Model_Neuronal, initial_state, times, args = (G,fixed_node))
    elif dynamics == 'NoisyVM':
        xx = odeint(nvm.Model_NoisyVM, initial_state, times, args = (G,fixed_node))
    else:
        print(dynamics)
        raise ValueError('Unknown dynamics. Manual exiting')
        
    # check if steady state is reached and filter
#    xx = Check_SteadyState(times,xx,G,epsilon = epsilon)

    # Compute characteristic time
    char_time = Check_Char_Time(xx,times,epsilon=epsilon)
        
    # plotting each node dynamics
    if show == True:
        metadata = dict(G.nodes(data=True))
        if metadata[list(G.nodes())[0]] == {}: #No community structure, no metadata
            plt.figure(1)
            if dynamics == 'Synchronization':
                for i in G.nodes():
                    if i == fixed_node:
                        plt.plot(times[:len(xx[:,i])],xx[:,i],'o-')
                    else:
                        plt.plot(times[:len(xx[:,i])],xx[:,i]%(2*np.pi))
                    plt.title("time evolution of nodes state - up to steady state [RESCALED]")
            else:
                for i in G.nodes():
                    if i == fixed_node:
                        plt.plot(times[:len(xx[:,i])],xx[:,i],'o-')
                    else:
                        plt.plot(times[:len(xx[:,i])],xx[:,i])
                    plt.axvline(char_time)
                    plt.title("time evolution of nodes state - up to steady state")
            plt.show()
        
        elif list(metadata[list(G.nodes())[0]].keys()) == ['block']: #I'm dealing with communities, called "blocks"
            
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
            plt.figure(1)
            for i in G.nodes():
                if i == fixed_node:
                    plt.plot(times[:len(xx[:,i])],xx[:,i],'o-')
                else:
                    plt.plot(times[:len(xx[:,i])],xx[:,i], color = colors[list(metadata.values())[i]['block']])
                plt.axvline(char_time)
                plt.title("time evolution of nodes state - up to steady state")
        	
            if dynamics == 'Synchronization':
                plt.figure(2)
                for i in G.nodes():
                    if i == fixed_node:
                        plt.plot(times[:len(xx[:,i])],xx[:,i],'o-')
                    else:
                        plt.plot(times[:len(xx[:,i])],xx[:,i]%(2*np.pi), color = colors[list(metadata.values())[i]['block']] )
                    plt.title("time evolution of nodes state - up to steady state [RESCALED]")
            plt.show()
                    
        else:
            raise ValueError('Metadata not identified, manual exiting!')
            
    return xx, char_time

def Numerical_Integration_perturbation(G, dynamics, SteadyState_ref, node1, node2, perturbation_strength,
                						times = np.linspace(0,20, num = 2000),
                						fixed_node = 1e+12, 
                						show = False, 
                						epsilon = 10**(-8)):
    """
	Kwargs:
	- fixed_nodes: list of integers of nodes indexes to be kept constant during integrations 
				   e.g.: if fixed_nodes = [n] -> dx_n/dt = 0, default is [1e+12] to not select nodes (unless
				   network has more than 1e+12 nodes ..)
    """

    ci = np.array(SteadyState_ref)
    ci[node1] = SteadyState_ref[node1] + perturbation_strength # why -? maybe it's indifferent if i perturb up or down...
    ci[node2] = SteadyState_ref[node2] + perturbation_strength

    if ci[node1] < 0 or ci[node2] < 0:
        ci[node1] = SteadyState_ref[node1] + perturbation_strength
        ci[node2] = SteadyState_ref[node2] + perturbation_strength
        raise ValueError("Manual exiting; %s or %s should be >=0" % (ci[node1], ci[node2]))

    if dynamics == 'Mutualistic':
        xx = odeint(mut.Model_Mutualistic, ci, times, args = (G,fixed_node))
    elif dynamics == 'Biochemical':
        xx = odeint(bio.Model_Biochemical, ci, times, args = (G,fixed_node))
    elif dynamics == 'Population':
        xx = odeint(pop.Model_Population, ci, times, args = (G,fixed_node))
    elif dynamics == 'Regulatory':
        xx = odeint(reg.Model_Regulatory, ci, times, args = (G,fixed_node))
    elif dynamics == 'Regulatory2':
        xx = odeint(reg2.Model_Regulatory2, ci, times, args = (G,fixed_node))
    elif dynamics == 'Epidemics':
        xx = odeint(epi.Model_Epidemics, ci, times, args = (G,fixed_node))
    elif dynamics == 'Synchronization':
        xx = odeint(syn.Model_Synchronization, ci, times, args = (G,fixed_node))
    elif dynamics == 'Neuronal':
        xx = odeint(neu.Model_Neuronal, ci, times, args = (G,fixed_node))
    elif dynamics == 'NoisyVM':
        xx = odeint(nvm.Model_NoisyVM, ci, times, args = (G,fixed_node))
    else:
        print(dynamics)
        raise ValueError('Unknown dynamics. Manual exiting')
        
#   plotting each node dynamics
    if show == True:
        for i in G.nodes():
            print(i)
            if i == fixed_node:
                plt.plot(times[:len(xx[:,i])],xx[:,i],'o-')
            else:
                plt.plot(times[:len(xx[:,i])],xx[:,i])
        plt.title("time evolution of nodes state - up to steady state")
        plt.show()

    return xx

def myNumerical_Integration_perturbation(G, dynamics, SteadyState_ref, node1, perturbation_strength,
                						times = np.linspace(0,20, num = 2000),
                						fixed_node = 1e+12, 
                						show = False, 
                						epsilon = 10**(-8)):
    """
	Kwargs:
	- fixed_nodes: list of integers of nodes indexes to be kept constant during integrations 
				   e.g.: if fixed_nodes = [n] -> dx_n/dt = 0, default is [1e+12] to not select nodes (unless
				   network has more than 1e+12 nodes ..)
    """

    ci = np.array(SteadyState_ref)
    ci[node1] = SteadyState_ref[node1] + perturbation_strength

    if ci[node1] < 0:
        raise ValueError("Manual exiting; %s should be >=0" % ci[node1])

    if dynamics == 'Mutualistic':
        xx = odeint(mut.Model_Mutualistic, ci, times, args = (G,fixed_node))
    elif dynamics == 'Biochemical':
        xx = odeint(bio.Model_Biochemical, ci, times, args = (G,fixed_node))
    elif dynamics == 'Population':
        xx = odeint(pop.Model_Population, ci, times, args = (G,fixed_node))
    elif dynamics == 'Regulatory':
        xx = odeint(reg.Model_Regulatory, ci, times, args = (G,fixed_node))
    elif dynamics == 'Regulatory2':
        xx = odeint(reg2.Model_Regulatory2, ci, times, args = (G,fixed_node))
    elif dynamics == 'Epidemics':
        xx = odeint(epi.Model_Epidemics, ci, times, args = (G,fixed_node))
    elif dynamics == 'Synchronization':
        xx = odeint(syn.Model_Synchronization, ci, times, args = (G,fixed_node))
    elif dynamics == 'Neuronal':
        xx = odeint(neu.Model_Neuronal, ci, times, args = (G,fixed_node))
    elif dynamics == 'NoisyVM':
        xx = odeint(nvm.Model_NoisyVM, ci, times, args = (G,fixed_node))
    else:
        print(dynamics)
        raise ValueError('Unknown dynamics. Manual exiting')
        
#   plotting each node dynamics
    if show == True:
        for i in G.nodes():
            print(i)
            if i == fixed_node:
                plt.plot(times[:len(xx[:,i])],xx[:,i],'o-')
            else:
                plt.plot(times[:len(xx[:,i])],xx[:,i])
        plt.title("time evolution of nodes state - up to steady state")
        plt.show()

    return xx

def oldJacobian(G, dynamics, SteadyState, perturbation_strength, t_list):
    
    num_nodes = G.number_of_nodes()
    
    if dynamics == 'Mutualistic':
        J = mut.Jacobian_Mutualistic(G, SteadyState)
    elif dynamics == 'Biochemical':
        J = bio.Jacobian_Biochemical(G, SteadyState)
    elif dynamics == 'Population':
        J = pop.Jacobian_Population(G, SteadyState)
    elif dynamics == 'Regulatory':
        J = reg.Jacobian_Regulatory(G, SteadyState)
    elif dynamics == 'Regulatory2':
        J = reg2.Jacobian_Regulatory2(G, SteadyState)
    elif dynamics == 'Epidemics':
        J = epi.Jacobian_Epidemics(G, SteadyState)
    elif dynamics == 'Synchronization':
        J = syn.Jacobian_Synchronization(G, SteadyState)
    elif dynamics == 'Neuronal':
        J = neu.Jacobian_Neuronal(G, SteadyState)
    elif dynamics == 'NoisyVM':
        J = nvm.Jacobian_NoisyVM(G, SteadyState)
    else:
        print(dynamics)
        raise ValueError('Unknown dynamics. Manual exiting')
    
    #Checking that matrix is symmetric (in the position of the elements, not in their value)
    J2 = J/J
    J2[np.isnan(J2)] = 0
    if check_symmetric(J2) == False:
        raise ValueError("The Jacobian is not symmetric. Manual exiting")
    
    d_t = [] # average distance at different t
    
    for t in t_list:
        expJ = expm(J*t)
    
        d = 0
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                d_ij_tmp = perturbation_strength*(expJ[:,i] - expJ[:,j]) # qui potrei mettere dopo perturbation strength
                d_ij = np.sqrt(d_ij_tmp.dot(d_ij_tmp))
                d += d_ij

        d = d / (num_nodes*num_nodes)

        d_t.append(d)

    return d_t

def Jacobian(G, dynamics, SteadyState, t_list, perturbation_strength=1., return_snapshot=False):
    
    num_nodes = G.number_of_nodes()
    T = len(t_list)
    
    if dynamics == 'Mutualistic':
        J = mut.Jacobian_Mutualistic(G, SteadyState)
    elif dynamics == 'Biochemical':
        J = bio.Jacobian_Biochemical(G, SteadyState)
    elif dynamics == 'Population':
        J = pop.Jacobian_Population(G, SteadyState)
    elif dynamics == 'Regulatory':
        J = reg.Jacobian_Regulatory(G, SteadyState)
    elif dynamics == 'Regulatory2':
        J = reg2.Jacobian_Regulatory2(G, SteadyState)
    elif dynamics == 'Epidemics':
        J = epi.Jacobian_Epidemics(G, SteadyState)
    elif dynamics == 'Synchronization':
        J = syn.Jacobian_Synchronization(G, SteadyState)
    elif dynamics == 'Neuronal':
        J = neu.Jacobian_Neuronal(G, SteadyState)
    elif dynamics == 'NoisyVM':
        J = nvm.Jacobian_NoisyVM(G, SteadyState)
    else:
        print(dynamics)
        raise ValueError('Unknown dynamics. Manual exiting')
    
    '''
    # Checking that matrix is symmetric (in the position of the elements, not in their value)
    J2 = J/J
    J2[np.isnan(J2)] = 0
    if check_symmetric(J2) == False:
        raise ValueError("The Jacobian is not symmetric. Manual exiting")
    '''
    
    # Normalize the time wrt norm of jacobian
    #print('J_norm: ', np.sum(np.abs(J)))
    #t_list = t_list / np.sum(np.abs(J))
    
    #larg_eig = eigh(J, eigvals_only=True, subset_by_index=[num_nodes-1, num_nodes-1])
    eigs = eigh(J, eigvals_only=True)
    larg_eig = np.max(np.abs(eigs))
    print('largest eig:', larg_eig)
    print('eigs sum:', np.sum(eigs))
    
    # Normalize times wrt larg eig
    #t_list = t_list / larg_eig
    
    d_t = np.zeros(T) # average distance at different t
    
    if return_snapshot:
        d_t_ij = np.zeros((T,num_nodes,num_nodes))
    
    for t in tqdm(range(T)):
        expJ = expm(J*t_list[t])
    
        d = 0
        
        for i in range(0, num_nodes):
            for j in range(i+1, num_nodes):
                d_ij_tmp = perturbation_strength*(expJ[:,i] - expJ[:,j]) # qui potrei mettere dopo perturbation strength
                d_ij = np.sqrt(d_ij_tmp.dot(d_ij_tmp))
                d += d_ij
                
                if return_snapshot:
                    d_t_ij[t,i,j] = d_ij
                    
        # symmetrize distance matrix       
        if return_snapshot:
            d_t_ij[t] = d_t_ij[t] + d_t_ij[t].T #- np.diag(d_t_ij[t].diagonal())

        d = d * 2 / num_nodes / (num_nodes-1)

        d_t[t] = d
        
    if return_snapshot:
        return d_t, d_t_ij
    else:
        return d_t
    
def Laplacian(Aij, t_list, norm=True, return_snapshot=False):
    num_nodes = len(Aij)
    T = len(t_list)
    
    if norm:
        L = np.eye(num_nodes) - Aij / np.sum(Aij, axis=1)[:,None]
    else:
        L = np.eye(num_nodes) - Aij
        
    eigs = eigh(L, eigvals_only=True)
    larg_eig = np.max(np.abs(eigs))
    print('largest eig:', larg_eig)
    print('eigs sum:', np.sum(eigs))
    
    d_t = np.zeros(T) # average distance at different t
    
    if return_snapshot:
        d_t_ij = np.zeros((T,num_nodes,num_nodes))
    
    for t in tqdm(range(T)):
        expL = expm(-L*t_list[t])
        d = 0
        
        for i in range(0, num_nodes):
            for j in range(i+1, num_nodes):
                d_ij_tmp = expL[:,i] - expL[:,j]
                d_ij = np.sqrt(d_ij_tmp.dot(d_ij_tmp))
                d += d_ij
                
                if return_snapshot:
                    d_t_ij[t,i,j] = d_ij
                    
        # symmetrize distance matrix       
        if return_snapshot:
            d_t_ij[t] = d_t_ij[t] + d_t_ij[t].T #- np.diag(d_t_ij[t].diagonal())

        d = d * 2 / num_nodes / (num_nodes-1)

        d_t[t] = d
        
    if return_snapshot:
        return d_t, d_t_ij
    else:
        return d_t

def LogHisto(data):
    bins = np.logspace(min(np.log10(data)), max(np.log10(data)), num = 10)
    xx = [0.5*(bins[i] + bins[i+1]) for i in range(0, len(bins) - 1)]
    histo, _ = np.histogram(data, bins = bins, density = True)
    return xx, histo

def LinHisto(data):
    bins = np.linspace(min(data), max(data), num = 10)
    xx = [0.5*(bins[i] + bins[i+1]) for i in range(0, len(bins) - 1)]
    histo, _ = np.histogram(data, bins = bins, density = True)
    return xx, histo


def PrintFiles(mean_d, d_t, d_ij, times_perturbation, str_tp, dynamics, G, infoG):
  print(infoG)
  if infoG[0] == 'GN':

    if len(infoG) != 2:
        raise ValueError('InfoG should have two args. Exiting')
    
    num_nodes = G.number_of_nodes()
    network_type = infoG[0]
    k_out = infoG[1]
    
    with open("Results/%s/%s_MeanDist_N%d_kout%g.dat" % (dynamics, network_type, num_nodes, k_out), "w+") as fp:
        for i in range(0, len(mean_d)):
            fp.write("%g %g\n" % (times_perturbation[i], mean_d[i]))

    with open("Results/%s/%s_MeanDist_N%d_kout%g_Theoretical.dat" % (dynamics, network_type, num_nodes, k_out), "w+") as fp:
        for i in range(0, len(d_t)):
            fp.write("%g %g\n" % (times_perturbation[i], d_t[i]))

    for tt in range(len(times_perturbation)):
        fname = "Results/%s/%s_Dij_N%d_kout%g_T%s.dat" % (dynamics, network_type, num_nodes, k_out, str('{:.6f}'.format(times_perturbation[tt])))
        header = "# %s" % str_tp
        mat = np.matrix(d_ij[tt])
        with open(fname, "w+") as fp:
            fp.write("# %s\n" % header)
            for line in mat:
                np.savetxt(fp, line, fmt='%g',delimiter=',', newline='\n')

    fname_network = "Results/%s/%s_N%d_kout%g_network.dat" % (dynamics, network_type, num_nodes, k_out)
    nx.write_edgelist(G, fname_network, data=False, encoding='utf-8')                   

  elif infoG[0] == 'LFR':

    if len(infoG) != 3:
        raise ValueError('InfoG should have two args. Exiting')
    
    num_nodes = G.number_of_nodes()
    network_type = infoG[0]
    k_mean = infoG[1]
    mixparm = infoG[2]
    
    with open("Results/%s/%s_MeanDist_N%d_kmean%g_mixparm%g.dat" % (dynamics, network_type, num_nodes, k_mean, mixparm), "w+") as fp:
        for i in range(0, len(mean_d)):
            fp.write("%g %g\n" % (times_perturbation[i], mean_d[i]))

    with open("Results/%s/%s_MeanDist_N%d_kmean%g_mixparm%g_Theoretical.dat" % (dynamics, network_type, num_nodes, k_mean, mixparm), "w+") as fp:
        for i in range(0, len(d_t)):
            fp.write("%g %g\n" % (times_perturbation[i], d_t[i]))

    for tt in range(len(times_perturbation)):
        fname = "Results/%s/%s_Dij_N%d_kmean%g_mixparm%g_T%s.dat" % (dynamics, network_type, num_nodes, k_mean, mixparm, str('{:.6f}'.format(times_perturbation[tt])))
        header = "# %s" % str_tp
        mat = np.matrix(d_ij[tt])
        with open(fname, "w+") as fp:
            fp.write("# %s\n" % header)
            for line in mat:
                np.savetxt(fp, line, fmt='%g',delimiter=',', newline='\n')

    fname_network = "Results/%s/%s_N%d_kmean%g_mixparm%g_network.dat" % (dynamics, network_type, num_nodes, k_mean, mixparm)
    nx.write_edgelist(G, fname_network, data=False, encoding='utf-8')

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    with open("Results/%s/%s_N%d_kmean%g_mixparm%g_CommStruct.dat" % (dynamics, network_type, num_nodes, k_mean, mixparm), "w+") as fp:
        fp.write("# Each row is a community \n")
        for comms in communities:
            for node in comms:
                fp.write("%d " % node)
#                print(node,end=' ')
#            print("\n")
            fp.write("\n")


  elif infoG[0] == 'ER':

    if len(infoG) != 2:
        raise ValueError('InfoG should have two args. Exiting')
    
    num_nodes = G.number_of_nodes()
    network_type = infoG[0]
    pp = infoG[1]
    
    with open("Results/%s/%s_MeanDist_N%d_p%g.dat" % (dynamics, network_type, num_nodes, pp), "w+") as fp:
        for i in range(0, len(mean_d)):
            fp.write("%g %g\n" % (times_perturbation[i], mean_d[i]))

    with open("Results/%s/%s_MeanDist_N%d_p%g_Theoretical.dat" % (dynamics, network_type, num_nodes, pp), "w+") as fp:
        for i in range(0, len(d_t)):
            fp.write("%g %g\n" % (times_perturbation[i], d_t[i]))

    for tt in range(len(times_perturbation)):
        fname = "Results/%s/%s_Dij_N%d_p%g_T%s.dat" % (dynamics, network_type, num_nodes, pp, str('{:.6f}'.format(times_perturbation[tt])))
        header = "# %s" % str_tp
        mat = np.matrix(d_ij[tt])
        with open(fname, "w+") as fp:
            fp.write("# %s\n" % header)
            for line in mat:
                np.savetxt(fp, line, fmt='%g',delimiter=',', newline='\n')

    fname_network = "Results/%s/%s_N%d_p%g_network.dat" % (dynamics, network_type, num_nodes, pp)
    nx.write_edgelist(G, fname_network, data=False, encoding='utf-8')

  else:
    print('Wrong infoG',infoG)
    raise ValueError('Manual exiting')  
    
  return





    
