import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve
import networkx as nx
from scipy.linalg import expm, eig

import utils.Dynamics.Mutualistic as mut
import utils.Dynamics.Biochemical as bio
import utils.Dynamics.Population as pop
import utils.Dynamics.Regulatory as reg
import utils.Dynamics.Regulatory2 as reg2
import utils.Dynamics.Epidemics as epi
import utils.Dynamics.Synchronization as syn
import utils.Dynamics.Neuronal as neu 
import utils.Dynamics.NoisyVM as nvm
import utils.Dynamics.DiffusionInteraction as dfi

from tqdm.auto import tqdm

dynamics_list = ['Mutualistic', 'Biochemical', 'Population', 'Regulatory','Epidemics','Synchronization', 'Neuronal', 'NoisyVM', 'Diffusion']
dynamics_short = ['MUT', 'BIO', 'POP', 'REG', 'EPI', 'SYN', 'NEU', 'NVM', 'DIF']

def get_average_distance_matrix(dist, norm=False):
    if norm:
        maxis = np.max(np.max(dist, axis=2), axis=1)
        dist = dist / maxis[:,None,None]
    return np.mean(dist, axis=0)

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
						times = np.linspace(0,50, num = 2000),
						fixed_node = 1e+12, 
						show = False, 
						epsilon = 1e0,
                        args = []):
    '''
	Kwargs:
	- fixed_nodes: list of integers of nodes indexes to be kept constant during integrations 
				   e.g.: if fixed_nodes = [n] -> dx_n/dt = 0, default is [1e+12] to not select nodes (unless
				   network has more than 1e+12 nodes ..)
    '''
    
    G = nx.from_numpy_array(G)
    print(args)
    
    if dynamics == 'Mutualistic':
        xx = odeint(mut.Model_Mutualistic, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Biochemical':
        xx = odeint(bio.Model_Biochemical, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Population':
        xx = odeint(pop.Model_Population, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Regulatory':
        xx = odeint(reg.Model_Regulatory, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Regulatory2':
        xx = odeint(reg2.Model_Regulatory2, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Epidemics':
        xx = odeint(epi.Model_Epidemics, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Epidemics_norm':
        xx = odeint(epi.Model_Epidemics_norm, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Synchronization':
        xx = odeint(syn.Model_Synchronization, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Synchronization_norm':
        xx = odeint(syn.Model_Synchronization_norm, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'Neuronal':
        xx = odeint(neu.Model_Neuronal, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'NoisyVM':
        xx = odeint(nvm.Model_NoisyVM, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'NoisyVM_norm':
        xx = odeint(nvm.Model_NoisyVM_norm, initial_state, times, args = (G,fixed_node, *args))
    elif dynamics == 'DiffInt':
        xx = odeint(dfi.Model_DiffusionInteraction, initial_state, times, args = (G,fixed_node, *args))
    else:
        print(dynamics)
        raise ValueError('Unknown dynamics. Manual exiting')
        
    # check if steady state is reached and filter
#    xx = Check_SteadyState(times,xx,G,epsilon = epsilon)

    # Compute characteristic time
    char_time = Check_Char_Time(xx,times,epsilon=epsilon)
        
    # plotting each node dynamics
    if show == True:
        #G = nx.from_numpy_array(G)
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
            
    return xx

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

def Jacobian(G, dynamics, SteadyState, norm = False, args = []):
    G = nx.from_numpy_array(G)
    print(args)
    
    if dynamics == 'Mutualistic':
        J = mut.Jacobian_Mutualistic(G, SteadyState, *args)
    elif dynamics == 'Biochemical':
        J = bio.Jacobian_Biochemical(G, SteadyState, *args)
    elif dynamics == 'Population':
        J = pop.Jacobian_Population(G, SteadyState, *args)
    elif dynamics == 'Regulatory':
        J = reg.Jacobian_Regulatory(G, SteadyState, *args)
    elif dynamics == 'Regulatory2':
        J = reg2.Jacobian_Regulatory2(G, SteadyState, *args)
    elif dynamics == 'Epidemics':
        J = epi.Jacobian_Epidemics(G, SteadyState, *args)
    elif dynamics == 'Epidemics_norm':
        J = epi.Jacobian_Epidemics_norm(G, SteadyState, *args)
    elif dynamics == 'Synchronization':
        J = syn.Jacobian_Synchronization(G, SteadyState, *args)
    elif dynamics == 'Synchronization_norm':
        J = syn.Jacobian_Synchronization_norm(G, SteadyState, *args)
    elif dynamics == 'Neuronal':
        J = neu.Jacobian_Neuronal(G, SteadyState, *args)
    elif dynamics == 'NoisyVM':
        J = nvm.Jacobian_NoisyVM(G, SteadyState, *args)
    elif dynamics == 'NoisyVM_norm':
        J = nvm.Jacobian_NoisyVM(G, SteadyState, *args)
    elif dynamics == 'DiffInt':
        J = dfi.Jacobian_DiffusionInteraction(G, SteadyState, *args)
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
    
    eigs = eig(J)[0]
    #larg_eig = np.max(np.abs(eigs))
    #print('largest eig:', larg_eig)
    if norm:
        print(r'Normalizing jacobian - $\lambda_{max}=$'+str(np.max(np.abs(eigs))))
        J = J / np.max(np.abs(eigs))

    return J  
