import numpy as np
import networkx as nx

def Model_Synchronization(xx, t, G, fixed_node, w = 0., R = 1.):
	"""
	m_0 = "w"; 
    m_1 = "R"; 
    m_2 = "sin(xx[j] - xx[i])"
	"""

	dxdt = []
	for i in range(G.number_of_nodes()):
		if i == fixed_node:
			dxdt.append(0.)
		else:
			m_0 = w
			m_1 = R
			m_2 = sum([np.sin(xx[j] - xx[i]) for j in G.neighbors(i)])

			dxdt.append(m_0 + m_1*m_2 )
	return np.array(dxdt)

def Jacobian_Synchronization(G, SteadyState, w = 0., R = 1.):

    num_nodes = G.number_of_nodes()
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        J[i][i] = -R*np.sum([np.cos(SteadyState[neighbor]-SteadyState[i]) for neighbor in list(G.neighbors(i)) if neighbor != i])
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*np.cos(SteadyState[neighbor]-SteadyState[i])

    return J

'''
def Model_Synchronization(xx, t, G, fixed_node, w = 0., R = 1.):
    """
    m_0 = "w"; 
    m_1 = "R"; 
    m_2 = "sin(xx[j] - xx[i])"
    """    
    
    #G = G / np.sum(G, axis=0)[:,None]
    
    return w + R * np.sum(G*np.sin(xx - xx[:,None]), axis=1)

def Jacobian_Synchronization(G, xx, w = 0., R = 1.):
    num_nodes = G.shape[0]
    
    #G = G / np.sum(G, axis=0)[:,None]
    
    G[np.diag_indices_from(G)] = 0
    J = - R * np.sum(G*np.cos(xx - xx[:,None]), axis=1) * np.eye(num_nodes) + R * G*np.cos(xx - xx[:,None])
    
    return J

def Model_Synchronization_norm(xx, t, G, fixed_node, w = 0., R = 1.):
    """
    m_0 = "w"; 
    m_1 = "R"; 
    m_2 = "sin(xx[j] - xx[i])"
    """    
    
    G = G / np.sum(G, axis=0)[:,None]
    
    return w + R * np.sum(G*np.sin(xx - xx[:,None]), axis=1)

def Jacobian_Synchronization_norm(G, xx, w = 0., R = 1.):
    num_nodes = G.shape[0]
    
    G = G / np.sum(G, axis=0)[:,None]
    
    G[np.diag_indices_from(G)] = 0
    J = - R * np.sum(G*np.cos(xx - xx[:,None]), axis=1) * np.eye(num_nodes) + R * G*np.cos(xx - xx[:,None])
    
    return J
'''