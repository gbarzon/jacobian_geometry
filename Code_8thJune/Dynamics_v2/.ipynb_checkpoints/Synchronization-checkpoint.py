import numpy as np
import networkx as nx


def Model_Synchronization(xx, t, G, fixed_node, w = .5, R = .5):
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

def Jacobian_Synchronization(G, SteadyState, w = .5, R = .5):

    num_nodes = G.number_of_nodes()
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        J[i][i] = -R*np.sum([np.cos(SteadyState[neighbor]-SteadyState[i]) for neighbor in list(G.neighbors(i)) if neighbor != i])
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*np.cos(SteadyState[neighbor]-SteadyState[i])

    return J


def Model_Synchronization(xx, t, G, fixed_node, w = .5, R = .5):
	"""
	m_0 = "w"; 
    m_1 = "R"; 
    m_2 = "sin(xx[j] - xx[i])"
    ** not using fixed nodes
	"""    
    num_nodes = len(G)

	dxdt = np.zeros(num_nodes)
    
	for i in range(num_nodes):
        m_0 = w
		m_1 = R
		m_2 = np.sum(G[i]*np.sin(xx - xx[i]))

		dxdt[i] = m_0 + m_1*m_2
        
	return dxdt


def Jacobian_Synchronization(G, SteadyState, w = .5, R = .5):

    num_nodes = len(G)
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        #diagonal terms
        J[i][i] = -R*np.sum([np.cos(SteadyState[neighbor]-SteadyState[i]) for neighbor in list(G.neighbors(i)) if neighbor != i])
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*np.cos(SteadyState[neighbor]-SteadyState[i])

    return J







