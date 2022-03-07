import numpy as np
import networkx as nx


def Model_Mutualistic(xx, t, G, fixed_node, B = .1, R = .1, b = 2):
	"""
	m_0 = "B * xx[i] * (1 - xx[i])"; m_1 = "R * xx[i]"; m_2 = "xx[j]**b / (1 + xx[j]**b)"
	"""
	
	dxdt = []
	for i in range(G.number_of_nodes()):
		if i == fixed_node:
			dxdt.append(0.)
		else:
			m_0 = B * xx[i] * ( 1 - xx[i] )
			m_1 = + R * xx[i]
			m_2 = sum([xx[j]**b / (1 + xx[j]**b) for j in G.neighbors(i)])

			dxdt.append(m_0 + m_1*m_2 )
	return np.array(dxdt)


def Jacobian_Mutualistic(G, SteadyState, B = .1, R = .1, b = 2):

    num_nodes = G.number_of_nodes()

    hill = np.power(SteadyState, b) / (1 + np.power(SteadyState, b))
    hill_prime = b*np.power(SteadyState, b-1) / np.power(1 + np.power(SteadyState, b), 2)
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = B*(1-2*SteadyState[i])
        term2 = R*np.sum([hill[neighbor] for neighbor in list(G.neighbors(i))])
        J[i][i] = term1+term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*SteadyState[i]*hill_prime[neighbor]

    return J







