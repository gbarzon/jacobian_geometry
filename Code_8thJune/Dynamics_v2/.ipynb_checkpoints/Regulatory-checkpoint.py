

import numpy as np
import networkx as nx



def Model_Regulatory(xx, t, G, fixed_node, B = .1, R = .1, a = 1, h = 1./3):
    """
    m_0 = "-B * xx[i] ** a"; 
    m_1 = "-R"; 
    m_2 = "xx[j]**h / (1 + xx[j]**h)"
    """    

    hill = np.power(xx,h) / (1 + np.power(xx,h))
    dxdt = []
    for i in range(G.number_of_nodes()):
        if i == fixed_node:
            dxdt.append(0.)
        else:
            m_0 = -B * xx[i]**a
            m_1 =  R
            m_2 = sum([hill[j] for j in G.neighbors(i)])

            dxdt.append(m_0 + m_1*m_2 )
    return np.array(dxdt)

def Jacobian_Regulatory(G, SteadyState, B = .1, R = .1, a = 1, h = 1./3):

    num_nodes = G.number_of_nodes()
    
    ss_pow = np.power(SteadyState,a-1)
    hill_prime = h*np.power(SteadyState, h-1) / np.power(1 + np.power(SteadyState, h), 2)
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = -B*a*ss_pow[i]
        if i in list(G.neighbors(i)): #selfloop!
            term2 = R*hill_prime[i]
        else:
            term2 = 0
    
        J[i][i] = term1+term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*hill_prime[neighbor]

    return J







