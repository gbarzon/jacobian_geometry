

import numpy as np
import networkx as nx



def Model_Biochemical(xx, t, G, fixed_node, F = .1, B = .1, R = .1):
    """
    m_0 = "F-B*xx[i]"
    m_1 = "xx[i]"
    m_2 = "R*xx[j]"
    """
    dxdt = []
    

    for i in range(G.number_of_nodes()):
        if i == fixed_node:
            dxdt.append(0.)
        else:
            m_0 = F - B * xx[i]
            m_1 = - R * xx[i]
            m_2 = sum([xx[j] for j in G.neighbors(i)])

            dxdt.append(m_0 + m_1*m_2 )
    return np.array(dxdt)


def Jacobian_Biochemical(G, SteadyState, F = .1, B = .1, R = .1):

    num_nodes = G.number_of_nodes()
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = -B
        term2 = -R*np.sum([SteadyState[neighbor] for neighbor in list(G.neighbors(i))])
        J[i][i] = term1+term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = -R*SteadyState[i]

    return J







