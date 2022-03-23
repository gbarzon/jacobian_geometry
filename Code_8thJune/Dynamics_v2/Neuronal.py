

import numpy as np
import networkx as nx


def Model_Neuronal(xx, t, G, fixed_node, B = 1., C = 1., R = 1.):
    """
    m_0 = "-B * xx[i] + C * tanh(xx[i])"; 
    m_1 = "R";
    m_2 = "tanh(xx[j])"
    """    
    
    tan_xx = np.tanh(xx)
    dxdt = []; 
        
    for i in range(G.number_of_nodes()):
        if i == fixed_node:
            dxdt.append(0.)
        else:
            m_0 = -B * xx[i] + C * tan_xx[i]
            m_1 = R
            m_2 = sum([tan_xx[j] for j in G.neighbors(i)])

            dxdt.append(m_0 + m_1*m_2 )
    return np.array(dxdt)


def Jacobian_Neuronal(G, SteadyState, B = 1., C = 1., R = 1.):

    num_nodes = G.number_of_nodes()

    sech2_ss = 1/(np.power(np.cosh(SteadyState),2))
     
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = -B + C*sech2_ss[i]
        if i in list(G.neighbors(i)): #selfloop!
            term2 = R*sech2_ss[i]
        else:
            term2 = 0
        J[i][i] = term1+term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*sech2_ss[neighbor]

    return J







