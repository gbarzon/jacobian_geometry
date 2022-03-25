import numpy as np
import networkx as nx


def Model_DiffusionInteraction(xx, t, G, fixed_node, R = 1., C = 1.):
    """
    m_0 = "- B xx[i]"; 
    m_1 = "R xx[j]/k_i";
    m_2 = "C xx[i]xx[j]/k_i"
    """    

    dxdt = []
        
    for i in range(G.number_of_nodes()):
        if i == fixed_node:
            dxdt.append(0.)
        else:
            m_0 = -1. * xx[i]
            m_1 = R
            m_2 = sum([xx[j]/G.degree(i) for j in G.neighbors(i)])
            m_3 = C * sum([xx[i]*xx[j]/G.degree(i) for j in G.neighbors(i)])
            
            #if abs(m_3)>1e-8:
            #    raise ValueError(sum([xx[i]*xx[j]/G.degree(i) for j in G.neighbors(i)]), R, C)
                #m_3 = 0.
            
            dxdt.append(m_0 + m_1*m_2 + m_3 )
    return np.array(dxdt)


def Jacobian_DiffusionInteraction(G, SteadyState, R = 1., C = 1.):

    num_nodes = G.number_of_nodes()

    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = -1.
        term2 = C * sum([SteadyState[j] for j in G.neighbors(i)]) / G.degree(i)
        '''
        if i in list(G.neighbors(i)): #selfloop!
            term2 = C/G.degree(i)
        else:
            term2 = 0
        '''
        J[i][i] = term1 + term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R/G.degree(i) +  C*SteadyState[i]/G.degree(i)

    return J







