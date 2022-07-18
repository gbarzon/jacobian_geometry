import numpy as np
import networkx as nx

def Model_Population(xx, t, G, fixed_node, B = 1., R = 1., b = 3, a = 2):
    """
    m_0 = "-B * xx[i]**b"; 
    m_1 = "R "; 
    m_2 = "xx[j]**a"
    """    

    xx_powa = np.power(xx,a)
    xx_powb = np.power(xx,b)
    
    dxdt = []
    for i in range(G.number_of_nodes()):
        if i == fixed_node:
            dxdt.append(0.)
        else:
            m_0 = -B * xx_powb[i]
            m_1 = R
            m_2 = sum([xx_powa[j] for j in G.neighbors(i)])
            
            dxdt.append(m_0 + m_1*m_2)
            
    return np.array(dxdt)


def Jacobian_Population(G, SteadyState, B = 1., R = 1., b = 3, a = 2):

    num_nodes = G.number_of_nodes()
    
    ss_powb = np.power(SteadyState,(b-1))
    ss_powa = np.power(SteadyState,(a-1))
    
    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = -B*b*ss_powb[i]
        if i in list(G.neighbors(i)): #selfloop!
            term2 = R*a*ss_powa[i]
        else:
            term2 = 0
        J[i][i] = term1+term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = R*a*ss_powa[neighbor]

    return J