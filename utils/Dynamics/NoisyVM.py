import numpy as np
import networkx as nx


def Model_NoisyVM(xx, t, G, fixed_node, A = 0., B = 1., C = 1.):
    """
    We need to ensure that B \geq A + C
    
    m_0 = "A - B xx[i]"; 
    m_1 = "C";
    m_2 = "xx[j]/k_i"
    """    

    num_nodes = len(xx)
    dxdt = np.zeros(num_nodes)
    
    G = G / np.sum(G,axis=1)[:,None]
    
    dxdt = A - B*xx + C * np.sum(xx*G, axis=1)
        
    return dxdt

def Jacobian_NoisyVM(G, SteadyState, A = 0., B = 1., C = 1.):

    num_nodes = G.shape[0]

    #diagonal terms
    J = - B * np.eye(num_nodes) + C * G / np.sum(G, axis=1)[:,None]

    return J

def Model_NoisyVM_norm(xx, t, G, fixed_node, A = 0., B = 1., C = 1.):
    """
    We need to ensure that B \geq A + C
    
    m_0 = "A - B xx[i]"; 
    m_1 = "C";
    m_2 = "xx[j]/k_i"
    """    

    num_nodes = len(xx)
    dxdt = np.zeros(num_nodes)
    
    G = G / np.sum(G,axis=1)[:,None]
    
    dxdt = A - B*xx + C * np.sum(xx*G, axis=1)
        
    return dxdt

def Jacobian_NoisyVM_norm(G, SteadyState, A = 0., B = 1., C = 1.):

    num_nodes = G.shape[0]

    #diagonal terms
    J = - B * np.eye(num_nodes) + C * G / np.sum(G, axis=1)[:,None]

    return J

'''
def Model_NoisyVM(xx, t, G, fixed_node, A = 0., B = 1., C = 1.):
    """
    We need to ensure that B \geq A + C
    
    m_0 = "A - B xx[i]"; 
    m_1 = "C";
    m_2 = "xx[j]/k_i"
    """    

    dxdt = []
        
    for i in range(G.number_of_nodes()):
        if i == fixed_node:
            dxdt.append(0.)
        else:
            m_0 = A - B * xx[i]
            m_1 = C
            m_2 = sum([xx[j]/G.degree(i) for j in G.neighbors(i)])

            dxdt.append(m_0 + m_1*m_2 )
    return np.array(dxdt)


def Jacobian_NoisyVM(G, SteadyState, A = 0., B = 1., C = 1.):

    num_nodes = G.number_of_nodes()

    J = np.zeros((num_nodes, num_nodes))

    for i in range(len(G.nodes())):
        #diagonal terms
        term1 = -B
        
        if i in list(G.neighbors(i)): #selfloop!
            term2 = C/G.degree(i)
        else:
            term2 = 0
            
        J[i][i] = term1 #+ term2
    
        #off-diagonal terms
        for neighbor in list(G.neighbors(i)):
            J[i][neighbor] = C/G.degree(i)

    return J
'''