import numpy as np
# import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
import pickle


########################################################################################################################################

def grid_graph_create(x0=-10, xn=10): # Grafo estilo malla, donde cada vértice está conectado por el que tiene arriba, abajo y a los lados
    x_range = np.arange(x0, xn + 1)
    y_range = np.arange(x0, xn + 1)
    
    G = nx.grid_2d_graph(len(x_range), len(y_range))
    
    coord_to_index = {}
    pos = {}
    index = 0
    for x in x_range:
        for y in y_range:
            coord_to_index[(x, y)] = index
            pos[index] = (x, y)
            index += 1
    
    mapping = {node: coord_to_index[(node[0] + x0, node[1] + x0)] for node in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    
    return G, pos


# Eliminar un 'size'% de las conexiones del grafo de forma aleatoria
def remove_random_edges(G, size=0.2, seed=0):
    random.seed(seed)
    
    edges = list(G.edges)
    num_edges_to_remove = int(len(edges) * size)
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    G.remove_edges_from(edges_to_remove)
    
    return G


# Eliminar un 'size'% de las conexiones del grafo de forma aleatoria manteniendo su conectividad, de forma que ningun vertice se queda sin conexión
def remove_random_edges_keeping_connectivity(G, size=0.2, seed=0):
    random.seed(seed)
    
    temp_graph = G.copy()
    edges = list(temp_graph.edges)
    num_edges_to_remove = int(len(edges) * size)
    removed_edges = []
    
    while len(removed_edges) < num_edges_to_remove:
        edge = random.choice(edges)
        temp_graph.remove_edge(*edge)
        
        if nx.is_connected(temp_graph):
            removed_edges.append(edge)
            edges.remove(edge)
        else:
            temp_graph.add_edge(*edge)
    
    return temp_graph


def f(x, y): # Función a minimizar: Mínimo global (2,-2) y local (7,7)
    return min(2 * (x - 2)**2 + (y + 2)**2, 20 + 3 * (x - 7)**2 + (y - 7)**2)


def g(v, pos): # Función a maximizar usando el vértice 'v': -f(x,y)
    x, y = pos[v]
    return -f(x, y)


def R(path, a, pos): # Recompensa
    n = len(path)+1
    reward = 0
    for i in range(1, n):
        reward += (2*i / (n*(n+1))) * g(path[i-1], pos)
    reward += (2 / (n+1)) * g(a, pos)
    
    return reward


########################################################################################################################################


def A(graph, path): #Acciones: vértice conectado al ultimo vértice del camino sin repetición
    adj = nx.adjacency_matrix(graph).toarray()
    vertex = path[-1]
    
    actions = [v for v in range(adj.shape[0])
               if adj[vertex, v] != 0 and v not in path] # Todos los vertices conectados al ultimo que no se encuentren en el camino
    
    # # Si se puede repetir el último vértice usar la siguiente función
    # actions.append(vertex)
    
    return actions


def A2(graph, path): #Acciones: vértice conectado al ultimo vértice del camino sin repetición
    adj = nx.adjacency_matrix(graph).toarray()
    vertex = path[-1]
    
    actions = [v for v in range(adj.shape[0])
               if adj[vertex, v] != 0 and v not in path] # Todos los vertices conectados al ultimo que no se encuentren en el camino
    
    # Si se puede repetir el último vértice usar la siguiente función
    actions.append(vertex)
    
    return actions


def path_extended(path, size): #Se repite el ultimo paso del camino hasta que consiga el tamaño 'size'
    if len(path) >= size:
        return path
    return path + [path[-1]] * (size - len(path))


def shortest_path_distance(graph, source, target): #Bellman-Ford algorithm para el cálcuo de la menor distancia entre vértices
    try:
        distance = nx.shortest_path_length(graph, source=source, target=target, weight=None)
        return distance
    except nx.NetworkXNoPath:
        return float('inf')


def d_H(graph, path_x, path_y, n_max): # Distancia de Hausdorff para caminos
    path_x_ext = path_extended(path_x, n_max)
    path_y_ext = path_extended(path_y, n_max)
    
    dist_H = 0
    for i in range(n_max):
        dist_H += (1 / (2**(i+1))) * shortest_path_distance(graph, path_x_ext[i], path_y_ext[i])
    return dist_H


# Distancia estado + acción: Suma de la distancia de causdorff para el camino (estado) + shortest_path_distance para la acción 
def d_1(graph, x, y, n_max): #d_H (p) para T(V) y shortest_path_distance (q) para A(V)
    return d_H(graph, x[0], y[0], n_max) + shortest_path_distance(graph, x[1], y[1])


########################################################################################################################################

def number_combinations(epochs, n_max):
    n = epochs * n_max
    n_comb = n*(n-1) / 2
    return int(n_comb)


# Calculo de la constante de Lipchitz dado un subconjunto 'rep' y una distáncia 'd'
def Lip(graph, SC, d, n_max): #p para T(V) y q para A(V)
    K = -1
    tuples = [(x, y) for i, x in enumerate(SC) for y in SC[i+1:]]

    for x, y in tqdm(tuples, desc="Processing pairs"):
        dist = d(graph, x, y, n_max)
        
        if dist == 0:
            raise ValueError("Sergi: Distance when use 'Lip' function cannot be 0.")
        K_calc = abs(x[2] - y[2]) / dist
        
        if K_calc > K:
            K = K_calc
            
    return K


# Calculo de la constante de Lipchitz al añadir 'n_add' nuevos elementos al subconjunto 'SC'. K es Lip del subconjunto sin los nuevos elementos
def Lip_add(graph, SC, n_add, K0, d, n_max): #p para T(V) y q para A(V)
    K = K0
    SC_add = SC[-n_add:]
    
    if n_add == 1:
        tuples = [(x, y) for x in SC_add for y in SC[:-n_add]]
    else:
        tuples = [(x, y) for x in SC_add for y in SC[:-n_add]] + [(x, y) for i, x in enumerate(SC_add) for y in SC_add[i+1:]]


    for x, y in tuples:
        dist = d(graph, x, y, n_max)
        if dist == 0:
            raise ValueError("Sergi: Distance when use 'Lip_add' function cannot be 0.")
        K_calc = abs(x[2] - y[2]) / dist
        
        if K_calc > K:
            K = K_calc
            
    return K


# Ext de MCShane Whitney del 'supremo' (fM) dado el subconjunto 'SC', la cns Lip(SC) = K, la distancia 'd', el estado 'e' y la acción 'a'
def ext_MCShane_Whitney_sup(graph, e, a, SC, d, n_max, K):   
    a_ = (e,a)
    fM = max([(b[2] - (K * d(graph, a_, b, n_max))) for b in SC])
    return fM


# Ext de MCShane Whitney del 'ínfimo' (fW) dado el subconjunto 'SC', la cns Lip(SC) = K, la distancia 'd', el estado 'e' y la acción 'a'
def ext_MCShane_Whitney_inf(graph, e, a, SC, d, n_max, K):   
    a_ = (e,a)
    fW = min([(b[2] + (K * d(graph, a_, b, n_max))) for b in SC])
    return fW


# Media aritmética de las Ext de MCShane Whitney dado el subconjunto 'SC', la cns Lip(SC) = K, la distancia 'd', el estado 'e' y la acción 'a'
def ext_MCShane_Whitney_mean(graph, e, a, SC, d, n_max, K):   
    fM = ext_MCShane_Whitney_sup(graph, e, a, SC, d, n_max, K)
    fW = ext_MCShane_Whitney_inf(graph, e, a, SC, d, n_max, K)
    return (fM + fW) / 2



########################################################################################################################################

def RL_graph_train(graph, pos, R, d, A, ext, source, epochs, n_max_path, size_test=0.2, seed=0, name_save='results/MODEL.pkl', checkpoint=50):
    nodes = list(graph.nodes())
    Z = []
    nZ = 0
    K = -1
    
    random.seed(seed)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        if source is None:
            path = [random.choice(nodes)]
        else:
            path = [source]
            
        actions = A(graph, path)
        i=0
        while i < n_max_path and actions != []:
            if (len(Z) < 3) or (random.random() < size_test):
                a = random.choice(actions)
                
            else:
                n_add = len(Z) - nZ
                if n_add == 0:
                    a = max(actions, key=lambda a:
                            ext(graph, path, a, Z, d, n_max_path, K))
                    
                else:
                    K = Lip_add(graph, Z, n_add, K, d, n_max_path)
                    a = max(actions, key=lambda a:
                            ext(graph, path, a, Z, d, n_max_path, K))
                
                    nZ = len(Z)
            
            reward = R(path, a, pos)
            
            Z_new = (list(path), a, reward)
            if Z_new not in Z:
                Z.append(Z_new)
            
            path.append(a)
            
            i += 1
            actions = A(graph, path)
            
        if (epoch+1) % checkpoint == 0:
            with open(f"results/progress/{name_save}_epoch_{int(epoch+1)}.pkl", "wb") as file:
                pickle.dump([K, Z], file)
                    
    return K, Z



def RL_graph_predict(graph, d, A, K, Z, ext, source, n_max_path):
    path = [source]
    actions = A(graph, path)

    with tqdm(total=n_max_path, desc="Predicting Path Progress") as pbar:
        i=0
        while i < n_max_path and actions != []:
            a = max(actions, key=lambda a:
                    ext(graph, path, a, Z, d, n_max_path, K))
                
            path.append(a)
            
            i += 1
            actions = A(graph, path)
            pbar.update(1)
    
    return path
