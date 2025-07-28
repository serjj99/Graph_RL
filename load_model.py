# import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# import random
# from tqdm import tqdm
from RL_graphs_functions import grid_graph_create, remove_random_edges, remove_random_edges_keeping_connectivity, g, R, A, d_1
from RL_graphs_functions import ext_MCShane_Whitney_sup, ext_MCShane_Whitney_inf, ext_MCShane_Whitney_mean
from RL_graphs_functions import RL_graph_train, RL_graph_predict
from RL_graphs_functions import number_combinations
import pickle

# NAME MODEL
model_name = 'model_2'


# LOAD MODEL
with open(f"results/RESULTS_{model_name}.pkl", "rb") as file:
    RESULT = pickle.load(file)

graph, positions, VARIABLES, K, Z = RESULT

locals().update(VARIABLES)
ext = VARIABLES['ext']
n_max_path = VARIABLES['n_max_path']

# PREDICCIONES
source_pred = 101

path = RL_graph_predict(graph, d_1, A, K, Z, ext, source_pred, n_max_path)


# RESULTADOS

values = [g(pa, positions) for pa in path]

print('\n RESULTS: g(path)')
for pa in path:
    print(f"{positions[pa]}: Value_maximize = {g(pa, positions)}")
    

# Gráfica de los valores a maximizar
x = range(len(values))

plt.figure(figsize=(10, 5))
plt.plot(x, values, marker='o', zorder=1)
plt.scatter(x[0], values[0], color='red', s=100, zorder=2)
plt.scatter(x[-1], values[-1], color='green', s=100, zorder=2)
plt.xlabel("Step")
plt.ylabel("Value to maximize")
plt.title("Evolution of the value to be maximized")
plt.grid(True)
plt.savefig(f"figures/errors_{model_name}.png", dpi=300, bbox_inches='tight')
plt.show()


# Gráfica de el camino por el grafo mallado
plt.figure(figsize=(12, 12))
nx.draw_networkx_edges(graph, pos=positions, edge_color='black', alpha=0.5)  # Conexiones en negro
nx.draw_networkx_nodes(graph, pos=positions, node_size=15, node_color='black')  # Nodos en negro

path_edges = list(zip(path, path[1:]))
nx.draw_networkx_edges(graph, pos=positions, edgelist=path_edges, edge_color='blue', width=2)
nx.draw_networkx_nodes(graph, pos=positions, nodelist=path, node_size=50, node_color='blue', label='Path')
nx.draw_networkx_nodes(graph, pos=positions, nodelist=[path[0]], node_size=100, node_color='red', label='Source')
nx.draw_networkx_nodes(graph, pos=positions, nodelist=[path[-1]], node_size=100, node_color='green', label='Target')

nx.draw_networkx_nodes(graph, pos=positions, nodelist=[260], node_color='purple', node_size=200, node_shape='X', label='Global Max') # (2,-2) global max
nx.draw_networkx_nodes(graph, pos=positions, nodelist=[374], node_color='c', node_size=200, node_shape='X', label='Local Max') # (7, 7) local max

plt.gca().set_facecolor('white')
plt.axis('off')
# plt.title(f"Final PATH [{x0},{xn}] x [{x0},{xn}]", fontsize=25)
plt.title("Final PATH", fontsize=25)
plt.legend(scatterpoints=1, loc='best', fontsize=15)
plt.savefig(f"figures/path_graph_{model_name}.png", dpi=300, bbox_inches='tight')
plt.show()