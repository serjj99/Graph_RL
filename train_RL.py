# import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
# import random
# from tqdm import tqdm
from RL_graphs_functions import grid_graph_create, remove_random_edges, remove_random_edges_keeping_connectivity, g, R, A, d_1
from RL_graphs_functions import ext_MCShane_Whitney_sup, ext_MCShane_Whitney_inf, ext_MCShane_Whitney_mean
from RL_graphs_functions import RL_graph_train, RL_graph_predict
from RL_graphs_functions import number_combinations
import config


cnf = config.config_2

# MODEL
model_name  = cnf.model_name
source = cnf.source
ext = cnf.ext
epochs = cnf.epochs
n_max_path = cnf.n_max_path
size_test = cnf.size_test
seed = cnf.seed
x0, xn = cnf.x0, cnf.xn
size_rm_edges = cnf.size_rm_edges

checkpoint = 50

print(
    f"model_name: {model_name}\n"
    f"source: {source}\n"
    f"ext: {ext}\n"
    f"epochs: {epochs}\n"
    f"n_max_path: {n_max_path}\n"
    f"size_test: {size_test}\n"
    f"seed: {seed}\n"
    f"x0, xn: {x0}, {xn}\n"
    f"size_rm_edges: {size_rm_edges}"
)


# GRAPH
grid_graph, positions = grid_graph_create(x0, xn)
grid_graph_modified = remove_random_edges_keeping_connectivity(grid_graph, size=size_rm_edges, seed=seed)


graph = grid_graph_modified



# TRAIN
K, Z = RL_graph_train(graph, positions, R, d_1, A, ext, source, epochs, n_max_path,
                       size_test, seed, name_save=f"MODEL_{model_name}", checkpoint=checkpoint)


# SAVE
VARIABLES = {
            "source": source,
            "ext": ext,
            "epochs": epochs,
            "n_max_path": n_max_path,
            "size_test": size_test,
            "seed": seed,
            "x0": x0,
            "xn": xn
            }

RESULT = [graph, positions, VARIABLES, K, Z]

with open(f"results/RESULTS_{model_name}.pkl", "wb") as file:
    pickle.dump(RESULT, file)

print('\n SAVED!')