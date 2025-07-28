from RL_graphs_functions import ext_MCShane_Whitney_sup, ext_MCShane_Whitney_inf, ext_MCShane_Whitney_mean


class config_1: # Slow +
    model_name  = 'model_1'
    ext = ext_MCShane_Whitney_mean
    source = None # Random
    epochs = 250
    n_max_path = 30
    size_test = 0.2
    seed = 0
    x0, xn = -10, 10
    size_rm_edges = 0.3 # Less connections

    
class config_2: # Slow
    model_name  = 'model_2'
    ext = ext_MCShane_Whitney_mean
    source = None # Random
    epochs = 250
    n_max_path = 25
    size_test = 0.2
    seed = 0
    x0, xn = -10, 10
    size_rm_edges = 0.25
    
    
class config_3:
    model_name  = 'model_3'
    ext = ext_MCShane_Whitney_mean
    source = 101 # (-6, 7)
    epochs = 100
    n_max_path = 25
    size_test = 0.2
    seed = 0
    x0, xn = -10, 10
    size_rm_edges = 0.25
    
    
class config_4:
    model_name  = 'model_4'
    ext = ext_MCShane_Whitney_mean
    source = 101 # (-6, 7)
    epochs = 30
    n_max_path = 5
    size_test = 0.25
    seed = 0
    x0, xn = -10, 10
    size_rm_edges = 0.25

    
class config_prueba:
    model_name  = 'prueba'
    ext = ext_MCShane_Whitney_mean
    source = 101 # (-6, 7)
    epochs = 100
    n_max_path = 10
    size_test = 0.2
    seed = 0
    x0, xn = -10, 10
    size_rm_edges = 0.3
    