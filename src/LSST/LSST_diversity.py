import polars as pl
import pandas as pd
import numpy as np
import scipy
import ts2vg
import matplotlib.pyplot as plt
import h5sparse
from multiprocessing import Pool

from diversity_measure import *

data: np.array = scipy.io.arff.loadarff("../../data/LSST/LSST_TRAIN.arff")
data = data[0]

def run_multiprocessing(func, iterable, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(func, iterable)
    
def calculate_node_distribution(graph: np.array) -> np.array:
    return node_distribution(graph)

def calculate_transition_matrix(graph: np.array) -> np.array:    
    return transition_matrix(graph)

for i in range(len(data)):
    node_distributions: list = []
    transition_matrices: list = []
    visibility_graphs: list = []
    time_series_set, label = data[i]
    print(f"Calculando grafo do exemplo {i}")
    for j in range(len(time_series_set)):
        graph: np.array = ts2vg.NaturalVG().build(time_series_set[j].tolist()).adjacency_matrix()
        visibility_graphs.append(graph)

    node_distributions = run_multiprocessing(calculate_node_distribution, visibility_graphs, n_processors=8)
    transition_matrices = run_multiprocessing(calculate_transition_matrix, visibility_graphs, n_processors=8)

    sparse_visibility_graphs = np.array([scipy.sparse.csr_matrix(graph, dtype=np.int8) for graph in visibility_graphs])
    less_contribute_rank: np.array = less_contribute(node_distributions, transition_matrices)
    k = 0
    with h5sparse.File("LSST.hdf5", "a") as f:
        for graph in sparse_visibility_graphs:
            f.create_dataset(f"visibility_graphs/{i}/{k}", data=graph)
            k += 1
        f.create_dataset(f"diversity_rank/{i}", data=less_contribute_rank)

