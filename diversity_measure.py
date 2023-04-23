import itertools
import numpy as np
import numpy.typing as npt

NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
from scipy.spatial.distance import jensenshannon as JSD
from scipy.sparse.csgraph import shortest_path

"""Calculates a graph node distribution

@type graph: NDArrayInt
@param graph: the graph whose node distribution will be calculated

@rtype: npt.NDArray
@returns: the node distribution of the graph
"""
def node_distribution(graph: NDArrayInt) -> npt.NDArray:
    dist: NDArrayInt = shortest_path(graph, directed=False, unweighted=True).astype(np.int_)
    dist[dist < 0] = dist.shape[0]
    N = dist.max() + 1
    dist_offsets = dist + np.arange(dist.shape[0])[:, None] * N
    return np.delete(np.bincount(dist_offsets.ravel(), minlength=dist.shape[0] * N).reshape(-1, N) / (dist.shape[0] - 1), 0, axis=1)

"""Calculates a graph transition matrix

@type graph: NDArrayInt
@param graph: the graph whose transisiton matrix will be calculated

@rtype: npt.NDArray
@returns: the transition matrix of the graph in the same shape of the graph
"""
def transition_matrix(graph: NDArrayInt) -> npt.NDArray:   
    transition_matrix: NDArrayFloat = graph/np.sum(graph, axis=0)[:, None]
    transition_matrix[np.isnan(transition_matrix)] = 0
    return transition_matrix

"""Calculates the layer difference between two layers/graphs

@type node_dist_G: NDArrayInt
@param node_dist_G: the graph whose transisiton matrix will be calculated

@type trans_mat_G: NDArrayInt
@param trans_mat_G: the graph whose transisiton matrix will be calculated

@type node_dist_H: NDArrayInt
@param node_dist_H: the graph whose transisiton matrix will be calculated

@type trans_mat_H: NDArrayInt
@param trans_mat_H: the graph whose transisiton matrix will be calculated

@rtype: np.float_
@returns: the float value between 0 and 1 that represents the difference between the layers/graphs
"""
def layer_difference(node_dist_G, trans_mat_G, node_dist_H, trans_mat_H) -> np.float_:

    node_dist_G = np.pad(node_dist_G, [(0, 0), (0, node_dist_G.shape[0] - node_dist_G.shape[1])])
    node_dist_H = np.pad(node_dist_H, [(0, 0), (0, node_dist_H.shape[0] - node_dist_H.shape[1])])
    
    node_distribution_diff: NDArrayFloat = JSD(node_dist_G, node_dist_H, axis = 1)
    transition_matrix_diff: NDArrayFloat = JSD(trans_mat_G, trans_mat_H, axis = 1) 

    node_distribution_diff[np.isnan(node_distribution_diff)] = 0
    transition_matrix_diff[np.isnan(transition_matrix_diff)] = 0

    node_difference: NDArrayFloat = (node_distribution_diff + transition_matrix_diff) / 2
    return np.around(np.average(node_difference), decimals=4)


"""Calculates the less contribute list of the layer/graph network

@type node_dist_G: npt.NDArray
@param node_dist_G: the list of node distributions for each layer/graph

@type node_dist_G: npt.NDArray
@param node_dist_G: the list of transition matrices for each layer/graph

@rtype: NDArrayInt
@returns: the less contribute list
"""
def less_contribute(node_distributions: npt.NDArray, trasition_matrices: npt.NDArray) -> NDArrayInt:

    number_of_layers: np.int_ = len(trasition_matrices)
    combinations: list[tuple[int]] = list(itertools.combinations(np.arange(number_of_layers), 2))
    layer_difference_matrix = np.zeros(shape=((number_of_layers, number_of_layers)), dtype=float)
    
    for pair in combinations:
        i , j = pair[0], pair[1]
        ld = layer_difference(
            node_dist_G = node_distributions[i],
            node_dist_H = node_distributions[j],
            trans_mat_G = trasition_matrices[i],
            trans_mat_H = trasition_matrices[j]
        )

        layer_difference_matrix[i][j] = layer_difference_matrix[j][i] = ld  

    div: np.float_ = 0

    np.fill_diagonal(layer_difference_matrix, 1)

    list_less_contribute = list()
    candidatos = list(range(0,layer_difference_matrix.shape[0]))

    for _ in range(layer_difference_matrix.shape[0]-1):    
     
        layer_a, layer_b = np.unravel_index(layer_difference_matrix.argmin(), layer_difference_matrix.shape)

        smallest_layer_difference: np.float_ = layer_difference_matrix[layer_a, layer_b]

        div += smallest_layer_difference

        dist_a_to_set = np.amin(layer_difference_matrix[layer_a], where = layer_difference_matrix[layer_a] != smallest_layer_difference, initial = np.inf)
        dist_b_to_set = np.amin(layer_difference_matrix[layer_b], where = layer_difference_matrix[layer_b] != smallest_layer_difference, initial = np.inf)

        less_contribute = layer_a if dist_a_to_set <= dist_b_to_set else layer_b
        
        list_less_contribute.append(less_contribute)

        layer_difference_matrix[less_contribute,:] = np.inf
        layer_difference_matrix[:,less_contribute] = np.inf
    
    list_less_contribute.append(np.where(np.isin(candidatos, list_less_contribute) == False)[0][0])

    return np.array(list_less_contribute)
