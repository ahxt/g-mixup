# some code are from https://github.com/HongtengXu/SGWB-Graphon

import copy
import cv2
import numpy as np
import torch

from skimage.restoration import denoise_tv_chambolle
from typing import List, Tuple


def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()


def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending


        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree
            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
            # normalized_node_degrees.append(normalized_node_degree)
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num


def align_graphs_centrality(graphs: List[np.ndarray],
                 padding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree
            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
            # normalized_node_degrees.append(normalized_node_degree)
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num




def estimate_target_distribution(probs: List[np.ndarray], dim_t: int = None) -> np.ndarray:
    """
    Estimate target distribution via the average of sorted source probabilities
    Args:
        probs: a list of node distributions [(n_s, 1) the distribution of source nodes]
        dim_t: the dimension of target distribution
    Returns:
        p_t: (dim_t, 1) vector representing a distribution
    """
    if dim_t is None:
        dim_t = min([probs[i].shape[0] for i in range(len(probs))])

    p_t = np.zeros((dim_t, 1))
    x_t = np.linspace(0, 1, p_t.shape[0])
    for i in range(len(probs)):
        p_s = probs[i][:, 0]
        p_s = np.sort(p_s)[::-1]
        x_s = np.linspace(0, 1, p_s.shape[0])
        p_t_i = np.interp(x_t, x_s, p_s) + 1e-3
        p_t[:, 0] += p_t_i

    p_t /= np.sum(p_t)
    return p_t


def estimate_graphon(graphs: List[np.ndarray], method, args) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'GWB' or method == 'SGWB' or method == 'FGWB' or method == 'SFGWB':
        aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=False)
    else:
        aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True)

    block_size = int(np.log2(max_num) + 1)
    num_blocks = int(max_num / block_size)
    p_b = estimate_target_distribution(normalized_node_degrees, dim_t=num_blocks)


    stepfunc = universal_svd(aligned_graphs, threshold=args.threshold_usvt)

    graphon = cv2.resize(stepfunc, dsize=(args.r, args.r), interpolation=cv2.INTER_LINEAR)
    return stepfunc, graphon







def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon





def guess_rank(matrix: torch.Tensor) -> int:
    """
    A function to guess the rank of a matrix
    :param matrix: a torch.Tensor matrix
    :return:
    """
    n = matrix.size(0)
    m = matrix.size(1)
    epsilon = torch.sum(matrix != 0) / ((n * m) ** 0.5)

    u, s, v = torch.svd(matrix, compute_uv=False)
    max_num = min([100, s.size(0)])
    s = s[:max_num]
    s, _ = torch.sort(s, descending=True)
    diff_s1 = s[:-1] - s[1:]
    diff_s1 = diff_s1 / torch.mean(diff_s1[-10:])
    r1 = torch.zeros(1)
    gamma = 0.05
    while r1.item() <= 0:
        cost = torch.zeros(diff_s1.size(0))
        for i in range(diff_s1.size(0)):
            cost[i] = gamma * torch.max(diff_s1[i:]) + i + 1

        idx = torch.argmin(cost)
        r1 = torch.argmax(idx)
        gamma += 0.05

    cost = torch.zeros(diff_s1.size(0))
    for i in range(diff_s1.size(0)):
        cost[i] = s[i + 1] + ((i + 1) * epsilon ** 0.5) * s[0] / epsilon

    idx = torch.argmin(cost)
    r2 = torch.max(idx)
    return max([r1.item(), r2.item()])



def averaging_graphs(aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph


def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float = 1e-10,
                prior: np.ndarray = None) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans


def node_cost_st(cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st


def gw_cost(cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)
