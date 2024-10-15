from dadapy import Clustering 
import numpy as np
import torch
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score
import warnings


def to_numpy_float64(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy().astype(np.float64)
    elif isinstance(arr, np.ndarray):
        arr = arr.astype(np.float64)
    else:
        raise TypeError("Input must be either a PyTorch tensor or a NumPy array.")
    return arr

def get_dpa(data, Zpar, verbose=False):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        data = to_numpy_float64(data)
        dpa = Clustering(data, verbose=verbose)
        dpa.compute_clustering_ADP(Z = Zpar, halo=False, v2=True) 
     
    return dpa
    
    
def get_Zkiller(dpa):
    # compute the Z-killer value for each cluster
    # return ndarray of shape (N_clusters,)
    w = np.zeros(dpa.N_clusters)
    for i in range(dpa.N_clusters):
        ro_i = dpa.log_den[dpa.cluster_centers[i]]
        e_i = dpa.log_den_err[dpa.cluster_centers[i]]
        zs = []
        for j in range(dpa.N_clusters):
            if i == j:
                continue
            ro_ij = dpa.log_den_bord[i,j]
            e_ij = dpa.log_den_bord_err[i,j]
            if e_i - e_ij == 0:
                continue
            z_ij = (ro_i - ro_ij)/(e_i + e_ij)

            zs.append(np.abs(z_ij)) 

        w[i] = np.min(zs) if len(zs) > 0 else 0.0
    return w

def random_split(data, chunk_size):
    # data.shape: N x D
    assert isinstance(chunk_size, int)
    assert data.shape[0] > data.shape[1]
    N = data.size(0)
    rand_idxs = torch.randperm(N)
    chunks = torch.chunk(rand_idxs, chunk_size)
    
    if len(chunks) != chunk_size:
        warning_message = f'Warning: the number of chunks \'chunk_size\' have been changed by \'torch.chunk\'. Using {len(chunks)} instead.'
        warnings.warn(warning_message)
    
    splitted_data = []
    indexes = []

    for i in range(chunk_size):
        
        idxs = []
        for j in range(len(chunks)):
            if j != i:
                idxs.append(chunks[j])
        idxs = torch.cat(idxs)
        indexes.append(idxs)
        

        # data[idxs]: (N -( N//chunk_size + reminder)) x D   
        splitted_data.append(data[idxs]) 
    
    return splitted_data, indexes


def get_intersection(pair, dict):
    cluster1 = dict[pair[0]]
    cluster2 = dict[pair[1]]
    idxs1 = list(cluster1.keys())
    idxs2 = list(cluster2.keys())
   
    intersection = np.intersect1d(idxs1, idxs2)
  
    cluster1_inter = {k: v for k, v in cluster1.items() if k in intersection}
    cluster2_inter = {k: v for k, v in cluster2.items() if k in intersection}

    return np.array(list(cluster1_inter.values())), np.array(list(cluster2_inter.values()))
    
def from_loader_to_tensor(loader):
    all_data = []


    for BATCH in loader:
        if isinstance(BATCH, list):
            x, _ = BATCH
        else:
            x = BATCH

        x = x.view(x.size(0), -1)
        #print(x.shape)
        all_data.append(x)

    data = torch.cat(all_data, dim=0)
    data.requires_grad = False
    return data


def heuristic(data, Z, chunk_size=5):
    assert isinstance(chunk_size, int)
    assert isinstance(Z, float)
    assert data.size(0) > data.size(1)
    assert chunk_size > 2, "chunk_size must be greater than 2"
    assert chunk_size < data.size(0), "chunk_size must be smaller than the number of data points"

    splitted_data, indexes = random_split(data, chunk_size=chunk_size)
    clusters_assignments = [] # dictionary containing: index of the point in the original dataset -> cluster_assignment by DPA
    for i in range(chunk_size):
        dpa = get_dpa(splitted_data[i], Z, verbose=False)
        dict_assignments = {int(indexes[i][j]): dpa.cluster_assignment[j] for j in range(len(dpa.cluster_assignment))}
        clusters_assignments.append(dict_assignments)
        del dpa
        del dict_assignments

    combs = list(combinations([i for i in range(chunk_size)], 2))
    
    MIscores = []
    for pair in combs:
        ass1, ass2 = get_intersection(pair, clusters_assignments)
        MIscores.append(normalized_mutual_info_score(ass1, ass2))
    
    return np.mean(MIscores)
