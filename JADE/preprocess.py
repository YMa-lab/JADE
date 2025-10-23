import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from tqdm import tqdm

def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    #sc.pp.filter_genes(adata, min_cells=1)
    #sc.pp.filter_genes(adata_sc, min_cells=1)
    
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def construct_interaction_KNN_minibatch(adata, n_neighbors=3, batch_size=5000):
    pos = adata.obsm["spatial"]
    n   = pos.shape[0]

    idx_model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(pos)

    rows, cols = [], []
    for s in tqdm(range(0, n, batch_size)):
        e        = min(s + batch_size, n)
        _, inds  = idx_model.kneighbors(pos[s:e])
        rows.append(np.repeat(np.arange(s, e)[:, None], n_neighbors, 1).ravel())
        cols.append(inds[:, 1:].ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    inter = coo_matrix((np.ones_like(rows, dtype=np.uint8), (rows, cols)), shape=(n, n)).tocsr()
    adj   = inter.maximum(inter.T)

    adata.obsm["graph_neigh"] = inter
    adata.obsm["adj"]         = adj
    
def construct_interaction_KNN(adata, n_neighbors=3):
    # print('construct neighbor using KNN')
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    # print(f"check row sum:{interaction.sum(1)}")
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    # print('Graph constructed!')   
    
def construct_interaction_radius(adata, radius=50.0):
    pos = adata.obsm["spatial"]          # (n_spot, 2)
    n   = pos.shape[0]

    nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(pos)
    neigh_idx = nbrs.radius_neighbors(pos, return_distance=False)

    interaction = np.zeros((n, n), dtype=np.uint8)
    for i, idx in enumerate(neigh_idx):
        idx = idx[idx != i]              # drop self‑index
        interaction[i, idx] = 1

    adata.obsm["graph_neigh"] = interaction

    adj = interaction + interaction.T    # make symmetric
    adj[adj > 1] = 1
    adata.obsm["adj"] = adj
    
    # --- average number of neighbors ---
    avg_nn = adj.sum(1).mean()          # row sums = neighbors per spot
    print(f"Average neighbors per spot: {avg_nn:.2f}")

def preprocess(adata, ngenes=3000):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=ngenes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # sc.pp.scale(adata, zero_center=True, max_value=10) # for merfish
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a   

def get_common_feature(adata, gene_index):
    feat =  adata[:, gene_index].X.toarray()[:, ]
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a   
    
from sklearn.decomposition import PCA
def get_feature_pca(srcadata, tgtadata, deconvolution=False, n_components=256):
    pca = PCA(n_components=n_components)  # Reduce to 2 dimensions
    feats = []
    for adata in [srcadata, tgtadata]:
        if deconvolution:
            adata_Vars = adata
        else:   
            adata_Vars =  adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ] 
        feat_pca = pca.fit_transform(feat)
        print(np.cumsum(pca.explained_variance_ratio_)[-1])
        feats.append(feat_pca)
    feats_permute = [permutation(feat) for feat in feats]
    srcadata.obsm['feat'] = feats[0]
    tgtadata.obsm['feat'] = feats[1]
    srcadata.obsm['feat_a'] = feats_permute[0]
    tgtadata.obsm['feat_a'] = feats_permute[1]
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float32)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(0))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)  

def preprocess_adj_decoder_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    adj_  = 2*sp.eye(adj.shape[0]) - adj_
    rowsum_= 2*np.ones(adj.shape[0]).reshape(-1, 1) + rowsum
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum_, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)  

def preprocess_adj_unnorm_sparse(adj):
    # only add self-loop
    adj = adj.T
    # print(f"check column sum: {adj.sum(0)}")
    adj = sp.coo_matrix(adj)
    return sparse_mx_to_torch_sparse_tensor(adj)

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
