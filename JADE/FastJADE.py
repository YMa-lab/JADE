import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import os
import anndata as ad
from .preprocess import *
from .model_fast import JADEAlignEncoder
import warnings
warnings.filterwarnings(
    "ignore",
    message=(
        r"reduction: 'mean' divides the total loss by both the batch size "
        r"and the support size.*"
    ),
    category=UserWarning
)
import numpy as np, torch
from sklearn.neighbors import NearestNeighbors  # ← explicit import

def knn_to_csr(pos: np.ndarray, k: int) -> torch.Tensor:
    # fit K-NN (k+1 because the first neighbour is the point itself)
    nbrs_idx = NearestNeighbors(n_neighbors=k + 1) \
               .fit(pos) \
               .kneighbors(return_distance=False)          # shape (n, k+1)

    n  = pos.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = nbrs_idx[:, 1:].ravel().astype(np.int64)        # skip self-edge
    data = np.ones_like(rows, dtype=np.float32)

    # build CSR indptr
    indptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)

    return torch.sparse_csr_tensor(
        torch.from_numpy(indptr),
        torch.from_numpy(cols),
        torch.from_numpy(data),
        size=(n, n)
    )

 
class FastJADE():
    def __init__(self, processed_data, **kwargs):
        """
        Initialize FastJADE with preprocessed data.
        
        Args:
            processed_data (dict): Dictionary containing all preprocessed data with keys:
                - 'adatalist': List of AnnData objects
                - 'features': List of feature tensors
                - 'features_a': List of augmented feature tensors
                - 'label_CSL': List of contrastive learning labels
                - 'adj': List of adjacency matrices (sparse)
                - 'graph_neigh': List of graph neighbor tensors
                - 'graph_neigh_numpy': List of graph neighbor numpy arrays
                - 'spot_features': List of spot-level features
                - 'embeddings': List of spatial embeddings
                - 'dist_mats': List of distance matrices
                - 'spot_cell_assignment': List of spot-cell assignments
                - 'ns': List of number of cells per slice
                - 'ms': List of number of spots per slice
                - 'gene_space_dim': Dimension of gene space
                - 'common_genes': List of common genes (if applicable)
        """
        self.kwargs = kwargs
        self.if_norm_distort = self.kwargs.get('if_norm_distort', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = self.kwargs.get('seed', 0)
        self.pretrain_epochs = self.kwargs.get('pretrain_epochs', 200)
        self.epochs = self.kwargs.get('epochs', 800)
        self.lr = self.kwargs.get('lr', 0.002)
        self.weight_decay = self.kwargs.get('weight_decay', 0.0001)
        self.pretrain_misalignment_weight = self.kwargs.get('pretrain_misalignment_weight', 5.0)
        self.misalignment_weight = self.kwargs.get('misalignment_weight', 0.1)
        self.mismaintainness_weight = self.kwargs.get('mismaintainness_weight', 5.0)
        self.marginal_weight = self.kwargs.get('marginal_weight', 1.0)
        self.alpha = self.kwargs.pop('alpha', 10)
        self.beta = self.kwargs.pop('beta', 1)
        self.max_iter = self.kwargs.pop('max_iter', 3)
        self.n_clusters = self.kwargs.pop('n_clusters', 7)
        self.reduced_dim = self.kwargs.pop('reduced_dim', 30)
        
        # Output parameters
        self.verbose = self.kwargs.pop('verbose', True)
        self.eval = self.kwargs.pop('eval', True)
        
        # En/Decoder dim
        self.domain_adaptor_hidden_dim = 64
        self.gcn_hidden_dim = 128
        self.gcn_latent_dim = 64
        
        # Seed everything
        fix_seed(self.random_seed)
        torch.use_deterministic_algorithms(True)
        
        # Load preprocessed data
        self.adatalist = processed_data['adatalist']
        self.features = processed_data['features']
        self.features_a = processed_data['features_a']
        self.label_CSL = processed_data['label_CSL']
        self.adj = processed_data['adj']
        self.graph_neigh = processed_data['graph_neigh']
        self.graph_neigh_numpy = processed_data['graph_neigh_numpy']
        self.spot_features = processed_data['spot_features']
        self.embeddings = processed_data['embeddings']
        self.dist_mats = processed_data['dist_mats']
        self.spot_cell_assignment = processed_data['spot_cell_assignment']
        self.ns = processed_data['ns']
        self.ms = processed_data['ms']
        self.gene_space_dim = processed_data['gene_space_dim']
        
        if 'common_genes' in processed_data:
            self.common_genes = processed_data['common_genes']
        
        self.n_slices = len(self.ns)
        
    def train(self):
        
        # self.graph_neigh_sparse  = [gn.to_sparse_csr() for gn in self.graph_neigh]       # list of CSR tensors
        self.graph_neigh_sparse = [knn_to_csr(a.obsm["spatial"], self.kwargs.get('n_neighbors', 3))
                           for a in self.adatalist] 
        self.graph_row_sums      = [torch.ones(gn.shape[0], 1, device=gn.device)         # shape (m,1)
                            for gn in self.graph_neigh_sparse]
        # Training method remains the same
        self.model = JADEAlignEncoder(
            self.gene_space_dim,
            self.gcn_latent_dim,
            self.n_slices,
            max_iter=self.max_iter,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr},
        ], weight_decay=self.weight_decay)
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        
        ### pre-train/train optimizer.
        print('Begin to train ST data...')
        self.model.train()
        self.losses = []
        
        self.plot_num = 0
        for epoch in tqdm(range(self.pretrain_epochs+self.epochs), 
                desc="JADE training"): 
            self.model.train()
            self.optimizer.zero_grad()
            if epoch <= self.pretrain_epochs: 
                self.pretrain = True
            else:
                self.pretrain = False
            
            self.batch_results, self.alignments, self.loss_sl, self.loss_sl_sh, \
            self.loss_feat, self.loss_domain, self.loss_align, self.loss_align_fix, \
            self.loss_maintain, self.loss_marginal, self.loss_sparsity = self.model(
                    self.features,
                    self.spot_features,
                    self.adj,
                    self.graph_neigh_sparse,   # ← sparse mask
                    self.graph_row_sums,       # ← pre-computed (m × 1) tensors
                    self.label_CSL,
                    self.dist_mats,
                    self.spot_cell_assignment,
                    if_norm_distort=self.if_norm_distort
            )
            
            loss =  self.alpha*self.loss_feat + self.beta*self.loss_sl + self.beta*self.loss_sl_sh
            # loss =  self.alpha*self.loss_feat + self.beta*self.loss_sl
            
            if self.pretrain:
                loss += self.pretrain_misalignment_weight * self.loss_align_fix
                loss += self.mismaintainness_weight * self.loss_maintain
                loss += self.marginal_weight * self.loss_marginal
            else:
                loss += self.misalignment_weight * self.loss_align
                loss += self.mismaintainness_weight * self.loss_maintain
                loss += self.marginal_weight * self.loss_marginal
            
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            
            ### print out losses
            if self.verbose and epoch % int((self.epochs+self.pretrain_epochs)/10) == 0 and epoch!=0:
                print(f'sl:{self.loss_sl}, sl_sh:{self.loss_sl_sh}, recon:{self.loss_feat}, domain:{self.loss_domain}, align:{self.loss_align}, main:{self.loss_maintain}, mar:{self.loss_marginal}, spa:{self.loss_sparsity}')
                self.plot_num += 1
            
            ### append loss to viz
            self.losses.append([
                self.loss_sl.cpu().detach().numpy(), 
                self.loss_feat.cpu().detach().numpy(), 
                self.loss_align.cpu().detach().numpy(),
                self.loss_maintain.cpu().detach().numpy(),
                self.loss_marginal.cpu().detach().numpy(),
                loss.cpu().detach().numpy()/2
                ])
                    
        print("Optimization finished for ST data!")
        
        self.model.eval()
        with torch.no_grad():
            for k in range(self.n_slices):
                emb_rec = self.batch_results[k]["out"]
                emb_rec = F.normalize(emb_rec, p=2, dim=1).detach().cpu().numpy()
                self.adatalist[k].obsm['emb_rec'] = emb_rec
                emb = self.batch_results[k]["h"]
                emb = F.normalize(emb, p=2, dim=1).detach().cpu().numpy() 
                self.adatalist[k].obsm['emb'] = emb
                self.adatalist[k].obsm['emb_unnorm'] = self.batch_results[k]["h"].cpu().detach().numpy()
                
            # NOTE: Downstream clustering task
            from scipy.special import softmax
            from JADE.utils import clustering, multiple_row_col_renormalizations
            # for adata in self.adatalist:
            #     clustering(adata, reduced_dim=self.reduced_dim, key="emb_rec", n_clusters=7, refinement=True, radius=25)

            return self.adatalist


def preprocess_jade_data(adatalist, **kwargs):
    """
    Preprocess data for FastJADE model.
    
    Args:
        adatalist: List of AnnData objects
        **kwargs: Additional parameters for preprocessing
            verbose (bool): Whether to print detailed progress information (default: True)
        
    Returns:
        dict: Dictionary containing all preprocessed data needed for FastJADE
    """
    # Extract verbose parameter
    verbose = kwargs.get('verbose', True)
    
    if verbose:
        print("=" * 60)
        print("Starting FastJADE Data Preprocessing")
        print("=" * 60)
    
    # Extract parameters
    if_norm_distort = kwargs.get('if_norm_distort', False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    select_common_genes = kwargs.get('select_common_genes', True)
    ngenes = kwargs.get('ngenes', 3000)
    n_neighbors = kwargs.get('n_neighbors', 3)
    
    if verbose:
        print(f"Device: {device}")
        print(f"Number of datasets: {len(adatalist)}")
        print(f"Normalization distortion: {if_norm_distort}")
        print(f"Select common genes: {select_common_genes}")
        print(f"Target HVG count: {ngenes}")
        print(f"Spatial neighbors: {n_neighbors}")
    
    # Copy data and print dataset info
    if verbose:
        print("\n" + "-" * 40)
        print("Dataset Information:")
        print("-" * 40)
    
    adatalist = [adata.copy() for adata in adatalist]
    ns = [adata.shape[0] for adata in adatalist]
    
    if verbose:
        for i, adata in enumerate(adatalist):
            print(f"Dataset {i}: {adata.shape[0]} cells × {adata.shape[1]} genes")
            if 'spatial' in adata.obsm:
                spatial_range = np.ptp(adata.obsm['spatial'], axis=0)
                print(f"  Spatial range: x={spatial_range[0]:.1f}, y={spatial_range[1]:.1f}")
    
    # Extract spot-cell assignments and spatial info
    spot_cell_assignment = [adata.uns['spot_cell_assignment'] for adata in adatalist]
    ms = [len(adata.uns['spot_cell_assignment']) for adata in adatalist]
    
    if verbose:
        print(f"\nSpot information:")
        for i, m in enumerate(ms):
            print(f"Dataset {i}: {m} spots")
    
    # Process spatial embeddings
    if verbose:
        print("\n" + "-" * 40)
        print("Processing Spatial Coordinates:")
        print("-" * 40)
    
    if if_norm_distort:
        # Normalize by range for each axis (x, y) separately
        normalizations = []
        for i, adata in enumerate(adatalist):
            spatial_coords = adata.obsm['spatial']
            x_range = np.ptp(spatial_coords[:, 0])  # peak-to-peak range for x
            y_range = np.ptp(spatial_coords[:, 1])  # peak-to-peak range for y
            normalizations.append([x_range, y_range])
            if verbose:
                print(f"Dataset {i} normalization ranges: x={x_range:.1f}, y={y_range:.1f}")
        
        embs = []
        for adata, norm in zip(adatalist, normalizations):
            spot_spatial = adata.uns['spot_spatial'].copy()
            # Normalize x and y coordinates by their respective ranges
            spot_spatial[:, 0] = spot_spatial[:, 0] / norm[0] if norm[0] > 0 else spot_spatial[:, 0]
            spot_spatial[:, 1] = spot_spatial[:, 1] / norm[1] if norm[1] > 0 else spot_spatial[:, 1]
            embs.append(torch.from_numpy(spot_spatial).float().to(device))
    
    else:
        if verbose:
            print("Using standard normalization (divide by 100)")
        embs = [torch.from_numpy(adata.uns['spot_spatial']/100.0).float().to(device) for adata in adatalist]
    
    dist_mats = [torch.cdist(emb, emb) for emb in embs]
    embeddings = embs
    
    # Preprocess AnnData objects
    if verbose:
        print("\n" + "-" * 40)
        print("Basic Preprocessing:")
        print("-" * 40)
    
    # # Prepross as in SLAT (using dual PCA)
    # adata_all = adatalist[0].concatenate(adatalist[1], join="inner")
    # sc.pp.highly_variable_genes(adata_all, n_top_genes=12000, flavor="seurat_v3")
    # adata_all = adata_all[:, adata_all.var.highly_variable]
    # sc.pp.normalize_total(adata_all)
    # sc.pp.log1p(adata_all)
    # adata_1 = adata_all[adata_all.obs["batch"] == "0"]
    # adata_2 = adata_all[adata_all.obs["batch"] == "1"]
    # sc.pp.scale(adata_1)
    # sc.pp.scale(adata_2)
    
    # from sklearn.utils.extmath import randomized_svd
    # X = adata_1.X
    # Y = adata_2.X
    # cor_var = X @ Y.T
    # cor_var = cor_var
    # U, S, Vh = randomized_svd(cor_var, n_components=64, random_state=0)
    # Z_x = U @ np.sqrt(np.diag(S))
    # Z_y = Vh.T @ np.sqrt(np.diag(S))
    # adatalist[0].obsm['feat'] = Z_x
    # adatalist[1].obsm['feat'] = Z_y
    # gene_space_dim = 64
    
    
    for i, adata in enumerate(adatalist):
        if 'highly_variable' not in adata.var.keys():
            if verbose:
                print(f"Dataset {i}: Running HVG selection...")
            preprocess(adata, ngenes)
        else:
            if verbose:
                print(f"Dataset {i}: HVG already computed")
            
        if verbose:
            hvg_count = adata.var['highly_variable'].sum()
            print(f"Dataset {i}: {hvg_count} highly variable genes")
    
    gene_space_dim = ngenes
    common_genes = None
    
    # Select common genes using concatenation approach
    if select_common_genes:
        if verbose:
            print("\n" + "-" * 40)
            print("Common Gene Selection (using concatenation):")
            print("-" * 40)
        
        # Create batch labels
        batch_labels = []
        for i, adata in enumerate(adatalist):
            batch_labels.extend([f'dataset_{i}'] * adata.shape[0])
        
        # Concatenate datasets
        if verbose:
            print("Concatenating datasets...")
        adata_concat = ad.concat(adatalist, axis=0, join='outer')
        if verbose:
            print(f"Concatenated shape: {adata_concat.shape[0]} cells × {adata_concat.shape[1]} genes")
        
        # Get HVG info from individual datasets
        hvg_per_dataset = {}
        for i, adata in enumerate(adatalist):
            hvg_genes = adata.var['highly_variable'][adata.var['highly_variable']==True].index.tolist()
            hvg_per_dataset[f'dataset_{i}'] = set(hvg_genes)
            if verbose:
                print(f"Dataset {i}: {len(hvg_genes)} HVGs")
        
        # Find intersection of HVGs across all datasets
        common_genes_set = hvg_per_dataset[f'dataset_0']
        for i in range(1, len(adatalist)):
            common_genes_set = common_genes_set.intersection(hvg_per_dataset[f'dataset_{i}'])
        
        common_genes = sorted(list(common_genes_set))
        gene_space_dim = len(common_genes)
        
        if verbose:
            print(f"\nCommon Gene Analysis:")
            print(f"  Original HVG counts: {[len(hvg_per_dataset[f'dataset_{i}']) for i in range(len(adatalist))]}")
            print(f"  Common genes across all datasets: {gene_space_dim}")
            print(f"  Retention rate: {gene_space_dim/ngenes*100:.1f}%")
        
        # Update concatenated adata with common gene info
        adata_concat.var['common_hvg'] = adata_concat.var_names.isin(common_genes)
        
        # Store common gene info back in individual datasets
        for adata in adatalist:
            adata.var['common_hvg'] = adata.var_names.isin(common_genes)
    
    # Construct nearest neighbor graphs
    if verbose:
        print("\n" + "-" * 40)
        print("Constructing Spatial Graphs:")
        print("-" * 40)
    
    for i, adata in enumerate(adatalist):
        if 'adj' not in adata.obsm.keys():
            if verbose:
                print(f"Dataset {i}: Building KNN graph with {n_neighbors} neighbors...")
            construct_interaction_KNN_minibatch(adata, n_neighbors)
            
            # Print graph statistics
            if verbose:
                adj_matrix = adata.obsm['adj']
                n_edges = (adj_matrix > 0).sum() // 2  # Divide by 2 for undirected graph
                avg_degree = n_edges * 2 / adata.shape[0]
                print(f"Dataset {i}: {n_edges} edges, {avg_degree:.1f} avg degree")
        else:
            if verbose:
                print(f"Dataset {i}: Spatial graph already computed")
    
    # Add contrastive learning labels
    if verbose:
        print("\n" + "-" * 40)
        print("Adding Contrastive Learning Labels:")
        print("-" * 40)
    
    for i, adata in enumerate(adatalist):
        if 'label_CSL' not in adata.obsm.keys():
            if verbose:
                print(f"Dataset {i}: Computing contrastive labels...")
            add_contrastive_label(adata)
        else:
            if verbose:
                print(f"Dataset {i}: Contrastive labels already computed")
    
    # Extract features
    if verbose:
        print("\n" + "-" * 40)
        print("Extracting Features:")
        print("-" * 40)
    
    for i, adata in enumerate(adatalist):
        if 'feat' not in adata.obsm.keys():
            if select_common_genes and common_genes is not None:
                if verbose:
                    print(f"Dataset {i}: Extracting features for {len(common_genes)} common genes...")
                get_common_feature(adata, common_genes)
            else:
                if verbose:
                    print(f"Dataset {i}: Extracting features for all HVGs...")
                get_feature(adata)
        else:
            if verbose:
                print(f"Dataset {i}: Features already computed")
        
        if verbose:
            feat_shape = adata.obsm['feat'].shape
            print(f"Dataset {i}: Feature shape {feat_shape}")
    
    # Convert to tensors
    if verbose:
        print("\n" + "-" * 40)
        print("Converting to Tensors:")
        print("-" * 40)
    
    features = [torch.FloatTensor(adata.obsm['feat'].copy()).to(device) for adata in adatalist]
    features_a = [torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device) for adata in adatalist]
    label_CSL = [torch.FloatTensor(adata.obsm['label_CSL']).to(device) for adata in adatalist]
    adj_raw = [adata.obsm['adj'] for adata in adatalist]
    graph_neigh = [torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(ns[q])).to(device) for q, adata in enumerate(adatalist)]
    graph_neigh_numpy = [adata.obsm['graph_neigh'].copy() + np.eye(ns[q]) for q, adata in enumerate(adatalist)]
    
    if verbose:
        print(f"Converted {len(features)} feature tensors to {device}")
    
    # Process adjacency matrices to sparse format
    if verbose:
        print("Converting adjacency matrices to sparse format...")
    adj = [preprocess_adj_sparse(adj).to(device) for adj in adj_raw]
    
    # Create spot-level features
    if verbose:
        print("\n" + "-" * 40)
        print("Creating Spot-level Features:")
        print("-" * 40)
    
    spot_features = []
    for k, feature in enumerate(features):
        tmp = torch.zeros((ms[k], gene_space_dim)).to(device)
        for i in range(ms[k]):
            tmp[i] = feature[spot_cell_assignment[k][i]].mean(dim=0)
        spot_features.append(tmp)
        if verbose:
            print(f"Dataset {k}: Spot features shape {tmp.shape}")
    
    # Package all processed data
    processed_data = {
        'adatalist': adatalist,
        'features': features,
        'features_a': features_a,
        'label_CSL': label_CSL,
        'adj': adj,
        'graph_neigh': graph_neigh,
        'graph_neigh_numpy': graph_neigh_numpy,
        'spot_features': spot_features,
        'embeddings': embeddings,
        'dist_mats': dist_mats,
        'spot_cell_assignment': spot_cell_assignment,
        'ns': ns,
        'ms': ms,
        'gene_space_dim': gene_space_dim,
    }
    
    if common_genes is not None:
        processed_data['common_genes'] = common_genes
    
    # Print final summary
    if verbose:
        print("\n" + "=" * 60)
        print("Preprocessing Summary:")
        print("=" * 60)
        print(f"Final gene space dimension: {gene_space_dim}")
        print(f"Total cells: {sum(ns)}")
        print(f"Total spots: {sum(ms)}")
        print(f"Device: {device}")
        if common_genes is not None:
            print(f"Common genes selected: {len(common_genes)}")
            print(f"Gene retention rate: {len(common_genes)/ngenes*100:.1f}%")
        print("Preprocessing completed successfully!")
        print("=" * 60)
    
    return processed_data
