import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import os
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess_adj_decoder_sparse, preprocess_adj_unnorm_sparse,  preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, get_common_feature, fix_seed
from .model import JADEAlignEncoder
 
class JADE():
    def __init__(self, adatalist, **kwargs):
        self.kwargs = kwargs
        self.if_norm_distort = self.kwargs.get('if_norm_distort', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.random_seed = self.kwargs.get('seed', 0)
        self.pretrain_epochs = self.kwargs.get('pretrain_epochs', 200)
        self.epochs = self.kwargs.get('epochs', 800)
        self.lr = self.kwargs.get('lr', 0.002)
        self.weight_decay = self.kwargs.get('weight_decay', 0.0001)
        self.pretrain_misalignment_weight = self.kwargs.get('pretrain_misalignment_weight',5.0)
        self.misalignment_weight = self.kwargs.get('misalignment_weight', 0.1)
        self.mismaintainness_weight = self.kwargs.get('mismaintainness_weight', 5.0)
        self.marginal_weight = self.kwargs.get('marginal_weight', 1.0)
        self.alpha = self.kwargs.pop('alpha', 10)
        self.beta = self.kwargs.pop('beta', 1)
        self.max_iter = self.kwargs.pop('max_iter', 3)
        self.n_clusters = self.kwargs.pop('n_clusters', 7)
        self.datatype = self.kwargs.pop('datatype', 'Slide')
        self.select_common_genes = self.kwargs.pop('select_common_genes', True)
        self.ngenes = self.kwargs.pop('ngenes', 3000)
        self.reduced_dim = self.kwargs.pop('reduced_dim', 20)
        # output parameter
        self.verbose = self.kwargs.pop('verbose', True)
        self.eval = self.kwargs.pop('eval', True)
        # En/Decoder dim
        self.domain_adaptor_hidden_dim = 64
        self.gcn_hidden_dim = 128
        self.gcn_latent_dim = 64
        # Seed everything
        fix_seed(self.random_seed)
        # AnnData
        self.adatalist = [adata.copy() for adata in adatalist]
        self.ns = [adata.shape[0] for adata in adatalist]
        # For FastJADE only, construct-hyper-spots
        # TODO: rename.
        self.spot_cell_assignment = [adata.uns['spot_cell_assignment'] for adata in adatalist]
        self.ms = [len(adata.uns['spot_cell_assignment']) for adata in adatalist]
        self.n_slices = len(self.ns)
        if self.if_norm_distort:
            normalizations = [np.min(adata.obsm['spatial'][adata.obsm['spatial']!=0]) for adata in adatalist]
            print(normalizations)
            embs = [torch.from_numpy(adata.obsm['spatial']/norm).float().to(self.device) for adata, norm in zip(adatalist, normalizations)]
        else:
            embs = [torch.from_numpy(adata.obsm['spatial']/100.0).float().to(self.device) for adata in adatalist]
        self.dist_mats = [torch.cdist(emb, emb) for emb in embs]
        self.embeddings = embs
        # preprocess AnnData
        for adata in self.adatalist:
            if 'highly_variable' not in adata.var.keys():
                preprocess(adata, self.ngenes)

        self.gene_space_dim = self.ngenes
        if self.select_common_genes:
            common_genes = None
            for adata in self.adatalist:
                if common_genes is None:
                    common_genes = set(adata.var['highly_variable'][adata.var['highly_variable']==True].index)
                else:
                    common_genes.intersection_update(set(adata.var['highly_variable'][adata.var['highly_variable']==True].index))
            common_genes = list(common_genes)
            common_genes = sorted(common_genes)
            self.gene_space_dim = len(common_genes)
            self.common_genes = common_genes
            print(f"ngenes:{self.gene_space_dim}")
        # Construct nn
        for adata in self.adatalist:
            if 'adj' not in adata.obsm.keys():
                construct_interaction_KNN(adata, kwargs.get('n_neighbors',3))   
        # Add contrastive learning label
        for adata in self.adatalist:
            if 'label_CSL' not in adata.obsm.keys():    
                add_contrastive_label(adata)
        # Extract feature for learning
        for adata in self.adatalist:
            if 'feat' not in adata.obsm.keys():
                if self.select_common_genes:
                    get_common_feature(adata, common_genes)
                else:
                    get_feature(adata)          
        self.features = [torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device) for adata in self.adatalist]
        self.features_a = [torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device) for adata in self.adatalist]
        # print(self.features_a)
        self.label_CSL = [torch.FloatTensor(adata.obsm['label_CSL']).to(self.device) for adata in self.adatalist]
        self.adj = [adata.obsm['adj'] for adata in self.adatalist]
        self.graph_neigh = [torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(self.ns[q])).to(self.device) for q, adata in enumerate(self.adatalist)]
        self.graph_neigh_numpy = [adata.obsm['graph_neigh'].copy() + np.eye(self.ns[q]) for q, adata in enumerate(self.adatalist)]
        # Building sparse matrix
        print('Building sparse matrix ...')
        tmp2 = self.adj
        self.adj = [preprocess_adj_sparse(adj).to(self.device) for adj in tmp2]
        # For pretrain misalignment loss (i.e., spot level feature)
        self.spot_features = []
        for k, feature in enumerate(self.features):
            tmp = torch.zeros((self.ms[k], self.gene_space_dim)).to(self.device)
            for i in range(self.ms[k]):
                tmp[i] = feature[self.spot_cell_assignment[k][i]].mean(dim=0)
            self.spot_features.append(tmp)
        

    def train(self, ):
        
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
        with tqdm(total=int(self.pretrain_epochs+self.epochs), 
                    desc="JADE training",
                        bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in tqdm(range(self.pretrain_epochs+self.epochs)): 
                self.model.train()
                self.optimizer.zero_grad()
                if epoch <= self.pretrain_epochs: 
                    self.pretrain = True
                else:
                    self.pretrain = False
                
                self.batch_results, self.alignments, self.loss_sl, self.loss_sl_sh, self.loss_feat, self.loss_domain, self.loss_align, self.loss_align_fix, self.loss_maintain, self.loss_marginal, self.loss_sparsity = self.model(
                    self.features,
                    self.spot_features,
                    self.adj,
                    self.graph_neigh,
                    self.label_CSL,
                    self.dist_mats,
                    if_norm_distort=self.if_norm_distort
                )
                
                loss =  self.alpha*self.loss_feat + self.beta*self.loss_sl
                
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
                pbar.update(1)
                
                ### print out losses
                if self.verbose and epoch % int((self.epochs+self.pretrain_epochs)/10) == 0:
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
            for adata in self.adatalist:
                clustering(adata, reduced_dim=self.reduced_dim, key="emb_rec", n_clusters=7, refinement=True, radius=25)
                
            # NOTE: return spot-level alignment
            # Post-processing alignment matrix
            pis = []
            for _ in self.model.Ms:
                tau = 1.0
                M = self.model.Ms[0].cpu().detach().numpy()
                srcemb = self.adatalist[0].obsm['emb_unnorm']
                tgtemb = self.adatalist[1].obsm['emb_unnorm']
                C = srcemb @ M @ M.T @ tgtemb.T / np.sqrt(M.shape[0]) / tau
                unnorm_alignment = softmax(C, axis=1) / srcemb.shape[0]
                pi = unnorm_alignment
                # multiple row-col normalizations 
                pi = multiple_row_col_renormalizations(pi, niter=8)
                pis.append(pi)

            return self.adatalist, self.alignments, pis