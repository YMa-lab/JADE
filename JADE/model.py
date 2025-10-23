import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from . preprocess import permutation
from . sinkhorn import NonSquareSinkhornKnopp


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # self.conv = GCNConv(in_channels, out_channels)
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.prelu = nn.PReLU()
        self.kwargs = kwargs
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x = self.conv(x, adj)
        # z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(x, self.weight)
        z = torch.spmm(adj, z)

        return z
    
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class GraphAutoEncoder(nn.Module):
    
    def __init__(self, in_dim, latent_dim):
        # one layer for each gcn, in_dim = out_dim
        super(GraphAutoEncoder, self).__init__()
        
        # 1) Graph encoder -> h
        self.encoder = GCN(in_dim, latent_dim)
        
        # 2) MLP heads for z and v
        self.z_head = MLPHead(latent_dim, latent_dim)
        self.v_head = MLPHead(latent_dim, latent_dim)
        
        # 3) Decoder h -> decoder
        self.decoder = GCN(latent_dim, in_dim)
        
        # self.fuse = nn.Linear(2 * latent_dim, latent_dim)
        
    def forward(self, x, adj):
        
        h = self.encoder(x, adj)
        # h_relu = F.relu(h)
        # z = self.z_head(h)
        # v = self.v_head(h)
        
        out = self.decoder(h, adj)
        
        x_a = permutation(x)
        h_a = self.encoder(x_a, adj)
        
        return out, h, h_a
    
class DomainClassifier(nn.Module):
    
    def __init__(self, in_dim, n_batch):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_batch)
        )

    def forward(self, v):
        return self.net(v)
    
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        # Forward pass is the identity function
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and multiply by lambda during the backward pass
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)
        
class JADEAlignEncoder(nn.Module):
    
    def __init__(self, in_dim, latent_dim, n_batch, **kwargs):
        super(JADEAlignEncoder, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_batch = n_batch
        
        # 1. contrastive learning based on graph infomax (shared across batches)
        self.disc = Discriminator(n_h=latent_dim)
        self.disc2 = Discriminator(n_h=latent_dim)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.bi_stochastic = NonSquareSinkhornKnopp(max_iter=kwargs.get("max_iter",3), epsilon=1e-3)
        
        # # 1A. single/common autoencoder
        # self.autoencoder = GraphAutoEncoder(in_dim, latent_dim)
        
        ## 1B. different autoencoders
        self.autoencoders = nn.ModuleList([
            GraphAutoEncoder(in_dim, latent_dim) for _ in range(n_batch)
        ])
        
        # # Create shared head modules
        # shared_z_head = MLPHead(latent_dim, latent_dim)
        # shared_v_head = MLPHead(latent_dim, latent_dim)

        # # Replace the heads in each autoencoder with the shared instances
        # for autoencoder in self.autoencoders:
        #     autoencoder.z_head = shared_z_head
        #     autoencoder.v_head = shared_v_head
        
        # 2. n_batch-1 alignments (assert that n_batch>=2)
        self.Ms = nn.ParameterList([
            nn.Parameter(
                torch.randn(latent_dim, latent_dim)
            )
            for _ in range(n_batch-1)
        ])
        for param in self.Ms:
            nn.init.xavier_uniform_(param) 
            
        # # 3. batch prediction 
        # self.domainclassifier = DomainClassifier(latent_dim, n_batch)
        
        # # 4. domain classifier for g (using gradient reversal layer)
        # self.domainclassifier = DomainClassifier(latent_dim, n_batch)
        
        # # 5. minimizing mutual info of (v, z) 
        # self.disc_zv = Discriminator(n_h=latent_dim)
        
        
            
    def get_MisAlignmentLoss(self, z1, z2, alignment, cxy=None, cyx=None, type='paste'):
        if type == 'paste':
            cross_dist = torch.cdist(z1, z2)
            return (alignment * cross_dist).sum()
        else:
            # matdiff = z1 - torch.matmul(alignment, z2)
            # misalignment_loss = (matdiff**2).sum() / self.nsrc
            # misalignment_loss = self.misalignment_weight * misalignment_loss
            misalignment_loss = F.mse_loss(z1, torch.mm(cxy, z2)) + F.mse_loss(z2, torch.mm(cyx,z1))
            # misalignment_loss *= self.misalignment_weight
            
        return misalignment_loss
    
    
    def get_MisMaintainnessLoss(self, cxy, cyx, type='paste'):
        if type == 'paste':
            pi_norm = cxy
            diff = self.dist_mat_src - torch.mm(pi_norm, torch.mm(self.dist_mat_tgt, pi_norm.T))
            # maintainness_loss = self.mismaintainness_weight * torch.norm(diff) / self.nsrc
        else:
            recon = torch.mm(cxy, self.dist_mat_tgt @ cxy.T)
            recon2 = torch.mm(cyx, self.dist_mat_src @ cyx.T)

            # mean squared (Frobenius) error
            loss1 = (recon.pow(2)).mean()
            loss2 = (recon2.pow(2)).mean()
            maintainness_loss = loss1 + loss2

            # … then weight it as you wish
            # maintainness_loss *= self.mismaintainness_weight
            return maintainness_loss
    
    def forward(self, feature_set, spot_feature_set, adj_set, graph_neigh_set, label_CSL_set, dist_mat_set, **kwargs):
        if_norm_distort = kwargs.get("if_norm_distort", False)
        
        # feature_set: feature mats for B batches
        # adj_set: adj for B batches 
        # graph_neigh_set: graph_neigh for B batches, diff is that graph_neigh is directed
        
        # get embeddings from autoencoder
        batch_results = []
        for k, (feature, adj, graph_neigh) in enumerate(zip(feature_set, adj_set, graph_neigh_set)):
            # out, h, z, z_a, v, v_a = self.autoencoder(feature, adj)
            out, h, h_a = self.autoencoders[k](feature, adj)
            
            # quantities for graph infomax
            h_relu = self.relu(h)
            h_a_relu = self.relu(h_a)
            
            
            g = self.read(h_relu, graph_neigh)
            g = self.sigm(g)
            g_a = self.read(h_a_relu, graph_neigh)
            g_a = self.sigm(g_a)
            ret = self.disc(g, h_relu, h_a_relu)
            ret_a = self.disc(g_a, h_a_relu, h_relu)
            
            # s, s_stacked = self.read_spot(h, spot_cell_assignment)
            # # s = self.sigm(s)
            # s_stacked = self.sigm(s_stacked)
            # s_a, s_a_stacked = self.read_spot(h_a, spot_cell_assignment)
            # # s_a = self.sigm(s_a)
            # s_a_stacked = self.sigm(s_a_stacked)
            # ret_s = self.disc2(s_stacked, h, h_a)
            # ret_s_a = self.disc2(s_a_stacked, h_a, h)

            # # quantity for domain prediction
            # logit = self.domainclassifier(v)
            
            # h_rev = grad_reverse(h)
            # logit = self.domainclassifier(h_rev)
            
            batch_results.append({
                "out": out,
                "h":   h,
                "h_a": h_a,
                "g": g,
                "g_a": g_a,
                "ret": ret,
                "ret_a": ret_a,
            })
            
        # calculate losses
        loss_sl = 0
        loss_sl_sh = 0
        loss_feat = 0 
        loss_domain = 0
        loss_align = 0
        loss_align_fix = 0
        loss_maintain = 0
        loss_marginal = 0
        loss_sparsity = 0
        alignments = [None for _ in range(self.n_batch-1)]
        
        for k in range(len(feature_set)):
            
            # z = batch_results[k]["z"]
            # v = batch_results[k]["v"]
            
            # 1. infomax loss
            loss_sl_1 = self.loss_CSL(batch_results[k]["ret"], label_CSL_set[k])
            loss_sl_2 = self.loss_CSL(batch_results[k]["ret_a"], label_CSL_set[k])
            loss_sl += loss_sl_1 +  loss_sl_2
            
            # # 1b. infomax for h-s pair
            # loss_sl_1_sh = self.loss_CSL(batch_results[k]["ret_s"], label_CSL_set[k])
            # loss_sl_2_sh = self.loss_CSL(batch_results[k]["ret_s_a"], label_CSL_set[k])
            # loss_sl_sh += loss_sl_1_sh +  loss_sl_2_sh
            
            # loss_sl_zv_1 = self.loss_CSL(batch_results[k]["ret_zv"], label_CSL_set[k])
            # loss_sl_zv_2 = self.loss_CSL(batch_results[k]["ret_zv_a"], label_CSL_set[k])
            # loss_sl_zv += loss_sl_zv_1 +  loss_sl_zv_2
            
            # # 1+. negative zv contrast loss
            # sim = z @ v.T
            # logits = sim / 1.0  
            # log_probs = F.log_softmax(logits, dim=1)
            # loss_sl_zv = -log_probs.diagonal().mean()
            
            # 2. recon loss
            loss_feat += F.mse_loss(feature_set[k], batch_results[k]["out"])
            
            # # 3. domain classification loss
            # domain_labels = torch.zeros(feature_set[k].shape[0], self.n_batch, device=feature_set[k].device)
            # domain_labels[:, k] = 1
            # loss_domain += self.domain_criterion(batch_results[k]["logit"], domain_labels)
            
            # 4. orthgonalize z and v
            # cos_sim = (z * v).sum(dim=1) / (z.norm(dim=1) * v.norm(dim=1) + 1e-8)
            # loss_zv += (cos_sim).pow(2).mean()
            # sim = torch.abs(z @ v.T)
            # logits = sim / 1.0 
            # log_probs = F.log_softmax(logits, dim=1)
            # loss_zv = torch.sigmoid(log_probs.diagonal().mean() / 50)
            
            # 5. alignment related loss
            if k > 0:
                srcemb = batch_results[k-1]["h"]
                tgtemb = batch_results[k]["h"]
                
                # compute alignment and conditional prob. mat
                C = srcemb @ self.Ms[k-1] @ self.Ms[k-1].T @ tgtemb.T / np.sqrt(self.Ms[k-1].shape[0])
                # C = srcemb @ tgtemb.T / np.sqrt(self.Ms[k-1].shape[0])
                unnorm_alignment = F.softmax(C, dim = 1) / feature_set[k-1].shape[0]
                alignment = self.bi_stochastic.fit(unnorm_alignment)
                self.cxy = alignment / alignment.sum(dim=1, keepdim=True)
                self.cyx = alignment.T / alignment.T.sum(dim=1, keepdim=True)
                alignments[k-1] = alignment
                
                ### mis alignment
                loss_align += self.get_MisAlignmentLoss(srcemb, tgtemb, alignment, self.cxy, self.cyx, 'paste')
                loss_align_fix += self.get_MisAlignmentLoss(feature_set[k-1], feature_set[k], alignment, self.cxy, self.cyx, 'notpaste')
                
                ### mis maintain
                diff1 = dist_mat_set[k-1] - torch.mm(self.cxy, torch.mm(dist_mat_set[k], self.cxy.T))
                diff2 = dist_mat_set[k] - torch.mm(self.cyx, torch.mm(dist_mat_set[k-1], self.cyx.T))
                # if if_norm_distort:
                #     loss_maintain += torch.norm(diff1)**2/feature_set[k-1].shape[0]/feature_set[k-1].shape[0] + torch.norm(diff2)**2/feature_set[k].shape[0]/feature_set[k].shape[0]
                # else:
                #     loss_maintain += torch.norm(diff1)/feature_set[k-1].shape[0] + torch.norm(diff2)/feature_set[k].shape[0]
                loss_maintain += torch.norm(diff1)/feature_set[k-1].shape[0] + torch.norm(diff2)/feature_set[k].shape[0]
                
                ### marginal constrain, ensuring the alignment mat is a doubly stochastic mat.
                p = torch.ones(feature_set[k].shape[0], device=feature_set[k].device)/feature_set[k].shape[0]
                q = torch.matmul(torch.ones(feature_set[k-1].shape[0], device=feature_set[k].device), alignment)
                loss_marginal += (F.kl_div(p, q)-F.kl_div(p, p)) * feature_set[k].shape[0]
                
                ### sparsity constrain
                loss_sparsity += -torch.sum(alignment * torch.log(alignment + 1e-10))
            
            # 5b. alignment related loss, but on a spot-level
            if k > 0 and 0:
                
                # extreme case: s=h
                srcemb = batch_results[k-1]["s"]
                tgtemb = batch_results[k]["s"]
                m1 = srcemb.shape[0]
                m2 = tgtemb.shape[0]
                
                # compute alignment and conditional prob. mat
                C = srcemb @ self.Ms[k-1] @ self.Ms[k-1].T @ tgtemb.T / np.sqrt(self.Ms[k-1].shape[0])
                # C = srcemb @ tgtemb.T / np.sqrt(self.Ms[k-1].shape[0])
                unnorm_alignment = F.softmax(C, dim = 1) / feature_set[k-1].shape[0]
                alignment = self.bi_stochastic.fit(unnorm_alignment)
                self.cxy = alignment / alignment.sum(dim=1, keepdim=True)
                self.cyx = alignment.T / alignment.T.sum(dim=1, keepdim=True)
                alignments[k-1] = alignment
                
                ### mis alignment
                loss_align += self.get_MisAlignmentLoss(srcemb, tgtemb, alignment, self.cxy, self.cyx, 'paste')
                loss_align_fix += self.get_MisAlignmentLoss(spot_feature_set[k-1], spot_feature_set[k], alignment, self.cxy, self.cyx, 'paste')
                
                ### mis maintain
                diff1 = dist_mat_set[k-1] - torch.mm(self.cxy, torch.mm(dist_mat_set[k], self.cxy.T))
                diff2 = dist_mat_set[k] - torch.mm(self.cyx, torch.mm(dist_mat_set[k-1], self.cyx.T))
                loss_maintain += torch.norm(diff1)/m1 + torch.norm(diff2)/m2
                
                ### marginal constrain, ensuring the alignment mat is a doubly stochastic mat.
                p = torch.ones(m2, device=feature_set[k].device)/m2
                q = torch.matmul(torch.ones(m1, device=feature_set[k].device), alignment)
                loss_marginal += (F.kl_div(p, q)-F.kl_div(p, p)) * m2
                
                ### sparsity constrain
                loss_sparsity += -torch.sum(alignment * torch.log(alignment + 1e-10))
            
            
            # # 6. reconstruct A loss.
            # print(self.sigm(h @ h.T / 64)[:5,:5], adj.to_dense()[:5,:5])
            # loss_adj += F.binary_cross_entropy(self.sigm(h @ h.T / 64), adj.to_dense())
            
               
        # return batch_results, alignments, loss_sl, loss_feat, loss_domain, loss_zv, loss_align, loss_align_fix, loss_maintain, loss_marginal
        return batch_results, alignments, loss_sl, loss_sl_sh, loss_feat, loss_domain, loss_align, loss_align_fix, loss_maintain, loss_marginal, loss_sparsity