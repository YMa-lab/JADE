import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from . preprocess import permutation
from . sinkhorn import NonSquareSinkhornKnopp
import time, contextlib, torch

# ---------- tiny helper ---------- #
# @contextlib.contextmanager
# def timer(tag):
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     t0 = time.perf_counter()
#     yield
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     print(f"{tag}: {(time.perf_counter()-t0)*1e3:.2f} ms")

# comment-out the old timer or leave it above
@contextlib.contextmanager
def timer(*_args, **_kwargs):   # ignores everything
    yield                       # does nothing

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
        super().__init__()

    # NEW signature: three positional args
    def forward(self,
                emb: torch.Tensor,          # (n × d) cell embeddings
                mask_sparse: torch.Tensor,  # (m × n) CSR 0–1 membership
                rowsum: torch.Tensor):      # (m × 1)   cached row-sums
        """
        Returns an ℓ2-normalised spot-level embedding (m × d).
        """
        vsum = torch.sparse.mm(mask_sparse, emb)   # fast CSR × dense GEMM
        global_emb = vsum / rowsum                 # broadcast divide
        return F.normalize(global_emb, p=2, dim=1) # m × d

    
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
            diff1 = self.dist_mat_src - torch.mm(cxy, torch.mm(self.dist_mat_tgt, cxy.T))
            diff2 = self.dist_mat_tgt - torch.mm(cyx, torch.mm(self.dist_mat_src, cyx.T))
            maintainness_loss = torch.norm(diff1)/self.nsrc + torch.norm(diff2)/self.ntgt
            # maintainness_loss *= self.mismaintainness_weight
        return maintainness_loss
    
        
    def read_spot(self, h, spot_cell_assignment):
        device = h.device
        n, d = h.shape
        m = len(spot_cell_assignment)
        
        # Create a mapping tensor: for each cell, record its spot index.
        # Each cell should belong to exactly one spot.
        assignment = torch.empty(n, dtype=torch.long, device=device)
        for spot_idx, cell_indices in enumerate(spot_cell_assignment):
            # Ensure cell_indices is a tensor on the correct device.
            if not torch.is_tensor(cell_indices):
                cell_indices = torch.tensor(cell_indices, dtype=torch.long, device=device)
            else:
                cell_indices = cell_indices.to(device)
            assignment[cell_indices] = spot_idx

        # Compute counts: how many cells per spot.
        counts = torch.bincount(assignment, minlength=m).unsqueeze(1).to(h.dtype)  # shape: (m, 1)

        # Sum cell embeddings per spot using scatter_add.
        s_sum = torch.zeros(m, h.shape[1], dtype=h.dtype, device=h.device)
        s_sum = s_sum.scatter_add(0, assignment.unsqueeze(1).expand(-1, h.shape[1]), h)

        # Compute the average (spot embeddings)
        s = s_sum / (counts + 1e-8)

        # Reconstruct stacked spot embeddings
        s_stacked = s[assignment]
        
        return s, s_stacked
        
    def forward(self, feature_set, spot_feature_set, adj_set,
            gmask_set, gmask_rowsum_set,   # ← two lists
            label_CSL_set, dist_mat_set, spot_cell_assignment_set, **kwargs):

        batch_results=[] ; alignments=[None]*(self.n_batch-1)
        (loss_sl,loss_sl_sh,loss_feat,loss_domain,
        loss_align,loss_align_fix,loss_maintain,
        loss_marginal,loss_sparsity)=(0,)*9

        for k, (feat, adj, gmask, rowsum, sca) in enumerate(
        zip(feature_set,
            adj_set,
            gmask_set,
            gmask_rowsum_set,     # ← new element
            spot_cell_assignment_set)):
            with timer(f"enc/dec[{k}]"):
                out,h,h_a = self.autoencoders[k](feat,adj)

            # finer-grained timers inside InfoMax
            with timer(f"InfoMax[{k}]"):                       # total
                with timer(f"IM[{k}]-relu"):
                    h_r, h_a_r = self.relu(h), self.relu(h_a)

                with timer(f"IM[{k}]-readG"):
                    g   = self.sigm(self.read(h_r,   gmask, rowsum))
                    g_a = self.sigm(self.read(h_a_r, gmask, rowsum))

                with timer(f"IM[{k}]-discG"):
                    ret   = self.disc(g,   h_r,  h_a_r)
                    ret_a = self.disc(g_a, h_a_r, h_r)

                with timer(f"IM[{k}]-readSpot"):
                    s,  s_stacked  = self.read_spot(h,  sca)
                    s_a,sA_stacked = self.read_spot(h_a, sca)

                with timer(f"IM[{k}]-discS"):
                    s_stacked  = self.sigm(s_stacked)
                    sA_stacked = self.sigm(sA_stacked)
                    ret_s   = self.disc2(s_stacked,  h,  h_a)
                    ret_s_a = self.disc2(sA_stacked, h_a, h)


            batch_results.append(dict(out=out,h=h,h_a=h_a,ret=ret,ret_a=ret_a,
                                    ret_s=ret_s,ret_s_a=ret_s_a,s=s))

        # ---------- loss & alignment loop ---------- #
        for k in range(len(feature_set)):
            br = batch_results[k]
            loss_sl     += self.loss_CSL(br["ret"],     label_CSL_set[k]) \
                        + self.loss_CSL(br["ret_a"],   label_CSL_set[k])
            loss_sl_sh  += self.loss_CSL(br["ret_s"],   label_CSL_set[k]) \
                        + self.loss_CSL(br["ret_s_a"], label_CSL_set[k])
            loss_feat   += F.mse_loss(feature_set[k], br["out"])

            if k==0: continue
            with timer(f"align[{k-1}->{k}]"):
                src,tgt   = batch_results[k-1]["s"], batch_results[k]["s"]
                m1,m2     = src.size(0), tgt.size(0)
                C = src @ self.Ms[k-1] @ self.Ms[k-1].T @ tgt.T / src.size(1)**0.5
                P = self.bi_stochastic.fit(F.softmax(C,dim=1)/m1)
                cxy,cyx   = P/P.sum(1,keepdim=True), P.T/P.T.sum(1,keepdim=True)
                alignments[k-1]=P

                loss_align      += self.get_MisAlignmentLoss(src,tgt,P,cxy,cyx,'paste')
                loss_align_fix  += self.get_MisAlignmentLoss(spot_feature_set[k-1],
                                                            spot_feature_set[k],P,cxy,cyx,'paste')
                diff1 = dist_mat_set[k-1]-cxy@dist_mat_set[k]@cxy.T
                diff2 = dist_mat_set[k]-cyx@dist_mat_set[k-1]@cyx.T
                loss_maintain   += diff1.norm()/m1 + diff2.norm()/m2
                p = torch.full((m2,),1/m2,device=feat.device)
                q = torch.ones(m1,device=feat.device)@P
                loss_marginal   += (F.kl_div(p,q)-F.kl_div(p,p))*m2
                loss_sparsity   += -(P*P.log()).sum()

        return (batch_results,alignments,loss_sl,loss_sl_sh,loss_feat,loss_domain,
                loss_align,loss_align_fix,loss_maintain,loss_marginal,loss_sparsity)
