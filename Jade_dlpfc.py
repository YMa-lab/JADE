import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import softmax
from JADE.JADE import JADE
from JADE.utils import * 
from utils import *
# Read DLPFC data.
sns.set_style('whitegrid')
sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('./data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"],[ "151669", "151670","151671", "151672"],[ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]
# Slice AB of Sample III.
srcadata = adatas["151673"]
tgtadata = adatas["151674"]
n1 = srcadata.shape[0]

# NOTE: We use nspot = 1000 for FastJADE
srcadata = add_spot_cell_assignment(layer_groups[2][0], 1000)
tgtadata = add_spot_cell_assignment(layer_groups[2][1], 1000)
adatalist = [srcadata, tgtadata]
aligner = JADE(
    adatalist, 
    ngenes=3000,
    seed=2024, 
    lr=0.002, 
    weight_decay=0.0001, 
    pretrain_epochs=200,
    epochs=800, 
    misalignment_weight=0.1,
    mismaintainness_weight=2.0,
    n_clusters=7,
    n_neighbors=3,
    max_iter=3,
    verbose=False
)
# Training....
adatalist, alignments = aligner.train()
# Postprocessing...
for adata in adatalist:
    clustering(adata, reduced_dim=20, key="emb_rec", n_clusters=7, refinement=True, radius=25)
ARI = []
for adata in adatalist:
    ARI.append(adjusted_rand_score(adata.obs['layer_guess_reordered'], adata.obs['domain']))
print(ARI)

# Show cluster results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))  # stack vertically
for k, adata in enumerate(adatalist):
    layer_to_color_map = {'Layer{0}'.format(i+1): sns.color_palette()[i] for i in range(6)}
    layer_to_color_map['WM'] = sns.color_palette()[6]
    layer_to_color_map_kmeans = {i: sns.color_palette()[i] for i in range(1, 8)}
    colors = list(adata.obs['domain'].astype('int').map(layer_to_color_map_kmeans))
    ax = axes[k]
    ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
               linewidth=0, s=200, marker=".", color=colors)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    if k == 0:
        ax.set_title('JADE', fontsize=40)
fig.tight_layout()

# Umap plot
combined_adata = ad.concat(
    adatalist,
    axis=0,
    join="outer",
    label="slice_id",
    keys=["slice A", "slice B"]
)
clustering(combined_adata, reduced_dim=30, key="emb", n_clusters=7, refinement=True, radius=25)
sc.pp.neighbors(
    combined_adata,
    use_rep='emb',
    n_neighbors=50,       
    n_pcs=30,             
    metric='euclidean', 
    random_state=2025
)
sc.tl.umap(
    combined_adata,
    min_dist=0.3,       
    spread=1.0,           
    random_state=2025,
)
sc.pl.umap(combined_adata, color='slice_id', title='', show=False)
ax = fig.axes[0]
legend = ax.get_legend()
if legend:
    for text in legend.get_texts():
        text.set_fontsize(20)  
    legend.set_title("Slice ID")
    legend.get_title().set_fontsize(20)
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.title('By Sample',fontsize=20)
sc.pl.umap(combined_adata, color='layer_guess_reordered', title='', show=False)
plt.xlabel('')
plt.ylabel('JADE',fontsize=20)
plt.xticks([])
plt.yticks([])
plt.title('By Cluster',fontsize=20)


# Post-processing alignment matrix
# TODO: integrate into our package
niter_values = [4]
colors = [generate_color(niter) for niter in niter_values]
tau = 1.0
for niter, color in zip(niter_values, colors):
    M = aligner.model.Ms[0].cpu().detach().numpy()
    srcemb = adatalist[0].obsm['emb_unnorm']
    tgtemb = adatalist[1].obsm['emb_unnorm']
    C = srcemb @ M @ M.T @ tgtemb.T / np.sqrt(M.shape[0]) / tau
    unnorm_alignment = softmax(C, axis=1) / srcemb.shape[0]
    pi = unnorm_alignment
    # multiple row-col normalizations 
    pi = multiple_row_col_renormalizations(pi, niter=niter)
    # output = np.zeros_like(pi)
    # max_indices = np.argmax(pi, axis=1)
    # output[np.arange(pi.shape[0]), max_indices] = 1
    # pi = output
    s = cal_pairwise_alignment_score(srcadata,tgtadata,pi)
    print(f'niter:{niter}, alignment score:{s}')
    marginal = np.sum(pi,axis=0)
    sorted_marginal = np.sort(marginal)
    plt.plot(np.arange(len(sorted_marginal)), np.cumsum(sorted_marginal), c=color)
