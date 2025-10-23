import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import anndata as ad

def generate_color(niter, max_niter=8):
    # Start with red: (1, 0, 0) and gradually increase the green and blue components
    r = 1.0
    g = min(1.0, niter / max_niter)  # Green increases as niter increases
    b = min(1.0, niter / max_niter)  # Blue also increases as niter increases
    return (r, g, b)  # Return RGB tuple


# visualization utils.
def prepare_merfish_anndata(name):
    df = pd.read_csv('./data/merfish/cntmat_'+name+'.csv', delimiter=' ')
    infodf = pd.read_csv('./data/merfish/info_'+name+'.csv')
    gene_expression = df.T
    spatial_location = infodf.iloc[:,1:3]

    adata = ad.AnnData(X=sparse.csr_matrix(gene_expression.values))
    adata.obsm['spatial'] = spatial_location.values
    adata.obs['layer_guess_reordered'] = infodf.iloc[:,5].astype('category').values
    cmap = plt.get_cmap("tab10")
    layer_to_color_map = {label:mcolors.to_hex(cmap(i)) for i, label in enumerate(set(adata.obs['layer_guess_reordered'].astype('str')))}
    adata.uns['layer_guess_reordered_colors'] = [mcolors.to_hex(cmap(i)) for i, label in enumerate(set(adata.obs['layer_guess_reordered'].astype('str')))]
    return adata

# def largest_indices(ary, n):
#     """Returns the n largest indices from a numpy array."""
#     flat = ary.flatten()
#     indices = np.argpartition(flat, -n)[-n:]
#     indices = indices[np.argsort(-flat[indices])]
#     return np.unravel_index(indices, ary.shape)

def plot2D_samples_mat(xs, xt, G, thr=1e-8,alpha=0.2,top=1000,weight_alpha=False,**kwargs):
    # 
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    idx = largest_indices(G,top)
    for l in range(len(idx[0])):
        plt.plot([xs[idx[0][l], 0], xt[idx[1][l], 0]], [xs[idx[0][l], 1], xt[idx[1][l], 1]],alpha=alpha*(1-weight_alpha)+(weight_alpha*G[idx[0][l],idx[1][l]] /mx),c='k')

def plot_slice_pairwise_alignment(slice1,slice2,pi,thr=1-1e-8,alpha=0.05,top=1000,name='',save=False,weight_alpha=False, rev_slice2=False):
    coordinates1,coordinates2 = slice1.obsm['spatial'],slice2.obsm['spatial']
    if rev_slice2:
        x_mean = np.mean(coordinates2[:, 0])
        coordinates2[:, 0] = 2 * x_mean - coordinates2[:, 0]

    offset = (coordinates1[:,0].max()-coordinates2[:,0].min())*1.1
    temp = np.zeros(coordinates2.shape)
    temp[:,0] = offset
    plt.figure(figsize=(20,10))
    plot2D_samples_mat(coordinates1, coordinates2+temp, pi,thr=thr, c='k',alpha=alpha,top=top,weight_alpha=weight_alpha)
    plt.scatter(coordinates1[:,0],coordinates1[:,1],linewidth=0,s=100, marker=".",color=list(slice1.obs['layer_guess_reordered'].map(dict(zip(slice1.obs['layer_guess_reordered'].cat.categories,slice1.uns['layer_guess_reordered_colors'])))))
    plt.scatter(coordinates2[:,0]+offset,coordinates2[:,1],linewidth=0,s=100, marker=".",color=list(slice2.obs['layer_guess_reordered'].map(dict(zip(slice2.obs['layer_guess_reordered'].cat.categories,slice2.uns['layer_guess_reordered_colors'])))))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

def plot_slice_spot_pairwise_alignment(slice1,slice2,pi_spot,thr=1-1e-8,alpha=0.05,top=1000,name='',save=False,weight_alpha=False):
    coordinates1,coordinates2 = slice1.obsm['spatial'],slice2.obsm['spatial']
    coordinates1_spot, coordinates2_spot = slice1.uns['spot_spatial'], slice2.uns['spot_spatial']
    offset = (coordinates1[:,0].max()-coordinates2[:,0].min())*1.1
    temp = np.zeros(coordinates2_spot.shape)
    temp[:,0] = offset
    plt.figure(figsize=(20,10))
    plot2D_samples_mat(coordinates1_spot, coordinates2_spot+temp, pi_spot,thr=thr, c='k',alpha=alpha,top=top,weight_alpha=weight_alpha)
    
    plt.scatter(coordinates1[:,0],coordinates1[:,1],linewidth=0,s=100, marker=".",color=list(slice1.obs['layer_guess_reordered'].map(dict(zip(slice1.obs['layer_guess_reordered'].cat.categories,slice1.uns['layer_guess_reordered_colors'])))))
    plt.scatter(coordinates2[:,0]+offset,coordinates2[:,1],linewidth=0,s=100, marker=".",color=list(slice2.obs['layer_guess_reordered'].map(dict(zip(slice2.obs['layer_guess_reordered'].cat.categories,slice2.uns['layer_guess_reordered_colors'])))))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
   
def plot3D_samples_mat(xs, xt, labels, labelt, G, z_offset, thr=1e-8, alpha=0.2, top=1000, weight_alpha=False, ax=None, **kwargs):
	if ax is None:
			ax = plt.gca()
	if ('color' not in kwargs) and ('c' not in kwargs):
			kwargs['color'] = 'k'
	mx = G.max()
	idx = largest_indices(G, top)
	for l in range(len(idx[0])):
		x_line = [xs[idx[0][l], 0], xt[idx[1][l], 0]]
		y_line = [xs[idx[0][l], 1], xt[idx[1][l], 1]]
		z_line = [0, z_offset]
		line_alpha = alpha * (1 - weight_alpha) + (weight_alpha * G[idx[0][l], idx[1][l]] / mx)
		if labels[idx[0][l]] == labelt[idx[1][l]]:
			ax.plot(x_line, y_line, z_line, alpha=line_alpha, c='red', linewidth=0.5)
		else:
			ax.plot(x_line, y_line, z_line, alpha=line_alpha, c='k', linewidth=0.5)
   
def plot_3d_slice_pairwise_alignment(slice1, slice2, pi, thr=1-1e-8, alpha=0.05, top=1000, name='', save=False, weight_alpha=False, z_offset=None):
	coordinates1 = slice1.obsm['spatial']
	coordinates2 = slice2.obsm['spatial']
	coordinates1 -= coordinates1.min(axis=0)
	coordinates2 -= coordinates2.min(axis=0)
	label1 = slice1.obs['layer_guess_reordered']
	label2 = slice2.obs['layer_guess_reordered']

	# Determine automatic z-offset if not provided
	if z_offset is None:
			z_offset = np.ptp(coordinates1[:,1]) * 0.5  # proportional vertical offset

	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(111, projection='3d')

	# Plot matching lines
	plot3D_samples_mat(coordinates1, coordinates2, label1, label2, pi, z_offset, thr=thr, alpha=alpha, top=top, weight_alpha=weight_alpha, ax=ax)

	# Plot points from slice1 at z=0
	colors1 = slice1.obs['layer_guess_reordered'].map(dict(zip(slice1.obs['layer_guess_reordered'].cat.categories,slice1.uns['layer_guess_reordered_colors'])))
	ax.scatter(coordinates1[:, 0], coordinates1[:, 1], np.zeros(coordinates1.shape[0]),
							c=list(colors1), s=60, marker='o', depthshade=False, label='Slice 1', alpha=0.2)

	# Plot points from slice2 at z=z_offset
	colors2 = slice2.obs['layer_guess_reordered'].map(dict(zip(slice2.obs['layer_guess_reordered'].cat.categories,slice2.uns['layer_guess_reordered_colors'])))
	ax.scatter(coordinates2[:, 0], coordinates2[:, 1], np.full(coordinates2.shape[0], z_offset),
							c=list(colors2), s=60, marker='o', depthshade=False, label='Slice 2', alpha=0.2)

	# Aesthetic settings
	# ax.set_xlabel('X axis')
	# ax.set_ylabel('Y axis')
	# ax.set_zlabel('Z axis')
	ax.set_title('3D Pairwise Alignment')
	ax.view_init(elev=30, azim=120)  # adjust angle as needed
	# plt.legend()

	if save:
			plt.savefig(f'{name}.png', bbox_inches='tight', dpi=300)

	plt.show()


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def combined_pca_plot(batch1: np.ndarray, batch2: np.ndarray, n_components=500, batch_labels=('batch1','batch2')):
    """
    Perform PCA on combined data from two batches and plot first two PCs.

    Parameters
    ----------
    batch1 : array-like, shape (n1, p)
    batch2 : array-like, shape (n2, p)
    batch_labels : tuple of str
        Labels for the two batches in the legend.
    """
    # Stack the two batches into one matrix
    X = np.vstack([batch1, batch2])
    # Create an array of labels (0 for batch1, 1 for batch2)
    labels = np.array([0]*batch1.shape[0] + [1]*batch2.shape[0])

    # Fit PCA on the combined data
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    # Split the transformed coordinates back into two batches
    Z1 = Z[labels == 0]
    Z2 = Z[labels == 1]

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(Z1[:, 0], Z1[:, 1], s=2, label=batch_labels[0])
    plt.scatter(Z2[:, 0], Z2[:, 1], s=2, label=batch_labels[1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return Z, pca


from sklearn.metrics.pairwise import rbf_kernel
def compute_mmd2_u(X, Y, gamma=None):
    X, Y = np.asarray(X), np.asarray(Y)
    m, n = X.shape[0], Y.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    sum_K_XX = K_XX.sum() - np.trace(K_XX)
    sum_K_YY = K_YY.sum() - np.trace(K_YY)
    mmd2 = ( sum_K_XX / (m*(m-1))
           + sum_K_YY / (n*(n-1))
           - 2 * K_XY.mean() )
    return mmd2


# TODO: alignment score; corrected alignment score
def evaluate_alignment(combined_adata, obsm_key: str, label_key: str, n_sliceA: int):
    """
    Evaluate 1-NN alignment accuracy and misalignment between two concatenated slices in AnnData.

    Parameters:
    - combined_adata: AnnData
        An AnnData object containing two concatenated slices (slice A first, then slice B).
    - obsm_key: str
        The key in .obsm where the embeddings are stored (e.g., 'emb', 'STAligner').
    - label_key: str
        The key in .obs where the true labels are stored (e.g., 'layer_guess_reordered').
    - n_sliceA: int
        Number of cells in slice A (the first slice).

    Returns: dict with keys
    - acc_A2B: float
        1-NN accuracy from slice A to slice B.
    - acc_B2A: float
        1-NN accuracy from slice B to slice A.
    - mean_acc: float
        Mean of A→B and B→A accuracies.
    - misalign_frac_A2B: float
        Fraction of slice A cells whose nearest neighbor in B has a different label.
    - misalign_frac_B2A: float
        Fraction of slice B cells whose nearest neighbor in A has a different label.
    - unmatched_frac_B: float
        Fraction of slice B cells never chosen as a nearest neighbor by any slice A cell.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Extract embeddings and labels as numpy arrays
    emb = combined_adata.obsm[obsm_key]
    labels = combined_adata.obs[label_key].astype(str).to_numpy()

    # Split embeddings and labels into the two slices
    embA, embB = emb[:n_sliceA], emb[n_sliceA:]
    lblA, lblB = labels[:n_sliceA], labels[n_sliceA:]

    # Helper to compute one-way NN accuracy and misalignment
    def one_way(src_emb, tgt_emb, src_lbls, tgt_lbls):
        nbrs = NearestNeighbors(n_neighbors=1).fit(tgt_emb)
        _, idx = nbrs.kneighbors(src_emb)
        idx = idx[:, 0]
        preds = tgt_lbls[idx]
        acc = np.mean(preds == src_lbls)
        mis_frac = np.mean(preds != src_lbls)
        return acc, mis_frac, idx

    # Compute A -> B
    acc_A2B, mis_frac_A2B, idx_A2B = one_way(embA, embB, lblA, lblB)
    # Compute B -> A
    acc_B2A, mis_frac_B2A, idx_B2A = one_way(embB, embA, lblB, lblA)

    # Compute fraction of B never matched by any A
    matched_B = set(idx_A2B.tolist())
    unmatched_B = set(range(len(lblB))) - matched_B
    unmatched_frac_B = len(unmatched_B) / len(lblB)

    return {
        'acc_A2B': acc_A2B,
        'acc_B2A': acc_B2A,
        'mean_acc': (acc_A2B + acc_B2A) / 2,
        'misalign_frac_A2B': mis_frac_A2B,
        'misalign_frac_B2A': mis_frac_B2A,
        'unmatched_frac_B': unmatched_frac_B
    }



def largest_indices(ary, n, epsilon=1e-8):
    flat = ary.ravel()
    valid = np.where(flat >= epsilon)[0]
    if valid.size == 0:
        return (np.array([], dtype=int),) * ary.ndim

    k = min(n, valid.size)
    vals = flat[valid]

    # deterministic jitter: proportional to flat‐index, << epsilon
    eps_j = np.finfo(flat.dtype).eps
    jitter = valid.astype(flat.dtype) * eps_j
    jittered = vals + jitter

    # argsort by descending jittered value
    order = np.argsort(-jittered, kind='stable')
    topk_flat = valid[order[:k]]

    return np.unravel_index(topk_flat, ary.shape)


def compute_cluster_accuracy(slice1, slice2, pi, top=1000, cluster=None):
    """
    Compute the fraction of the top‐`top` links (by pi value) that align labels,
    optionally restricted to one cluster in slice1.
    Uses deterministic jitter internally to break any exact ties.
    """
    # restrict to your cluster if given
    if cluster is not None:
        mask = (slice1.obs['layer_guess_reordered'] == cluster).to_numpy()
        pi = pi.copy()
        pi[~mask, :] = -1

    # get top links (deterministic tie break via jitter in largest_indices)
    i_idx, j_idx = largest_indices(pi, top)

    labels1 = slice1.obs['layer_guess_reordered'].to_numpy()
    labels2 = slice2.obs['layer_guess_reordered'].to_numpy()

    if len(i_idx) == 0:
        return np.nan
    correct = (labels1[i_idx] == labels2[j_idx]).sum()
    return correct / len(i_idx)
# row_normalize as you defined
def row_normalize(mat, eps=1e-12):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, eps, row_sums)
    return mat / row_sums

def col_normalize(mat, eps=1e-12):
    # sum over rows to get 1×n_cols
    col_sums = mat.sum(axis=0, keepdims=True)
    # avoid division by zero
    col_sums = np.where(col_sums == 0, eps, col_sums)
    return mat / col_sums

# helper already in your code
def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

import numpy as np
from sklearn.neighbors import NearestNeighbors

def local_simpson_index(coords, labels, k=20, complement=True):
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    n = len(labels)

    # find k nearest neighbors (including self)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coords)
    # neighbors indices: shape (n_spots, k)
    neigh_idx = nbrs.kneighbors(return_distance=False)

    # pre-allocate output
    D = np.zeros(n, dtype=float)

    for i in range(n):
        # gather the labels of the k neighbors of spot i
        lab_neighbors = labels[neigh_idx[i]]
        # count occurrences of each label
        # we can use np.bincount over the label range
        counts = np.bincount(lab_neighbors, minlength=labels.max()+1)
        freqs = counts / counts.sum()
        simpson = np.sum(freqs**2)
        D[i] = 1 - simpson if complement else simpson

    return D

def compute_iLISI(emb, batch_labels, k=30):
    """
    emb           : array (n_cells × n_dims) embedding coordinates
    batch_labels  : integer array (n_cells,) with values 0…B‑1
    k             : number of neighbors (including self)
    returns       : iLISI (n_cells,)
    """
    n = emb.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(emb)
    neigh_idx = nbrs.kneighbors(return_distance=False)

    iLISI = np.zeros(n, float)
    for i in range(n):
        labs = batch_labels[neigh_idx[i]]
        counts = np.bincount(labs, minlength=batch_labels.max()+1)
        p = counts / counts.sum()
        simpson = np.sum(p**2)
        iLISI[i] = 1.0 / simpson

    return iLISI


def compute_cluster_accuracy(slice1, slice2, pi, top=1000, cluster=None):
    # optionally zero out rows not in the cluster
    if cluster is not None:
        mask = (slice1.obs['layer_guess_reordered'] == cluster).to_numpy()
        pi = pi.copy()
        pi[~mask, :] = 0

    # get top links
    i_idx, j_idx = largest_indices(pi, top)

    # get labels
    labels1 = slice1.obs['layer_guess_reordered'].to_numpy()
    labels2 = slice2.obs['layer_guess_reordered'].to_numpy()

    # compute accuracy
    correct = np.sum(labels1[i_idx] == labels2[j_idx])
    return correct / len(i_idx) if len(i_idx)>0 else np.nan

# row_normalize as you defined
def row_normalize(mat, eps=1e-12):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, eps, row_sums)
    return mat / row_sums

def col_normalize(mat, eps=1e-12):
    # sum over rows to get 1×n_cols
    col_sums = mat.sum(axis=0, keepdims=True)
    # avoid division by zero
    col_sums = np.where(col_sums == 0, eps, col_sums)
    return mat / col_sums

# helper already in your code
def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

# # New: version that draws onto a given Axes
# def plot_slice_pairwise_alignment_ax(
#     ax,
#     slice1,
#     slice2,
#     labels1,
#     labels2,
#     pi,
#     thr=1-1e-8,
#     alpha=0.05,
#     top=1000,
#     weight_alpha=False,
#     cluster=None
# ):
    
#     # scatter spots
#     import matplotlib.colors as mcolors
#     import colorsys
#     def desaturate(color, factor=0.6):
#         r, g, b = mcolors.to_rgb(color)
#         h, l, s = colorsys.rgb_to_hls(r, g, b)
#         r2, g2, b2 = colorsys.hls_to_rgb(h, l, s*factor)
#         return mcolors.to_hex((r2, g2, b2))
    
#     # coordinates and offset
#     coords1 = slice1.obsm['spatial']
#     coords2 = slice2.obsm['spatial']
#     offset = (coords1[:,0].max() - coords2[:,0].min()) * 1.1
#     coords2_off = coords2 + np.array([offset, 0])

#     # mask to cluster if requested
#     if cluster is not None:
#         mask = (slice1.obs['layer_guess_reordered'] == cluster).to_numpy()
#         pi_masked = pi.copy()
#         pi_masked[~mask, :] = -1
#     else:
#         mask = np.ones(len(coords1), bool)
#         pi_masked = pi

#     # pick top matches
#     i_idx, j_idx = largest_indices(pi_masked, top)
#     G_masked = np.zeros_like(pi)
#     G_masked[i_idx, j_idx] = pi_masked[i_idx, j_idx]

#     # draw lines
#     mx = pi.max() if pi.size>0 else 1.0
#     for i, j in zip(i_idx, j_idx):
#         correct = (labels1[i] == labels2[j])
#         col = desaturate('blue',0.8) if correct else desaturate('yellow',0.8)
#         a = alpha*(1-weight_alpha) + (weight_alpha * pi[i,j]/mx)
#         ax.plot(
#             [coords1[i,0], coords2_off[j,0]],
#             [coords1[i,1], coords2_off[j,1]],
#             color=col, alpha=2.0*a, linewidth=10.0
#         )

#     raw = ["F7D86A","D4A5A5","A29BFE","F49097","FFA473","A3E4D7","56B4D3"]
#     new_colors = [f"#{h}" for h in raw]
#     cats = slice1.obs['layer_guess_reordered'].cat.categories.tolist()
#     orig_cmap = {cat: new_colors[i] for i,cat in enumerate(cats)}

#     # 2) build new cmap by mapping each original color through `desaturate`
#     new_cmap = { cat: desaturate(col, factor=0.6) 
#                 for cat, col in orig_cmap.items() }

#     cols1 = [new_cmap.get(cat, "#cccccc")
#          for cat in slice1.obs['layer_guess_reordered']]
#     cols2 = [new_cmap.get(cat, "#cccccc")
#          for cat in slice2.obs['layer_guess_reordered']]

#     ax.scatter(coords1[:,0], coords1[:,1], s=100, marker='.', color=cols1)
#     ax.scatter(coords2_off[:,0], coords2_off[:,1], s=100, marker='.', color=cols2)

#     ax.invert_yaxis()
#     ax.axis('off')
    
#     return i_idx, j_idx


# New: version that draws onto a given Axes
def plot_slice_pairwise_alignment_ax(
    ax,
    slice1,
    slice2,
    labels1,
    labels2,
    pi,
    thr=1-1e-8,
    alpha=0.05,
    top=1000,
    weight_alpha=False,
    cluster=None
):
    
    # scatter spots
    import matplotlib.colors as mcolors
    import colorsys
    def desaturate(color, factor=0.6):
        r, g, b = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s*factor)
        return mcolors.to_hex((r2, g2, b2))
    
    # coordinates and offset
    coords1 = slice1.obsm['spatial']
    coords2 = slice2.obsm['spatial']
    offset = (coords1[:,0].max() - coords2[:,0].min()) * 1.1
    coords2_off = coords2 + np.array([offset, 0])

    # mask to cluster if requested
    if cluster is not None:
        mask = (slice1.obs['layer_guess_reordered'] == cluster).to_numpy()
        pi_masked = pi.copy()
        pi_masked[~mask, :] = -1
    else:
        mask = np.ones(len(coords1), bool)
        pi_masked = pi

    # pick top matches
    i_idx, j_idx = largest_indices(pi_masked, top)
    G_masked = np.zeros_like(pi)
    G_masked[i_idx, j_idx] = pi[i_idx, j_idx]

    # draw lines
    mx = pi.max() if pi.size>0 else 1.0
    for i, j in zip(i_idx, j_idx):
        correct = (labels1[i] == labels2[j])
        col = desaturate('blue',1.0) if correct else desaturate('grey',0.6)
        a = alpha*(1-weight_alpha) + (weight_alpha * pi[i,j]/mx)
        ax.plot(
            [coords1[i,0], coords2_off[j,0]],
            [coords1[i,1], coords2_off[j,1]],
            color=col, alpha=a, linewidth=20.0
        )

    # 1) build original cmap
    categories = slice1.obs['layer_guess_reordered'].cat.categories
    orig_colors = slice1.uns['layer_guess_reordered_colors']
    orig_cmap = dict(zip(categories, orig_colors))

    # 2) build new cmap by mapping each original color through `desaturate`
    # new_cmap = { cat: desaturate(col, factor=0.4) 
    #             for cat, col in orig_cmap.items() }
    # cols1 = [new_cmap.get(cat, "#cccccc")
    #      for cat in slice1.obs['layer_guess_reordered']]
    # cols2 = [new_cmap.get(cat, "#cccccc")
    #      for cat in slice2.obs['layer_guess_reordered']]
    domains = slice1.obs['inferred_label']
    cols1 = domains.map(combined_cmap).astype(object) 
    domains = slice2.obs['inferred_label'] 
    cols2 = domains.map(combined_cmap).astype(object) 

    ax.scatter(coords1[:,0], coords1[:,1], s=20, marker='.', color=cols1)
    ax.scatter(coords2_off[:,0], coords2_off[:,1], s=20, marker='.', color=cols2)

    ax.invert_yaxis()
    ax.axis('off')
