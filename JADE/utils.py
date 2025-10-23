import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ot


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    # # Define an R command to install the package if it's not already installed
    # install_mclust = '''
    # if (!require("mclust")) {
    #     install.packages("mclust", repos="http://cran.r-project.org")
    # }
    # '''

    # # Execute the R command from within Python
    # robjects.r(install_mclust)

    # Now load the mclust library
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata



def combined_clustering_general(adatalist, alignments, reduced_dim, n_clusters=7, radius=50, used_obsm='emb_pca', key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False, refine_option="single"):
    print(f"using key={key}")
    combined_adata = ad.concat(adatalist)
    pca = PCA(n_components=reduced_dim, random_state=2025) 
    embedding = pca.fit_transform(combined_adata.obsm[key].copy())
    combined_adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
        combined_adata = mclust_R(combined_adata, used_obsm=used_obsm, num_cluster=n_clusters)
        n_points = [adata.shape[0] for adata in adatalist]
        cum_indices = [0] + np.cumsum(n_points).tolist()
        for i, adata in enumerate(adatalist):
            start_idx = cum_indices[i]
            end_idx = cum_indices[i+1]
            adata.obs['mclust'] = combined_adata.obs['mclust'][start_idx:end_idx].copy()
            adata.obs['domain'] = adata.obs['mclust']
        if refinement:
            if refine_option == "single":
                for adata in adatalist:
                    new_type = refine_label(adata, radius, key='domain')
                    adata.obs['domain'] = new_type 
            
            elif refine_option == "mulitiple":
                # Refine using multiple slices.
                new_type = refine_label_multiple_pairwise(adatalist, alignments, radius, key='domain')
                for k, adata in enumerate(adatalist):
                    adata.obs['domain'] = new_type[k]
            
            else:
                print("refine_option not specified or not reconized!")



def clustering(adata, reduced_dim=20, n_clusters=7, radius=50, used_obsm='emb_pca', key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    
    pca = PCA(n_components=reduced_dim, random_state=2025) 
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding
    # print(f"using key={key}, var explained:{np.cumsum(pca.explained_variance_ratio_)}")
    
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm=used_obsm, num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
    
    
       
def refine_label_multiple_pairwise(adatalist, alignments, radius=50, key='label'):
    n_neigh = radius
    counts_all = []
    
    rev_map = {key:k for k, key in enumerate(set().union(*(adata.obs[key].values for adata in adatalist))
)}
    print(rev_map)
    for adata in adatalist:
        counts = []
        old_type = adata.obs[key].values
        
        #calculate distance
        position = adata.obsm['spatial']
        distance = ot.dist(position, position, metric='euclidean')
            
        n_cell = distance.shape[0]
        
        for i in range(n_cell):
            count_i = np.zeros(len(rev_map))
            vec  = distance[i, :]
            index = vec.argsort()
            neigh_type = []
            for j in range(1, n_neigh+1):
                neigh_type.append(old_type[index[j]])
                count_i[rev_map[old_type[index[j]]]] += 1
            counts.append(count_i)
        counts_all.append(np.array(counts))
        
        ## debug
        print(counts_all[0][:5,:])
    
    newtype = []
    for k, adata in enumerate(adatalist):
        if k == 0:
            # NOTE: only involve self + next
            alignment = alignments[0]
            alignment /= alignment.sum(axis=1, keepdims=True) # n0 times n1
            counts = counts_all[0] # n0 times nclust
            counts_next = counts_all[1] # n1 times nclust
            counts_final = counts + alignment @ counts_next # n0 times nclust
            ## debug
            print(counts_final[:5,:])
            newtype_k = np.argmax(counts_final,axis=1)
            newtype.append(newtype_k)
        if k == len(adatalist)-1:
            # NOTE: only involve self+prev
            alignment = alignments[-1].T
            alignment /= alignment.sum(axis=1, keepdims=True)
            counts = counts_all[-1] 
            counts_prev = counts_all[-2] 
            counts_final = counts + alignment @ counts_prev 
            newtype_k = np.argmax(counts_final,axis=1)
            newtype.append(newtype_k)
    return newtype
     
                
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type



def extract_top_value(map_matrix, retain_percent = 0.1): 
    '''\
    Filter out cells with low mapping probability

    Parameters
    ----------
    map_matrix : array
        Mapped matrix with m spots and n cells.
    retain_percent : float, optional
        The percentage of cells to retain. The default is 0.1.

    Returns
    -------
    output : array
        Filtered mapped matrix.

    '''

    #retain top 1% values for each spot
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    
    return output 



def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    #res = mat.sum()
    return mat



def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    '''\
    Project cell types onto ST data using mapped matrix in adata.obsm

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    adata_sc : anndata
        AnnData object of scRNA-seq reference data.
    retrain_percent: float    
        The percentage of cells to retain. The default is 0.1.
    Returns
    -------
    None.

    '''
    
    # read map matrix 
    map_matrix = adata.obsm['map_matrix']   # spot x cell
   
    # extract top-k values for each spot
    map_matrix = extract_top_value(map_matrix) # filtering by spot
    
    # construct cell type matrix
    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values
       
    # projection by spot-level
    matrix_projection = map_matrix.dot(matrix_cell_type)
   
    # rename cell types
    cell_type = list(adata_sc.obs['cell_type'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    #cell_type = [s.replace(' ', '_') for s in cell_type]
    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)  # spot x cell type
    
    #normalize by row (spot)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

    #add projection results to adata
    adata.obs[df_projection.columns] = df_projection
  
  
    
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    



def cal_pairwise_alignment_score(adata1, adata2, pi):
    fields = set(adata1.obs['layer_guess_reordered'])
    field_index_A = []
    field_index_B = []
    for field in fields:
        field_index_A.append(np.where(np.array(adata1.obs['layer_guess_reordered']==field))[0])
        field_index_B.append(np.where(np.array(adata2.obs['layer_guess_reordered']==field))[0])
    pi = pi/pi.sum() # re-normalize pi to have sum 1.
    s = 0
    for index_A, index_B in zip(field_index_A, field_index_B):
        s += pi[np.ix_(index_A, index_B)].sum()
    return s



def add_spot_cell_assignment(adata, nspots):
  # 1. Extract spatial coordinates from adata.obsm.
	# adata.obsm["spatial"] should be an (n, 2) matrix, where n is the number of cells.
	spatial_coords = adata.obsm["spatial"]  # shape: (n, 2)

	# 2. Cluster the spatial coordinates using KMeans.
	# Set the desired number of clusters (spots); for example, 100.
	n_clusters = nspots  
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	cluster_labels = kmeans.fit_predict(spatial_coords)  # shape: (n,)
	spot_centroids = kmeans.cluster_centers_  # shape: (n_clusters, 2)

	# 3. Create cell-to-spot assignment as a list of lists.
	# Each element in the list corresponds to a spot and contains the indices of cells in that spot.
	spot_cell_assignment = []
	for spot in range(n_clusters):
		# np.where returns a tuple; take the first element and convert to list.
		cell_indices = np.where(cluster_labels == spot)[0].tolist()
		spot_cell_assignment.append(cell_indices)
	# print(np.array(spot_cell_assignment))
	adata.uns['spot_cell_assignment'] = spot_cell_assignment
	adata.uns['spot_spatial'] = spot_centroids
 
	# 4. add spot attribution.
	adata.uns['layer_guess_reordered_spot'] = []
	for j in range(nspots):
		tmp = pd.Categorical(adata.obs['layer_guess_reordered'].iloc[spot_cell_assignment[j]]).as_ordered().max()
		adata.uns['layer_guess_reordered_spot'].append(tmp)
		
	return adata



def plot_adata_spotwise(adata):
    # Get cell spatial coordinates: shape (n_cells, 2)
    cell_coords = adata.obsm["spatial"]

    # Get spot spatial locations from adata.uns
    spot_coords = adata.uns['spot_spatial']  # assumed shape (n_spots, 2)

    # Optionally, get spot assignment for additional visualization
    spot_cell_assignment = adata.uns['spot_cell_assignment']

    plt.figure(figsize=(4,4))
    # Plot cell spatial coordinates as small dots.
    plt.scatter(cell_coords[:, 0], cell_coords[:, 1],
                c='gray', s=5, label='Spots', alpha=0.6)

    # Plot spot centroids as larger red markers.
    cmap = plt.get_cmap("tab10")
    layer_to_color_map = {label:mcolors.to_hex(cmap(i)) for i, label in enumerate(set(adata.uns['layer_guess_reordered_spot']))}
    plt.scatter(
        spot_coords[:, 0], 
        spot_coords[:, 1],
        c=pd.Series(adata.uns['layer_guess_reordered_spot']).map(layer_to_color_map),  # <-- pass color array via c
        s=20, 
        marker='X', 
        label='HyperSpots'
    )

    plt.title("Spot and HyperSpot Spatial Locations")
    plt.xlabel("Spatial X")
    plt.ylabel("Spatial Y")
    plt.legend()
    plt.show()



def multiple_row_col_renormalizations(pi,niter=4):
    # return a row normalized matrix.
    for iter in range(niter):
        col_sums = pi.sum(axis=0)
        pi_col_normalized = pi / col_sums[np.newaxis, :]
        
        row_sums = pi_col_normalized.sum(axis=1)               # shape: (n,)
        pi = pi_col_normalized / row_sums[:, np.newaxis]
        
    return pi



def multiple_mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', seeds=[2020,2021,2022,2023,2024]):
    """\
    Clustering using the mclust algorithm, with multiple EM starts.
    Tries each seed in `seeds`, then selects the classification with best BIC.
    """
    import numpy as np
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    # load library and activate numpy conversion
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    Mclust = robjects.r['Mclust']
    set_seed = robjects.r['set.seed']

    X = rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm])

    best_bic = -np.inf
    best_class = None

    for seed in seeds:
        # set R’s RNG
        set_seed(int(seed))
        # fit the model
        res = Mclust(X, num_cluster, modelNames)
        # extract BIC and classification
        bic = np.array(res.rx2("bic"))  
        if bic > best_bic:
            best_bic = bic
            best_class = np.array(res.rx2("classification"))

    # store best result in adata
    adata.obs['mclust'] = best_class.astype("int")
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

