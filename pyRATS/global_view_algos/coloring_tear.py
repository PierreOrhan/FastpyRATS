import numpy as np
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
import scipy
from ..util_ import compute_tear_graph

def compute_color_of_pts_on_tear(
        y, i_mat, partition, opts,
        overlap, i_mat_in_emb=None):
    if opts['tear_color_method'] == 'spectral':
        return compute_spectral_color_of_pts_on_tear(y, i_mat, partition, opts, overlap,
                                                     i_mat_in_emb=i_mat_in_emb)
    else:
        raise RuntimeError('tear_color_method=' + opts['tear_color_method'] + 'is not implemented')
        # return compute_color_of_pts_on_tear_heuristic(y, i_mat, partition, opts, overlap,
        #                                               i_mat_in_emb=i_mat_in_emb)

def compute_spectral_color_of_pts_on_tear_util(buml_obj, y, tear_color_eig_inds=None):
    opts = copy.deepcopy(buml_obj.global_opts)
    if tear_color_eig_inds:
        opts['tear_color_eig_inds'] = tear_color_eig_inds
    return compute_spectral_color_of_pts_on_tear(y, buml_obj.IntermedViews.i_mat,
                                                 buml_obj.IntermedViews.partition, opts,
                                                 buml_obj.GlobalViews.overlap)

def compute_spectral_color_of_pts_on_tear(
        y,                  # embedding |#points| x embedding dimension
        i_mat,              # incidence matrix |#views| x |#points|
        partition,          # partition matrix |#partion| x |#points| (#partition = #views)
        opts,
        overlap,            # size of overlap between views |#views| x |#views|
        i_mat_in_emb=None,  # incidence matrix in the embedding |#views| x |#points|
        return_tear_graph_info=False
    ):
    
    pts_across_tear, tear_graph = compute_tear_graph(
        y,
        i_mat,
        partition,
        opts,
        overlap,
        i_mat_in_emb=i_mat_in_emb,
    )
    n_pts_across_tear = len(pts_across_tear)
    n_comp, labels = connected_components(tear_graph, directed=False, return_labels=True)
    print('Number of components in the tear graph:', n_comp, flush=True)

    n_points_in_comp = []
    for i in range(n_comp):
        comp_i = labels==i
        n_points_in_comp.append(np.sum(comp_i))
    

    _, n = i_mat.shape
    tear_color_eig_inds = opts['tear_color_eig_inds']
    max_diversity = np.max(tear_color_eig_inds)+1
    color_of_pts_on_tear = np.zeros((n, max_diversity)) + np.nan
    offset = np.zeros(max_diversity)
    if return_tear_graph_info:
        tear_graph_info = []
    
    # iterate from large to small components in tear graph
    for i in np.flip(np.argsort(n_points_in_comp)).tolist():
        comp_i = labels==i
        n_comp_i = np.sum(comp_i)
        scale = n_comp_i/n_pts_across_tear
        print('#points in the tear component no.', i, 'are:', n_comp_i, flush=True)

        # If the component is very small then assign the constant color
        if n_comp_i <= max(3, int(opts['color_cutoff_frac']*n)):
            if n_comp_i <= int(opts['color_cutoff_frac2']*n):
                continue
            color_of_pts_on_tear[pts_across_tear[comp_i],:] = offset + scale/2
            offset += scale
            continue

        # Construct laplacian on the i-th component of the tear graph
        tear_graph_comp_i = tear_graph[np.ix_(comp_i, comp_i)]
        tear_graph_comp_i = laplacian(tear_graph_comp_i.astype('float'))
        if return_tear_graph_info:
            tear_graph_info.append([i, tear_graph_comp_i]) # NOTE: Laplacian not adjacency

        # Compute color of points on the points in the i-th component of the tear graph
        np.random.seed(42)
        v0 = np.random.uniform(0, 1, tear_graph_comp_i.shape[0])
        n_eigs = min(n_comp_i-1, max_diversity)
        _, colors_ = scipy.sparse.linalg.eigsh(tear_graph_comp_i, v0=v0, k=n_eigs, sigma=-1e-3)
        colors_max = np.max(colors_, axis=0)[None,:]
        colors_min = np.min(colors_, axis=0)[None,:]
        colors_ = (colors_-colors_min)/(colors_max-colors_min + 1e-12) # scale to [0,1]
        colors_ = offset[None,:n_eigs] + colors_*scale
        
        # repeat the last color (max_diversity-n_eigs) times
        if max_diversity-n_eigs > 0:
            colors_ = np.concatenate([colors_, colors_[:,-1]*np.ones((1,max_diversity-n_eigs))], axis=1)

        #color_of_pts_on_tear[np.ix_(pts_on_tear[comp_i],np.arange(n_eigs))] = colors_
        color_of_pts_on_tear[pts_across_tear[comp_i],:] = colors_
        if opts['color_largest_tear_comp_only']:
            break
        
        offset += scale
        
    if return_tear_graph_info:
        return color_of_pts_on_tear, [labels, tear_graph_info]
    else:
        return color_of_pts_on_tear

# def compute_color_of_pts_on_tear_heuristic(y, i_mat, partition, opts,
#                                                 overlap, i_mat_in_emb=None):
#     M,n = i_mat.shape

#     # partitionompute |i_mat_in_emb_{mm'}| if not provided
#     if i_mat_in_emb is None:
#         i_mat_in_emb = compute_incidence_matrix_in_embedding(y, partition, opts['k'], opts['nu'], opts['metric'])

#     color_of_pts_on_tear = np.zeros(n)+np.nan

#     # partitionompute the tear: a graph between views where ith view
#     # is connected to jth view if they are neighbors in the
#     # ambient space but not in the embedding space
#     # tear = i_mat -  i_mat.multiply(i_mat_in_emb)
#     # tear = tear.dot(tear.T)
#     tear = overlap - overlap.multiply(i_mat_in_emb.dot(i_mat_in_emb.T))
#     tear.eliminate_zeros()
#     # Keep track of visited views across partitions of manifolds
#     is_visited = np.zeros(M, dtype=bool)
#     n_visited = 0
#     while n_visited < M: # boundary of a partition remain to be colored
#         # track the next color to assign
#         cur_color = 1

#         s0 = np.argmax(is_visited == 0)
#         seq, rho = breadth_first_order(overlap, s0, directed=False) #(ignores edge weights)
#         is_visited[seq] = True
#         n_visited = np.sum(is_visited)

#         # Iterate over views
#         for m in seq:
#             to_tear_mth_view_with = tear[m,:].nonzero()[1].tolist()
#             if len(to_tear_mth_view_with):
#                 # Points in the overlap of mth view and the views
#                 # on the opposite side of the tear
#                 i_mat_m = i_mat[m,:]
#                 for i in range(len(to_tear_mth_view_with)):
#                     mpp = to_tear_mth_view_with[i]
#                     temp0 = np.isnan(color_of_pts_on_tear)
#                     temp_i = i_mat_m.multiply(i_mat[mpp,:])
#                     temp_m = partition[m,:].multiply(temp_i).multiply(temp0)
#                     temp_mp = partition[mpp,:].multiply(temp_i).multiply(temp0)
#                     if temp_m.sum():
#                         color_of_pts_on_tear[(temp_m).nonzero()[1]] = cur_color
#                         cur_color += 1
#                     if temp_mp.sum():
#                         color_of_pts_on_tear[(temp_mp).nonzero()[1]] = cur_color
#                         cur_color += 1
#     return color_of_pts_on_tear

