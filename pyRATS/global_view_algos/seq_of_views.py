import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, breadth_first_order, shortest_path
from ..util_ import center_of_tree, lexargmax, compute_farthest_points
from scipy.sparse import csr_matrix, triu
from scipy.linalg import svdvals

from joblib import Parallel, delayed

def compute_seq_of_views(
        d,          # embedding dimension
        i_mat,      # incidence matrix |#views| x |#points|
        n_C,
        overlap,    # size of overlap between views |#views| x |#views|
        param,      # d-dimensional parameterization of points in each view
        opts,
        tol=1e-6
    ):
    n_views = i_mat.shape[0]
    n_proc = opts['n_proc']

    # W_{mm'} = W_{m'm} measures the ambiguity between
    # the two embeddings of the points on the overlap
    # between mth and m'th intermediate views
    W_rows, W_cols = triu(overlap).nonzero()
    n_elem = W_rows.shape[0]
    overlap_svals = np.zeros((n_elem,d))
    chunk_sz = int(n_elem/n_proc)

    def target_proc(p_num):
        start_ind = p_num*chunk_sz
        if p_num == (n_proc-1):
            end_ind = n_elem
        else:
            end_ind = (p_num+1)*chunk_sz
        overlap_svals_ = np.zeros((end_ind-start_ind,d))
        for i in range(start_ind, end_ind):
            m = W_rows[i]
            mpp = W_cols[i]
            mask = i_mat[m,:].multiply(i_mat[mpp,:]).nonzero()[1]
            V_mmp = param.eval_({'view_index': m, 'data_mask': mask})
            V_mpm = param.eval_({'view_index': mpp, 'data_mask': mask})
            Vbar_mmp = V_mmp - np.mean(V_mmp,0)[np.newaxis,:]
            Vbar_mpm = V_mpm - np.mean(V_mpm,0)[np.newaxis,:]
            # Compute ambiguity of the overlaps captured by singular values
            svdvals_ = svdvals(np.dot(Vbar_mmp.T,Vbar_mpm))
            #overlap_svals_[i-start_ind] = svdvals_[-1]
            overlap_svals_[i-start_ind,:] = svdvals_ 
        return (start_ind, end_ind, overlap_svals_)
    
    res = Parallel(n_jobs=opts['n_proc'], return_as="generator_unordered")(
        delayed(target_proc)(p_num) for p_num in range(n_proc)
    )
    for value in res:
        start_ind, end_ind, overlap_svals_ = value
        overlap_svals[start_ind:end_ind,:] = overlap_svals_
    
    #overlap_svals[overlap_svals<tol] = 0
    W_data = overlap_svals[:,-1]
    for i in range(overlap_svals.shape[1]-2,-1,-1):
        mask = W_data==0
        print('Iter:', i, ':: Updating scores of', np.sum(mask),
              'edges out of', W_data.shape[0], 'edges.')
        n_zero_elem = np.sum(mask)
        if n_zero_elem == 0:
            break
        temp = overlap_svals[mask,i]
        if n_zero_elem < len(W_data):
            temp2 = 0.5*W_data[~mask]
            W_data[mask] = temp*(np.min(temp2)/(np.max(temp)+1e-12))
        else:
            W_data[mask] = temp
    
    W = csr_matrix((W_data, (W_rows, W_cols)),
                   shape=(n_views, n_views)) # strict upper triangular
    W = W + W.T
    
    if opts['tree'] == 'spt':
        _, min_max = compute_farthest_points(W > 0, n_points=100, return_min_max=True)
        W_ = W.copy()
        W_.data = 1/W_.data
        _, pred = shortest_path(W_, return_predecessors=True, directed=False, indices=[min_max])
        pred = pred[0,:]
        M = len(pred)
        row = []
        col = []
        data = []

        for child, parent in enumerate(pred):
            if parent >= 0:
                row.append(parent)
                col.append(child)
                data.append(1)
        T = csr_matrix((data, (row, col)), shape=(M, M))
    else: #mst
        # Compute maximum spanning tree/forest of W
        T = minimum_spanning_tree(-W)

    n_comp, comp_labels = connected_components(T, directed=False,
                                            return_labels=True)
    
    # Remove edges to force clusters if desired
    if opts['n_forced_clusters'] > n_comp:
        inds = np.argsort(T.data)[-(opts['n_forced_clusters']-n_comp):]
        T.data[inds] = 0
        T.eliminate_zeros()
        n_comp, comp_labels = connected_components(T, directed=False, return_labels=True)

    print('No. of connected components (manifolds):', n_comp)
        
    # Create a sequence of views for each cluster representing a manifold 
    seq_of_views_in_cluster = []
    parents_of_views_in_cluster = []
    cluster_of_view = np.zeros(n_views,dtype=int)

    if opts['root_view'] == 'center':
        for i in range(n_comp):
            views_in_this_comp = np.where(comp_labels==i)[0]
            T_i = T[views_in_this_comp,:][:,views_in_this_comp]
            center_i = center_of_tree(T_i)
            #center_i = np.argmax(n_C[views_in_this_comp])
            center_i = views_in_this_comp[center_i]
            seq, rho = breadth_first_order(T, center_i, directed=False) #(ignores edge weights)
            seq_of_views_in_cluster.append(seq)
            parents_of_views_in_cluster.append(rho)
            cluster_of_view[seq] = i
    else:
        n_visited = 0
        is_visited = np.zeros(n_views, dtype=bool)
        cluster_num = 0
        rank_arr = np.zeros((n_views,3))
        rank_arr[:,1] = n_C
        np.random.seed(42)
        rank_arr[:,2] = -param.zeta
        #rank_arr[:,2] = np.random.uniform(0, 1, n_views)
        while n_visited < n_views:
            # First intermediate view in the sequence
            rank_arr[:,0] = 1-is_visited
            s_1 = lexargmax(rank_arr)
            # Compute breadth first order in T starting from s_1
            s_, rho_ = breadth_first_order(T, s_1, directed=False) #(ignores edge weights)
            seq_of_views_in_cluster.append(s_)
            parents_of_views_in_cluster.append(rho_)
            is_visited[s_] = True
            cluster_of_view[s_] = cluster_num
            n_visited = np.sum(is_visited)
            cluster_num = cluster_num + 1
    return seq_of_views_in_cluster, parents_of_views_in_cluster, cluster_of_view