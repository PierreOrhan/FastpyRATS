import numpy as np
from scipy.sparse import csr_matrix
from ..util_ import compute_incidence_matrix_in_embedding
from joblib import Parallel, delayed

def compute_tear_graph(
        y,                  # embedding |#points| x embedding dimension
        i_mat,              # incidence matrix |#views| x |#points|
        partition,          # partition matrix |#partion| x |#points| (#partition = #views)
        opts,
        overlap,            # size of overlap between views |#views| x |#views|
        i_mat_in_emb=None,  # incidence matrix in the embedding |#views| x |#points|
        return_views=False
    ):
    k = opts['k']
    nu = opts['nu']
    metric = opts['metric']

    _, n = i_mat.shape
    # If i_mat_in_emb if not provided
    if i_mat_in_emb is None:
        i_mat_in_emb = compute_incidence_matrix_in_embedding(
            y, partition, k, nu, metric
        )

    # compute the tear graph: a graph between partitions/views where ith partition
    # is connected to jth partition if they are across the tear i.e.
    # if the corresponding views are overlapping in the
    # ambient space but not in the embedding space
    views_across_tear_graph = overlap.multiply(i_mat_in_emb.dot(i_mat_in_emb.T))
    views_across_tear_graph = overlap - views_across_tear_graph
    views_across_tear_graph.eliminate_zeros()
     # If no partitions/views are across the tear then there is no tear
    if len(views_across_tear_graph.data) == 0:
        print('No views across the tear detected.')
        return None
    
    print('total #pairs of overlapping partitions/views:', overlap.count_nonzero(), flush=True)
    tear_graph, pts_on_tear, views_cont_pts_across_tear = compute_points_across_tear_graph(
        views_across_tear_graph, i_mat, partition, return_views=return_views, n_batches=opts['n_proc']
    )
    return pts_on_tear, tear_graph, 


def compute_points_across_tear_graph(
    views_across_tear_graph,
    i_mat,
    partition,
    return_views=False,
    n_batches=64
):
    _, n = i_mat.shape
    views_across_tear_graph_row, views_across_tear_graph_col = views_across_tear_graph.nonzero()
    n_pairs =  len(views_across_tear_graph_row)
    print('#pairs of partitions/views across tear:', n_pairs, flush=True)

    # Define per-pair processing function
    def process_pairs(ij_pairs):
        rows = []
        cols = []
        pts = []
        empty_lists = True
        pts_across_tear_exist = []
        for i,j in zip(ij_pairs[0],ij_pairs[1]):
            part_i = partition[i, :]
            part_j = partition[j, :]
            imat_j = i_mat[j, :]
            imat_i = i_mat[i, :]

            T_ij = imat_j.multiply(part_i).nonzero()[1]
            T_ji = imat_i.multiply(part_j).nonzero()[1]
            n_T_ij = len(T_ij)
            n_T_ji = len(T_ji)
            if (n_T_ij == 0) or (n_T_ji == 0):
                pts_across_tear_exist.append(False)
                continue

            pts_across_tear_exist.append(True)
            rows.append(
                np.concatenate([
                    np.repeat(T_ij, n_T_ji),
                    np.repeat(T_ji, n_T_ij)
                ])
            )
            cols.append(
                np.concatenate([
                    np.tile(T_ji, n_T_ij),
                    np.tile(T_ij, n_T_ji)
                ])
            )
            pts.append(np.concatenate([T_ij, T_ji]).tolist())
            empty_lists = False
        if empty_lists:
            temp = np.array([]).astype(int)
            return temp, temp, temp, pts_across_tear_exist
        else:
            return np.concatenate(rows), np.concatenate(cols), np.concatenate(pts), pts_across_tear_exist

    if n_batches < n_pairs:
        chunk_sz = int(n_pairs/n_batches)
        ij_pairs_batches = []
        for i in range(n_batches):
            start_ind = i*chunk_sz
            if i < n_batches:
                end_ind = start_ind+chunk_sz
            else:
                end_ind = n_pairs
            ij_pairs_batches.append((
                views_across_tear_graph_row[start_ind:end_ind],
                views_across_tear_graph_col[start_ind:end_ind]
            ))
    else:
        ij_pairs_batches = [(
            views_across_tear_graph_row,
            views_across_tear_graph_col
        )]

    # Run in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_pairs)(ij_pairs) for ij_pairs in ij_pairs_batches
    )

    # Aggregate results
    tear_graph_row_inds = np.concatenate([r for r, _, _, _ in results if len(r)])
    tear_graph_col_inds = np.concatenate([c for _, c, _, _ in results if len(c)])
    
    pts_on_tear_mask = np.zeros(n, dtype=bool)
    for _, _, pts in results:
        pts_on_tear_mask[pts] = True

    print('Computing tear graph.', flush=True)
    # Build sparse tear graph
    tear_graph = csr_matrix(
        (np.ones(len(tear_graph_row_inds), dtype=bool), 
         (tear_graph_row_inds, tear_graph_col_inds)),
        shape=(n, n),
        dtype=bool
    )
    # Restrict to points on the tear
    pts_on_tear = np.where(pts_on_tear_mask)[0]
    tear_graph = tear_graph[np.ix_(pts_on_tear, pts_on_tear)]

    views_cont_pts_across_tear = None
    if return_views:
        views_cont_pts_across_tear = {}
        for i in range(len(tear_graph_row_inds)):
            edge_i = min(tear_graph_row_inds[i], tear_graph_col_inds[i])
            edge_j = max(tear_graph_row_inds[i], tear_graph_col_inds[i])
            views_cont_pts_across_tear[(edge_i,edge_j)] = []
        
        for j in range(len(ij_pairs_batches)):
            r, c, _, flag = results[j]
            v_r, v_c = ij_pairs_batches[j]
            k = 0
            for i in range(len(flag)):
                if flag[i]:
                    edge_i = min(r[k], c[k])
                    edge_j = max(r[k], c[k])
                    views_cont_pts_across_tear[(edge_i,edge_j)] += [v_r[i],v_c[i]]
                    k += 1

    return tear_graph, pts_on_tear, views_cont_pts_across_tear



# sequential
# def compute_points_across_tear_graph(
#         views_across_tear_graph,
#         i_mat,
#         partition
# ):
#     _, n = i_mat.shape
#     views_across_tear_graph_row, views_across_tear_graph_col = views_across_tear_graph.nonzero()
#     tear_graph_row_inds = []
#     tear_graph_col_inds = []
#     pts_on_tear = np.zeros(n, dtype=bool)
#     print('#pairs of partitions/views across tear:', len(views_across_tear_graph_row), flush=True)
#     # Iterate over pairs of partitions/views that lie across the tear
#     for edge_ind in range(len(views_across_tear_graph_row)):
#         i = views_across_tear_graph_row[edge_ind]
#         j = views_across_tear_graph_col[edge_ind]
#         T_ij = i_mat[j,:].multiply(partition[i,:]).nonzero()[1]
#         T_ji = i_mat[i,:].multiply(partition[j,:]).nonzero()[1]
#         n_T_ij = len(T_ij)
#         n_T_ji = len(T_ji)

#         # if both T_ij and T_ji are non-empty
#         if (n_T_ij*n_T_ji) > 0:
#             pts_on_tear[T_ij] = True
#             pts_on_tear[T_ji] = True
#             tear_graph_row_inds += np.repeat(T_ij, n_T_ji).tolist()
#             tear_graph_col_inds += np.tile(T_ji, n_T_ij).tolist()
#             tear_graph_row_inds += np.repeat(T_ji, n_T_ij).tolist()
#             tear_graph_col_inds += np.tile(T_ij, n_T_ji).tolist()

#     print('Computing tear graph.', flush=True)
#     tear_graph = csr_matrix((np.ones(len(tear_graph_row_inds), dtype=bool),
#                             (tear_graph_row_inds, tear_graph_col_inds)),
#                             shape=(n,n), dtype=bool)
    
#     tear_graph = tear_graph[np.ix_(pts_on_tear,pts_on_tear)]
#     pts_on_tear = np.where(pts_on_tear)[0]
#     return tear_graph, pts_on_tear