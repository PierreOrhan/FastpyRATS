import pdb
import numpy as np
from scipy.sparse import csr_matrix, vstack

from ..util_ import procrustes, sparse_matrix, nearest_neighbors

# # Computes Z_s for the case when to_tear is True.
# # Input Z_s is the Z_s for the case when to_tear is False.
# # Output Z_s is a subset of input Z_s.
# def compute_Z_s_to_tear(y, s, Z_s, C, c, k, metric='euclidean'):
#     n_Z_s = Z_s.shape[0]
#     # C_s_U_C_Z_s = (self.C[s,:]) | np.isin(self.c, Z_s)
#     C_s_U_C_Z_s = np.where(C[s,:] + C[Z_s,:].sum(axis=0))[1]
#     n_ = C_s_U_C_Z_s.shape[0]
#     k_ = min(k,n_-1)
#     _, neigh_ind_ = nearest_neighbors(y[C_s_U_C_Z_s,:], k_, metric)
#     U_ = sparse_matrix(neigh_ind_, np.ones(neigh_ind_.shape, dtype=bool))
#     Utilde_ = C[np.ix_(Z_s,C_s_U_C_Z_s)].dot(U_)
#     Utilde_ = vstack([Utilde_, C[s,C_s_U_C_Z_s].dot(U_)])
#     n_Utildeg_Utilde_ = Utilde_.dot(Utilde_.T) 
#     n_Utildeg_Utilde_.setdiag(False)
#     return Z_s[n_Utildeg_Utilde_[-1,:-1].nonzero()[1]].tolist()

def procrustes_init(seq, rho, y, is_visited_view, d, Utilde, n_Utilde_Utilde,
                    C, c, intermed_param, global_opts, print_freq=1000):   
    n = Utilde.shape[1]
    # Traverse views from 2nd view
    for m in range(1,seq.shape[0]):
        if print_freq and np.mod(m, print_freq)==0:
            print('Initial alignment of %d views completed' % m, flush=True)
        s = seq[m]
        # pth view is the parent of sth view
        p = rho[s]
        Utilde_s = Utilde[s,:]

        # If to tear apart closed manifolds
        if global_opts['to_tear']:
            if global_opts['align_w_parent_only']:
                Z_s = [p]
            else:
                raise RuntimeError('global_opts[align_w_parent_only]=False is not implemented')
                # # Compute T_s and v_s by aligning
                # # the embedding of the overlap Utilde_{sp}
                # # due to sth view with that of the pth view
                # Utilde_s_p = Utilde_s.multiply(Utilde[p,:]).nonzero()[1]
                # V_s_p = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s_p})
                # V_p_s = intermed_param.eval_({'view_index': p, 'data_mask': Utilde_s_p})
                # intermed_param.T[s,:,:], intermed_param.v[s,:] = procrustes(V_s_p, V_p_s)
                
                # # Compute temporary global embedding of point in sth cluster
                # C_s = C[s,:].indices
                # y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
                # # Find more views to align sth view with
                # Z_s = n_Utilde_Utilde[s,:].multiply(is_visited_view)
                # Z_s_all = Z_s.nonzero()[1]
                # Z_s = compute_Z_s_to_tear(y, s, Z_s_all, C, c, global_opts['k'], global_opts['metric'])
                # # The parent must be in Z_s
                # if p not in Z_s:
                #     Z_s.append(p)
        # otherwise
        else:
            if global_opts['align_w_parent_only']:
                Z_s = [p]
            else:
                # Align sth view with all the views which have
                # an overlap with sth view in the ambient space
                Z_s = n_Utilde_Utilde[s,:].multiply(is_visited_view)
                Z_s = Z_s.nonzero()[1].tolist()
                # If for some reason Z_s is empty
                if len(Z_s)==0:
                    Z_s = [p]
                
        # Compute centroid mu_s
        # n_Utilde_s_Z_s[k] = #views in Z_s which contain
        # kth point if kth point is in the sth view, else zero
        n_Utilde_s_Z_s = np.zeros(n, dtype=int)
        mu_s = np.zeros((n,d))
        cov_s = csr_matrix((1,n), dtype=bool)
        for mp in Z_s:
            Utilde_s_mp = Utilde_s.multiply(Utilde[mp,:]).nonzero()[1]    
            n_Utilde_s_Z_s[Utilde_s_mp] += 1
            mu_s[Utilde_s_mp,:] += intermed_param.eval_({'view_index': mp,
                                                         'data_mask': Utilde_s_mp})

        # Compute T_s and v_s by aligning the embedding of the overlap
        # between sth view and the views in Z_s, with the centroid mu_s
        temp = n_Utilde_s_Z_s > 0
        mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]
        V_s_Z_s = intermed_param.eval_({'view_index': s, 'data_mask': temp})

        T_s, v_s = procrustes(V_s_Z_s, mu_s)

        # Update T_s, v_
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s

        # Mark sth view as visited
        is_visited_view[s] = True

        # Compute global embedding of point in sth cluster
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, is_visited_view

def procrustes_final(y, d, Utilde, C, intermed_param,
                     seq_of_intermed_views_in_cluster, global_opts):
    M,n = Utilde.shape
    # Traverse over intermediate views in a random order
    seq = np.random.permutation(M)
    is_first_view_in_cluster = np.zeros(M, dtype=bool)
    # is_first_view_in_cluster[i] = True if the ith view is the first
    # view in some cluster of views
    for i in range(len(seq_of_intermed_views_in_cluster)):
        is_first_view_in_cluster[seq_of_intermed_views_in_cluster[i][0]] = True
        
    y_due_to_all_views = []
    for k in range(n):
        y_due_to_all_views.append({})
        
    for s in range(M):
        Utilde_s = Utilde[s,:].nonzero()[1]
        y_Utilde_s = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s})
        for k in range(len(Utilde_s)):
            y_due_to_all_views[Utilde_s[k]][s] = y_Utilde_s[k,:]

    # For a given seq, refine the global embedding
    for it1 in range(global_opts['max_internal_iter']):
        for s in seq.tolist():
            # # Never refine s_0th intermediate view
            # if is_first_view_in_cluster[s]:
            #     continue

            Utilde_s = Utilde[s,:].nonzero()[1]
            mu_s = []
            Utilde_s_ = []
            for k_ in range(len(Utilde_s)):
                k = Utilde_s[k_]
                if len(y_due_to_all_views[k]) == 1:
                    continue
                Utilde_s_.append(k)
                mu = np.array(list(y_due_to_all_views[k].values())).sum(axis=0)
                mu = (mu - y_due_to_all_views[k][s])/(len(y_due_to_all_views[k])-1)
                mu_s.append(mu)
            mu_s = np.array(mu_s)
            Utilde_s_ = np.array(Utilde_s_)

            # Compute T_s and v_s by aligning the embedding of the overlap
            # between sth view and the views in Z_s, with the centroid mu_s
            V_s_Z_s = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s_})
            T_s, v_s = procrustes(V_s_Z_s, mu_s)

            # Update T_s, v_s
            intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
            intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s

            y_Utilde_s = intermed_param.eval_({'view_index': s, 'data_mask': Utilde_s})
            for k in range(len(Utilde_s)):
                y_due_to_all_views[Utilde_s[k]][s] = y_Utilde_s[k,:]
    
    y = np.zeros((n,d))
    for k in range(n):
        y[k,:] = np.array(list(y_due_to_all_views[k].values())).mean(axis=0)
    return y