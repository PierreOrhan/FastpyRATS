import pdb
import numpy as np
import time

import scipy
from scipy.sparse import csr_matrix, block_diag, bmat
from scipy.sparse.csgraph import dijkstra

import multiprocess as mp
from multiprocess import shared_memory

from sklearn.utils.extmath import svd_flip

def compute_far_off_points(d_e, global_opts, force_compute=False):
    if 'reuse' in global_opts['far_off_points_type'] and (not force_compute):
        return global_opts['far_off_points']

    if global_opts['far_off_points_type'] != 'random':
        np.random.seed(42)
    far_off_points = []
    dist_from_far_off_points = None
    while len(far_off_points) < global_opts['n_repel']:
        if len(far_off_points) == 0:
            far_off_points = []
            dist_from_far_off_points = np.zeros(d_e.shape[0])+np.inf
            while np.sum(np.isinf(dist_from_far_off_points)):
                inf_inds = np.where(np.isinf(dist_from_far_off_points))[0]
                np.random.shuffle(inf_inds)
                far_off_points.append(inf_inds[0])
                dist_from_far_off_points = np.minimum(dist_from_far_off_points,
                                                  dijkstra(d_e, directed=False,
                                                           indices=far_off_points[-1]))
        else:
            far_off_points.append(np.argmax(dist_from_far_off_points))
            dist_from_far_off_points = np.minimum(dist_from_far_off_points,
                                                  dijkstra(d_e, directed=False,
                                                           indices=far_off_points[-1]))
    return far_off_points

def compute_Lpinv_helpers(W):
    M, n = W.shape
    # B_ = W.copy().transpose().astype('int')
    B_ = W.copy().transpose().astype('float')
    D_1 = np.asarray(B_.sum(axis=1))
    D_2 = np.asarray(B_.sum(axis=0))
    D_1_inv_sqrt = np.sqrt(1/D_1)
    D_2_inv_sqrt = np.sqrt(1/D_2)
    B_tilde = B_.multiply(D_2_inv_sqrt).multiply(D_1_inv_sqrt)
    # TODO: U12 is dense of size nxM
    print('Computing svd', flush=True)
    U12,SS,VT = scipy.linalg.svd(B_tilde.todense(), full_matrices=False)
    U12, VT = svd_flip(U12, VT)
    print('Done', flush=True)
    # U12,SS,VT = slinalg.svds(B_tilde, k=M, solver='propack')
    V = VT.T
    mask = np.abs(SS-1)<1e-6
    m_1 = np.sum(mask)
    Sigma = np.expand_dims(SS[m_1:], 1)
    Sigma_1 = 1/(1-Sigma**2)
    Sigma_2 = Sigma*Sigma_1
    U1 = U12[:,:m_1]
    U2 = U12[:,m_1:]
    V1 = V[:,:m_1]
    V2 = V[:,m_1:]
    return [D_1_inv_sqrt, D_2_inv_sqrt, U1, U2, V1, V2, Sigma_1, Sigma_2]

# Ngoc-Diep Ho, Paul Van Dooren, On the pseudo-inverse of the Laplacian of a bipartite graph
def compute_Lpinv_MT(Lpinv_helpers, B):
    D_1_inv_sqrt, D_2_inv_sqrt, U1, U2, V1, V2, Sigma_1, Sigma_2 = Lpinv_helpers
    n = D_1_inv_sqrt.shape[0]
    B_mean = B.mean(axis=1)
    if len(B_mean.shape) == 1:
        B_mean = B_mean[:,None]
    B_n = B - B_mean
    B_n = np.asarray(B_n)
    B1T = D_1_inv_sqrt * (B_n[:, :n].T)
    B2T = D_2_inv_sqrt.T * (B_n[:, n:].T)
    
    U1TB1T = np.matmul(U1.T, B1T)
    U2TB1T = np.matmul(U2.T, B1T)
    V1TB2T = np.matmul(V1.T, B2T)
    V2TB2T = np.matmul(V2.T, B2T)
    
    temp1 = -0.75*np.matmul(U1,U1TB1T)-0.25*np.matmul(U1,V1TB2T) +\
            np.matmul(U2, ((Sigma_1-1))*(U2TB1T)) + np.matmul(U2, Sigma_2*(V2TB2T)) + B1T
    temp1 = temp1 * D_1_inv_sqrt
    
    temp2 = -0.25*np.matmul(V1, U1TB1T) + 0.25*np.matmul(V1,V1TB2T) +\
            np.matmul(V2, Sigma_2*(U2TB1T)) + np.matmul(V2, Sigma_1*(V2TB2T))
    temp2 = temp2 * D_2_inv_sqrt.T 
    
    temp = np.concatenate((temp1, temp2), axis=0)
    temp = temp - np.mean(temp, axis=0, keepdims=True)
    return temp

def compute_CC(D, B, Lpinv_BT):
    CC = D - B.dot(Lpinv_BT)
    return 0.5*(CC + CC.T)

def build_ortho_optim(d, Utilde, intermed_param,
                      far_off_points=[], repel_by=0.,
                      max_var_by=None, beta=None, ret_CCs=False):
    M,n = Utilde.shape
    B_row_inds = []
    B_col_inds = []
    B_vals = []
    D = []
    
    if (beta is not None) and (beta['align'] is not None):
        W_row_inds = []
        W_col_inds = []
        W_vals = []
        for i in range(M):
            Utilde_i = Utilde[i,:].indices
            w_ki = intermed_param.alignment_wts({'view_index': i,
                                                  'data_mask': Utilde_i,
                                                  'beta': beta['align']})
            col_inds = Utilde_i.tolist()
            W_row_inds += [i]*len(col_inds)
            W_col_inds += col_inds
            W_vals += w_ki.tolist()
        W_row_inds = np.array(W_row_inds)
        W_col_inds = np.array(W_col_inds)
        W_vals = np.array(W_vals)
        for k in range(n):
            mask = W_col_inds == k
            temp = W_vals[mask]
            temp = np.exp(temp - temp.max())
            temp *= temp.shape[0]/np.sum(temp)
            W_vals[mask] = temp
        W = csr_matrix((W_vals, (W_row_inds, W_col_inds)), shape=(M,n), dtype=float)
    else:
        W = Utilde.astype(float)
        W_vals = W.data
        
    
    for i in range(M):
        Utilde_i = Utilde[i,:].indices
        X_ = intermed_param.eval_({'view_index': i,
                                   'data_mask': Utilde_i})
        sqrt_p_ki = np.sqrt(np.array(W[i,:].data).flatten()[:,None])
        X_ = sqrt_p_ki * X_
        D.append(np.matmul(X_.T,X_))
        
        row_inds = list(range(d*i,d*(i+1)))
        col_inds = Utilde_i.tolist()
        
        B_row_inds += (row_inds + np.repeat(row_inds, len(col_inds)).tolist())
        B_col_inds += (np.repeat([n+i], d).tolist() + np.tile(col_inds, d).tolist())
        
        X_ = sqrt_p_ki * X_
        B_vals += (np.sum(-X_.T, axis=1).tolist() + X_.T.flatten().tolist())
    
    D = block_diag(D, format='csr')
    B = csr_matrix((B_vals, (B_row_inds, B_col_inds)), shape=(M*d,n+M))
    
    print('min and max weights:', np.array(W_vals).min(), np.array(W_vals).max())

    n_repel = len(far_off_points)
    print('Computing Pseudoinverse of a matrix of L of size', n, '+', M, 'multiplied with B', flush=True)
    Lpinv_helpers = compute_Lpinv_helpers(W)
    
    Lpinv_BT = compute_Lpinv_MT(Lpinv_helpers, B)
    CC = compute_CC(D, B, Lpinv_BT)

    # if max_var_by:
    #     Lpinv_BT_first_n_rows = Lpinv_BT[:n,:]
    #     Lpinv_BT_first_n_rows = Lpinv_BT_first_n_rows - Lpinv_BT_first_n_rows.sum(axis=0)[None,:]/n
    #     CC = CC - max_var_by*Lpinv_BT_first_n_rows.T.dot(Lpinv_BT_first_n_rows)
    
    if n_repel > 0:
    #if (n_repel > 0) and (repel_by>1e-3):
        temp_arr = (-np.ones(n_repel)).tolist()
        L_r = np.zeros((n_repel, n_repel))
        L__row_inds = []
        L__col_inds = []
        L__vals = []
        for i in range(n_repel):
            L__row_inds += [far_off_points[i]]*n_repel
            L__col_inds += far_off_points
            if (beta is not None) and (beta['repel'] is not None):
                p_llp = intermed_param.repulsion_wts({'pt_index': far_off_points[i],
                                                      'repelling_pts_indices': far_off_points,
                                                      'beta': beta['repel']})
                p_llp[i] = 0
                p_llp_sum = np.sum(p_llp)
                p_llp *= -1
                p_llp[i] = p_llp_sum
                L_r[i,:] = p_llp
            else:
                temp_arr[i] = n_repel-1
                L_r[i,:] = temp_arr
                temp_arr[i] = -1
            L__vals += L_r[i,:].tolist()
        
        L_ = csr_matrix((L__vals, (L__row_inds, L__col_inds)), shape=(n+M,n+M))
        L_ = repel_by*L_
        L__Lpinv_BT = L_.dot(Lpinv_BT)
        CC_repel = (Lpinv_BT.T).dot(L__Lpinv_BT)
        CC_net = CC - CC_repel

        ##########
        # B_Omega_row_inds = []
        # B_Omega_col_inds = []
        # B_Omega_vals = []
        # edge_ctr = 0
        # for i in range(n_repel):
        #     for j in range(i+1,n_repel):
        #         B_Omega_col_inds.append(edge_ctr)
        #         B_Omega_row_inds.append(far_off_points[i])
        #         B_Omega_vals.append(-1)
        #         B_Omega_col_inds.append(edge_ctr)
        #         B_Omega_row_inds.append(far_off_points[j])
        #         B_Omega_vals.append(1)
        #         edge_ctr += 1
        # B_Omega = csr_matrix((B_Omega_vals, (B_Omega_row_inds, B_Omega_col_inds)),
        #                      shape=(n+M,edge_ctr))
        # B_OmegaT_Lpinv_BT = B_Omega.T.dot(Lpinv_BT)
        # Lpinv_B_Omega = compute_Lpinv_MT(Lpinv_helpers, B_Omega.T)
        # B_OmegaT_Lpinv_B_Omega = B_Omega.T.dot(Lpinv_B_Omega)
        # CC_repel = scipy.linalg.pinv(np.eye(edge_ctr) + repel_by*B_OmegaT_Lpinv_B_Omega)
        # CC_repel = B_OmegaT_Lpinv_BT.T.dot(CC_repel)
        # CC_repel = B_OmegaT_Lpinv_BT.T.dot(CC_repel.T)
        # CC_repel = repel_by*CC_repel.T
        # CC_net = CC - CC_repel
    else:
        CC_net = CC
        CC_repel = np.eye(CC.shape[0])
        
    if ret_CCs:
        return (CC, CC_repel), Lpinv_BT, D, B
    else:
        return CC_net, Lpinv_BT, D, B

def build_ortho_optim_v2(d, Utilde, intermed_param,
                          far_off_points=[], repel_by=0.,
                          max_var_by=None, beta=None, ret_CCs=False):
    M,n = Utilde.shape
    B_row_inds = []
    B_col_inds = []
    B_vals = []
    D = []
    
    if (beta is not None) and (beta['align'] is not None):
        W_row_inds = []
        W_col_inds = []
        W_vals = []
        for i in range(M):
            Utilde_i = Utilde[i,:].indices
            w_ki = intermed_param.alignment_wts({'view_index': i,
                                                  'data_mask': Utilde_i,
                                                  'beta': beta['align']})
            col_inds = Utilde_i.tolist()
            W_row_inds += [i]*len(col_inds)
            W_col_inds += col_inds
            W_vals += w_ki.tolist()
        W_row_inds = np.array(W_row_inds)
        W_col_inds = np.array(W_col_inds)
        W_vals = np.array(W_vals)
        for k in range(n):
            mask = W_col_inds == k
            temp = W_vals[mask]
            temp = np.exp(temp - temp.max())
            temp *= temp.shape[0]/np.sum(temp)
            W_vals[mask] = temp
        W = csr_matrix((W_vals, (W_row_inds, W_col_inds)), shape=(M,n), dtype=float)
    else:
        W = Utilde.astype(float)
        W_vals = W.data

    L = bmat([[None, W.T],[W, None]])
    
    for i in range(M):
        Utilde_i = Utilde[i,:].indices
        X_ = intermed_param.eval_({'view_index': i,
                                   'data_mask': Utilde_i})
        sqrt_p_ki = np.sqrt(np.array(W[i,:].data).flatten()[:,None])
        X_ = sqrt_p_ki * X_
        D.append(np.matmul(X_.T,X_))
        
        row_inds = list(range(d*i,d*(i+1)))
        col_inds = Utilde_i.tolist()
        
        B_row_inds += (row_inds + np.repeat(row_inds, len(col_inds)).tolist())
        B_col_inds += (np.repeat([n+i], d).tolist() + np.tile(col_inds, d).tolist())
        
        X_ = sqrt_p_ki * X_
        B_vals += (np.sum(-X_.T, axis=1).tolist() + X_.T.flatten().tolist())
    
    D = block_diag(D, format='csr')
    B = csr_matrix((B_vals, (B_row_inds, B_col_inds)), shape=(M*d,n+M))
    
    print('min and max weights:', np.array(W_vals).min(), np.array(W_vals).max())

    n_repel = len(far_off_points)
    if n_repel > 0:
    #if (n_repel > 0) and (repel_by>1e-3):
        temp_arr = (-np.ones(n_repel)).tolist()
        L_r = np.zeros((n_repel, n_repel))
        L__row_inds = []
        L__col_inds = []
        L__vals = []
        for i in range(n_repel):
            L__row_inds += [far_off_points[i]]*n_repel
            L__col_inds += far_off_points
            if (beta is not None) and (beta['repel'] is not None):
                p_llp = intermed_param.repulsion_wts({'pt_index': far_off_points[i],
                                                      'repelling_pts_indices': far_off_points,
                                                      'beta': beta['repel']})
                p_llp[i] = 0
                p_llp_sum = np.sum(p_llp)
                p_llp *= -1
                p_llp[i] = p_llp_sum
                L_r[i,:] = p_llp
            else:
                temp_arr[i] = n_repel-1
                L_r[i,:] = temp_arr
                temp_arr[i] = -1
            L__vals += L_r[i,:].tolist()
        
        L_ = csr_matrix((L__vals, (L__row_inds, L__col_inds)), shape=(n+M,n+M))
        L_ = repel_by*L_
        L = L + L_   
    return D, B, L.tocsc()
    
# unscaled alignment error
def compute_alignment_err(d, Utilde, intermed_param, scale_num, far_off_points=[], repel_by=0., beta=None):
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=far_off_points,
                                     repel_by=repel_by, beta=beta)
    M,n = Utilde.shape
    
    ## Check if C is pd or psd
    #np.random.seed(42)
    #v0 = np.random.uniform(0,1,CC.shape[0])
    #sigma_min_C = slinalg.eigsh(CC, k=1, v0=v0, which='SM',return_eigenvectors=False)
    #print('Smallest singular value of C', sigma_min_C, flush=True)
    
    CC_mask = np.tile(np.eye(d, dtype=bool), (M,M))
    #scale_denom = Utilde.sum()
    #scale = (scale_num/scale_denom)
    scale = 1
    err = np.sum(CC[CC_mask]) * scale
    return err

# Kunal N Chaudhury, Yuehaw Khoo, and Amit Singer, Global registration
# of multiple point clouds using semidefinite programming
def spectral_alignment(y, d, Utilde,
                      C, intermed_param, global_opts, 
                      seq_of_intermed_views_in_cluster = None,
                      affine_transform=False):
    if seq_of_intermed_views_in_cluster is None:
        seq_of_intermed_views_in_cluster = [np.arange(Utilde.shape[0])]
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=global_opts['far_off_points'],
                                     repel_by=global_opts['repel_by'], max_var_by=global_opts['max_var_by'], 
                                     beta=global_opts['beta'])
        
    M,n = Utilde.shape
    n_clusters = len(seq_of_intermed_views_in_cluster)
    CC0 = np.zeros((M*d,d))
    # for s in range(M):
    #     CC0[s*d:(s+1)*d,:] = intermed_param.T[s,:,:]
    for s in range(M):
        CC0[s*d:(s+1)*d,:] = np.eye(d)
    CC0 = CC0/np.sqrt(M)
    v0 = CC0[:,0]
    
    print('Computing eigh(C,k=d)', flush=True)
    np.random.seed(42)
    v0 = np.random.uniform(0,1,CC.shape[0])
    if (global_opts['n_repel'] == 0) or (global_opts['repel_by'] == 0) or (global_opts['max_var_by'] == 0):
        # To find smallest eigenvalues, using shift-inverted algo with mode=normal and which='LM'
        W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, sigma=-1e-3)
        # W_,V_ = scipy.sparse.linalg.lobpcg(CC, CC0.T, largest=False)
        # or just pass which='SM' without using sigma
        #W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, which='SM')
        V_, _ = svd_flip(V_, V_.T)
    else:
        # To find smallest eigenvalues, using shift-inverted algo with mode=normal and which='LM'
        CC_frob = np.linalg.norm(CC)
        W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, sigma=-2*CC_frob)
        # or just pass which='SM' without using sigma
        # W_,V_ = scipy.sparse.linalg.eigsh(CC, k=d, v0=v0, which='SA')
        V_, _ = svd_flip(V_, V_.T)
    print('Done.', flush=True)
    
    if affine_transform:
        Tstar = V_.T
    else:
        Wstar = np.sqrt(M)*V_.T
        Tstar = np.zeros((d, M*d))
        for i in range(n_clusters):
            # the first view in each cluster is not rotated
            seq = seq_of_intermed_views_in_cluster[i]
            s0 = seq[0]
            U_,S_,VT_ = scipy.linalg.svd(Wstar[:,d*s0:d*(s0+1)])
            U_, VT_ = svd_flip(U_, VT_)
            Q =  np.matmul(U_,VT_)
            if (global_opts['init_algo_name'] != 'spectral') and (np.linalg.det(Q) < 0): # remove reflection
                VT_[-1,:] *= -1
                Q = np.matmul(U_, VT_)
            Q = Q.T
            
            for m_ in range(len(seq)):
                m = seq[m_]
                U_,S_,VT_ = scipy.linalg.svd(Wstar[:,d*m:d*(m+1)])
                temp_ = np.matmul(U_,VT_)
                if (global_opts['init_algo_name'] != 'spectral') and (np.linalg.det(temp_) < 0): # remove reflection
                    VT_[-1,:] *= -1
                    temp_ = np.matmul(U_, VT_)
                Tstar[:,m*d:(m+1)*d] = np.matmul(Q, temp_)
    
    Zstar = Tstar.dot(Lpinv_BT.transpose())
    
    # the first view in each cluster is not translated
    for i in range(n_clusters):
        seq = seq_of_intermed_views_in_cluster[i]
        s0 = seq[0]
        temp = Zstar[:,n+s0][:,None].copy()
        Zstar[:,n+seq] -= temp
        C_seq = C[seq,:].sum(axis=0).nonzero()[1]
        Zstar[:,C_seq] -= temp
        
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, Zstar[:,:n].T

def rgd_alignment(y, d, Utilde, C, intermed_param, global_opts):
    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, intermed_param,
                                     far_off_points=global_opts['far_off_points'],
                                     repel_by=global_opts['repel_by'], max_var_by=global_opts['max_var_by'], 
                                     beta=global_opts['beta'])
    M,n = Utilde.shape
    n_proc = min(M,global_opts['n_proc'])
    barrier = mp.Barrier(n_proc)

    def update(alpha, max_iter, shm_name_O, O_shape, O_dtype,
               shm_name_CC, CC_shape, CC_dtype, barrier):
        ###########################################
        # Parallel Updates
        ###########################################
        def target_proc(p_num, chunk_sz, barrier):
            existing_shm_O = shared_memory.SharedMemory(name=shm_name_O)
            O = np.ndarray(O_shape, dtype=O_dtype, buffer=existing_shm_O.buf)
            existing_shm_CC = shared_memory.SharedMemory(name=shm_name_CC)
            CC = np.ndarray(CC_shape, dtype=CC_dtype, buffer=existing_shm_CC.buf)

            def unique_qr(A):
                Q, R = np.linalg.qr(A)
                signs = 2 * (np.diag(R) >= 0) - 1
                Q = Q * signs[np.newaxis, :]
                R = R * signs[:, np.newaxis]
                return Q, R
            
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = M
            else:
                end_ind = (p_num+1)*chunk_sz
            for _ in range(max_iter):
                O_copy = O.copy()
                barrier.wait()
                for i in range(start_ind, end_ind):
                    xi_ = 2*np.matmul(O_copy, CC[:,i*d:(i+1)*d])
                    temp0 = O[:,i*d:(i+1)*d]
                    temp1 = np.matmul(xi_,temp0.T)
                    skew_temp1 = 0.5*(temp1-temp1.T)
                    Q_,R_ = unique_qr(temp0 - alpha*np.matmul(skew_temp1,temp0))
                    O[:,i*d:(i+1)*d] = Q_
                barrier.wait()
            
            existing_shm_O.close()
            existing_shm_CC.close()
        
        
        proc = []
        chunk_sz = int(M/n_proc)
        for p_num in range(n_proc):
            proc.append(mp.Process(target=target_proc,
                                   args=(p_num,chunk_sz, barrier),
                                   daemon=True))
            proc[-1].start()

        for p_num in range(n_proc):
            proc[p_num].join()
        ###########################################
        
        # Sequential version of above
        # for i in range(M):
        #     temp0 = O[:,i*d:(i+1)*d]
        #     temp1 = skew(np.matmul(xi[:,i*d:(i+1)*d],temp0.T))
        #     Q_,R_ = unique_qr(temp0 - t*np.matmul(temp1,temp0))
        #     O[:,i*d:(i+1)*d] = Q_

    alpha = global_opts['alpha']
    max_iter = global_opts['max_internal_iter']
    Tstar = np.zeros((d,M*d))
    for s in range(M):
        Tstar[:,s*d:(s+1)*d] = np.eye(d)
    
    print('Descent starts', flush=True)
    shm_Tstar = shared_memory.SharedMemory(create=True, size=Tstar.nbytes)
    np_Tstar = np.ndarray(Tstar.shape, dtype=Tstar.dtype, buffer=shm_Tstar.buf)
    np_Tstar[:] = Tstar[:]
    shm_CC = shared_memory.SharedMemory(create=True, size=CC.nbytes)
    np_CC = np.ndarray(CC.shape, dtype=CC.dtype, buffer=shm_CC.buf)
    np_CC[:] = CC[:]
    
    update(alpha, max_iter, shm_Tstar.name, Tstar.shape, Tstar.dtype,
           shm_CC.name, CC.shape, CC.dtype, barrier)
    
    Tstar[:] = np_Tstar[:]
    
    del np_Tstar
    shm_Tstar.close()
    shm_Tstar.unlink()
    del np_CC
    shm_CC.close()
    shm_CC.unlink()
    
    Zstar = Tstar.dot(Lpinv_BT.transpose())
    
    for s in range(M):
        T_s = Tstar[:,s*d:(s+1)*d].T
        v_s = Zstar[:,n+s]
        intermed_param.T[s,:,:] = np.matmul(intermed_param.T[s,:,:], T_s)
        intermed_param.v[s,:] = np.matmul(intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
        C_s = C[s,:].indices
        y[C_s,:] = intermed_param.eval_({'view_index': s, 'data_mask': C_s})
    return y, Zstar[:,:n].T