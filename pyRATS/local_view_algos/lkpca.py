import numpy as np
import multiprocess as mp

from sklearn.decomposition import KernelPCA

from ..common_ import *
from ..util_ import Param

def lkpca(d, X, nbrhd_graph, opts, verbose=False, print_prop = 0.25):
    n, p = X.shape
    print_freq = int(print_prop * n)
    
    local_param = Param('LKPCA')
    local_param.X = X
    local_param.model = np.empty(n, dtype=object)
    local_param.zeta = np.zeros(n)

    n_proc = opts['n_proc']
    lkpca_kernel = opts['lkpca_kernel']
    lkpca_fit_inverse_transform = opts['lkpca_fit_inverse_transform']
    
    def target_proc(p_num, chunk_sz, q_):
        start_ind = p_num*chunk_sz
        if p_num == (n_proc-1):
            end_ind = n
        else:
            end_ind = (p_num+1)*chunk_sz

        model_ = np.empty(end_ind-start_ind, dtype=object)
        for k in range(start_ind, end_ind):
            model_[k-start_ind] = KernelPCA(n_components=d, kernel=lkpca_kernel,
                                             fit_inverse_transform=lkpca_fit_inverse_transform,
                                             eigen_solver='arpack',
                                             random_state=42)
            U_k = nbrhd_graph.get_nbr_inds(k)
            X_k = X[U_k,:]
            model_[k-start_ind].fit(X_k)
        q_.put((start_ind, end_ind, model_))
    
    q_ = mp.Queue()
    chunk_sz = int(n/n_proc)
    proc = []
    for p_num in range(n_proc):
        proc.append(mp.Process(target=target_proc,
                                args=(p_num,chunk_sz,q_),
                                daemon=True))
        proc[-1].start()

    for p_num in range(n_proc):
        start_ind, end_ind, model_ = q_.get()
        local_param.model[start_ind:end_ind] = model_

    q_.close()

    for p_num in range(n_proc):
        proc[p_num].join()

    print('local_param: all %d points processed...' % n)
    return local_param

# def lkpca(d, X, nbrhd_graph, opts, verbose=False, print_prop = 0.25):
#     n, p = X.shape
#     print_freq = int(print_prop * n)
    
#     local_param = Param('LKPCA')
#     local_param.X = X
#     local_param.model = np.empty(n, dtype=object)
#     local_param.zeta = np.zeros(n)

#     n_proc = opts['n_proc']
#     lkpca_kernel = opts['lkpca_kernel']
#     if lkpca_kernel=='correlation':
#         local_param.standardize = True
#         standardize = True
#         lkpca_kernel = 'linear'
#     else:
#         standardize = False
#     lkpca_fit_inverse_transform = opts['lkpca_fit_inverse_transform']
    
#     def target_proc(p_num, chunk_sz, q_):
#         start_ind = p_num*chunk_sz
#         if p_num == (n_proc-1):
#             end_ind = n
#         else:
#             end_ind = (p_num+1)*chunk_sz

#         for k in range(start_ind, end_ind):
#             local_param.model[k] = KernelPCA(n_components=d, kernel=lkpca_kernel,
#                                              fit_inverse_transform=lkpca_fit_inverse_transform,
#                                              random_state=42)
#             U_k = nbrhd_graph.get_nbr_inds(k)
#             X_k = X[U_k,:]
#             if standardize:
#                 X_k = X_k - np.mean(X_k,axis=0)[None,:]
#                 X_k = X_k/np.std(X_k, axis=0, ddof=1)[None,:]
#             local_param.model[k].fit(X_k)
    
#     q_ = mp.Queue()
#     chunk_sz = int(n/n_proc)
#     for p_num in range(n_proc):
#         target_proc(p_num,chunk_sz,q_)

#     print('local_param: all %d points processed...' % n)
#     return local_param