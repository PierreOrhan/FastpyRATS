import pdb
import numpy as np

# from sklearn.decomposition import SparsePCA
# from scipy.sparse.linalg import svds
# from scipy.linalg import pinv, svd

from ..common_ import *
from ..util_ import Param

# from joblib import Parallel, delayed
import torch
from ..nbrhd_graph_ import NbrhdGraph

def lpca(d: int, X: np.ndarray, nbrhd_graph: NbrhdGraph, 
         opts: dict, verbose: bool = False, print_prop: float = 0.25, seq_mode: bool = True):
    """
        Fast Local PCA using SVD on each neighborhood.

        Instead of relying on sparses matrices, this computes batched SVD on GPU using PyTorch.
        Distortion is then estimated using pyKeOps.

        Inputs:
            d: target dimension of the local subspaces  
            X: data matrix (n x p)
            nbrhd_graph: neighborhood graph (NbrhdGraph object), prefitted.
    """
    
    n, p = X.shape
    
    local_param = Param('LPCA')
    local_param.Psi = torch.zeros((n,p,d),device="cuda")
    local_param.mu = torch.zeros((n,p),device="cuda")
    local_param.zeta = torch.zeros(n,device="cuda")
    local_param.var_explained = torch.zeros((n,p),device="cuda")
    n_pc_dir_chosen = torch.zeros(X.shape[0],device="cuda")


    U = nbrhd_graph.neigh_ind # (n,k)    
    X = torch.tensor(X,dtype=torch.float32).to("cuda")
    X_k = X[U] # (n,k,p)
    xbar_k = torch.mean(X_k,axis=1,keepdim=True) #average across each neighborhood (n,1,p)
    
    X_k = X_k - xbar_k
    _,Sigma_k,Q_kt= torch.linalg.svd(X_k)
    Q_kt = Q_kt.transpose(1,2)
    
    ##  Either we have to explain some variance, or we just take d directions:
    if opts['explain_var'] > 0:
        var = torch.cumsum(Sigma_k/torch.sum(Sigma_k,dim=1,keepdim=True),dim=1)
        d1 = torch.minimum(torch.sum(var < opts['explain_var'],dim=1)+1,torch.tensor(d))

        ## In this case the number of pc directions chosen could be different for each point:
        for idn,subd in enumerate(d1):
            local_param.Psi[idn,:,:subd] = Q_kt[idn,:,:subd]
            local_param.var_explained[idn,:,:subd] = var[idn,:,:subd]
    else:
        d1 = d
        var_explained = (Sigma_k[:,:d]/torch.sum(Sigma_k[:,:d],dim=1,keepdim=True))
        local_param.var_explained[:,:d] = var_explained          
        
        local_param.Psi[:,:,:d] = Q_kt[:,:,:d]
    
    local_param.mu[:] = xbar_k.squeeze(1)
    local_param.n_pc_dir_chosen[:] = d1


    if verbose:
        print('local_param: all %d points processed...' % n)
    
    return local_param



    # local_param = Param('LPCA')
    # local_param.X = X
    # local_param.Psi = np.zeros((n,p,d))
    # local_param.mu = np.zeros((n,p))
    # local_param.var_explained = np.zeros((n,p))
    # local_param.n_pc_dir_chosen = np.zeros(n)
    
    # n_proc = opts['n_proc']
    # lpca_variant = opts['lpca_variant']

    # U = nbrhd_graph.sparse_matrix(nbr_inds_only=True)
    
    # def target_proc(p_num, chunk_sz):
    #     start_ind = p_num*chunk_sz
    #     if p_num == (n_proc-1):
    #         end_ind = n
    #     else:
    #         end_ind = (p_num+1)*chunk_sz

    #     n_inds = end_ind - start_ind
    #     Psi = np.zeros((n_inds,p,d))
    #     mu = np.zeros((n_inds,p))
    #     var_explained = np.zeros((n_inds,p))
    #     n_pc_dir_chosen = np.zeros(n_inds)

    #     for k in range(start_ind, end_ind):
    #         i = k - start_ind
    #         #U_k = nbrhd_graph.get_nbr_inds(k)
    #         #U_k = np.sort(nbrhd_graph.get_nbr_inds(k))
    #         U_k = U[k,:].indices
    #         X_k = X[U_k,:]
            
    #         xbar_k = np.mean(X_k,axis=0)[np.newaxis,:]
    #         X_k = X_k - xbar_k
    #         X_k = X_k.T
    #         if opts['explain_var'] > 0:
    #             Q_k,Sigma_k,_ = svd(X_k)
    #             var_explained[i,:] = Sigma_k**2/np.sum(Sigma_k**2)
    #             var = np.cumsum(var_explained[i,:])
    #             d1 = min(d, np.sum(var < opts['explain_var'])+1)
    #         else:
    #             d1 = d
    #             if d in X_k.shape:
    #                 Q_k,Sigma_k,_ = svd(X_k)
    #             else:
    #                 np.random.seed(42)
    #                 v0 = np.random.uniform(0,1,np.min(X_k.shape))
    #                 Q_k,Sigma_k,_ = svds(X_k, k=d, which='LM', v0=v0)
    #                 #Q_k,Sigma_k,_ = svds(X_k, k=d, which='LM')

    #                 # if p_num == 0:
    #                 #     print('svd end')
    #             #var_explained[i,:d] = Sigma_k/np.sum(Sigma_k)
    #             var_explained[i,:d] = Sigma_k**2
    #             var_explained[i,:d] /= np.sum(var_explained[i,:d])
    #             n_pc_dir_chosen[i] = d1

    #         Psi[i,:,:d1] = Q_k[:,:d1]
    #         mu[i,:] = xbar_k
        
    #     return start_ind, end_ind, Psi, mu, var_explained, n_pc_dir_chosen
    
    # if seq_mode:
    #     n_proc = 1
    # chunk_sz = int(n/n_proc)
    # results = Parallel(n_jobs=n_proc)(delayed(target_proc)(i, chunk_sz) for i in range(n_proc))

    # for i in range(len(results)):
    #     start_ind, end_ind, Psi_, mu_, var_explained_, n_pc_dir_chosen_ = results[i]
    #     local_param.Psi[start_ind:end_ind,:] = Psi_
    #     local_param.mu[start_ind:end_ind,:] = mu_
    #     local_param.var_explained[start_ind:end_ind,:] = var_explained_
    #     local_param.n_pc_dir_chosen[start_ind:end_ind] = n_pc_dir_chosen_

    # if verbose:
    #     print('local_param: all %d points processed...' % n)
    
    # return local_param