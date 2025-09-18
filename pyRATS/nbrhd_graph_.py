import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import subspace_angles

from .util_ import nearest_neighbors, sparse_matrix, radial_nearest_neighbors

class NbrhdGraph:
    def __init__(
            self,
            k_nn=5,
            metric='euclidean',
            radius=None
        ):
        self.k_nn = k_nn
        self.metric = metric
        self.principal_angles = None
        self.radius = radius
        if self.radius is not None:
            self.k_nn = None
    
    def fit(self, X):
        if self.radius is not None:
            neigh_dist, neigh_ind = radial_nearest_neighbors(
                X,
                radius=self.radius,
                metric=self.metric
            )
        else:
            neigh_dist, neigh_ind = nearest_neighbors(
                X,
                k_nn=self.k_nn,
                metric=self.metric
            )
        self.neigh_dist = neigh_dist
        self.neigh_ind = neigh_ind

    
    def compute_principal_angles(self, Psi):
        self.principal_angles = np.zeros(self.neigh_ind.shape)
        for i in range(self.principal_angles.shape[0]):
            Psi_i = Psi[i,:]
            for j_ in range(self.neigh_ind[i,:].shape[0]):
                j = self.neigh_ind[i,j_]
                Psi_j = Psi[j,:]
                theta = subspace_angles(Psi_i, Psi_j)
                self.principal_angles[i,j_] = np.sum(theta)

    def get_num_nodes(self):
        return self.neigh_ind.shape[0]
    
    def get_nbrs(self, i):
        return self.neigh_ind[i], self.neigh_dist[i]

    def get_nbr_inds(self, i):
        return self.neigh_ind[i]
    
    def truncate(self, k_nn_new):
        if k_nn_new >= self.k_nn:
            return
        self.neigh_dist = self.neigh_dist[:,:k_nn_new]
        self.neigh_ind = self.neigh_ind[:,:k_nn_new]
        self.k_nn = k_nn_new

    def sparse_matrix(self, nbr_inds_only = False, symmetrize=False):
        if nbr_inds_only:
            n,k = self.neigh_ind.shape
            output = sparse_matrix(self.neigh_ind, np.ones((n, k), dtype=bool))
        else:
            output = sparse_matrix(self.neigh_ind, self.neigh_dist)
        
        if symmetrize:
            output = output.maximum(output.transpose())
        
        return output
    
    def get_distance_to_kth_nbr(self, k):
        return self.neigh_dist[:,k-1]
    
    def get_row_inds(self):
        n, k = self.neigh_ind.shape
        return np.repeat(np.arange(n), k)
    
    def get_col_inds(self):
        return self.neigh_ind.flatten()
    
    def get_data(self):
        return self.neigh_dist.flatten()
    
    def induce_connections(self, X, cond_num):
        d_e = squareform(pdist(X, metric=self.metric))
        neigh_ind = np.zeros_like(self.neigh_ind)
        neigh_dist = np.zeros_like(self.neigh_dist)
        uniq_cond_nums = np.unique(cond_num)
        for i in range(uniq_cond_nums.shape[0]):
            print('Processing condition number:', i, flush=True)
            cond_num_i = uniq_cond_nums[i]
            mask = cond_num == cond_num_i
            inds = np.where(mask)[0]
            d_e_ = d_e.copy()
            d_e_[np.ix_(mask,mask)] = np.inf
            for k in inds.tolist():
                d_e_[k,self.neigh_ind[k,:]] = np.inf
                neigh_ind[k,:] = np.argpartition(d_e_[k,:], self.k_nn)[:self.k_nn]
                neigh_dist[k,:] = d_e_[k,neigh_ind[k,:]]
        
        self.neigh_ind = np.concatenate([self.neigh_ind, neigh_ind], axis=1)
        self.neigh_dist = np.concatenate([self.neigh_dist, neigh_dist], axis=1)
        self.k_nn *= 2

    def get_state(self):
        state = {
            'k_nn': self.k_nn,
            'metric': self.metric,
            'neigh_ind': self.neigh_ind,
            'neigh_dist': self.neigh_dist,
            'principal_angles': self.principal_angles
        }
        return state
    
    def set_state(self, state):
        self.k_nn = state['k_nn']
        self.metric = state['metric']
        self.neigh_ind = state['neigh_ind']
        self.neigh_dist = state['neigh_dist']
        self.principal_angles = state['principal_angles']