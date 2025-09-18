import pdb
import time
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from .util_ import print_log

def sinkhorn(
        K,
        maxiter=10000,
        delta=1e-12,
        boundC = 1e-8,
        print_freq=1000
    ):
    """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
    n = K.shape[0]
    r = np.ones((n,1))
    u = np.ones((n,1))
    v = r/(K.dot(u))
    x = np.sqrt(u*v)
    assert np.min(x) > boundC, 'assert min(x) > boundC failed.'
    for tau in range(maxiter):
        error =  np.max(np.abs(u*(K.dot(v)) - r))
        if tau%print_freq:
            print('Error:', error, flush=True)
        
        if error < delta:
            print('Sinkhorn converged at iter:', tau)
            break

        u = r/(K.dot(v))
        v = r/(K.dot(u))
        x = np.sqrt(u*v)
        if np.sum(x<boundC) > 0:
            print('boundC not satisfied at iter:', tau)
            x[x < boundC] = boundC
        
        u=x
        v=x
    x = x.flatten()
    K.data = K.data*x[K.row]*x[K.col]
    return K

def graph_laplacian(
        nbrhd_graph,
        k_tune=7,
        gl_type='unnorm',
        return_diag=False,
        use_out_degree=True,
        tuning='self',
        kernel='gaussian',
        ds_max_iter=0,
        pa_sigma=np.pi/16):
    n = nbrhd_graph.get_num_nodes()
    row_inds = nbrhd_graph.get_row_inds()
    col_inds = nbrhd_graph.get_col_inds()
    if tuning is not None:
        # Compute local scale
        sigma = nbrhd_graph.get_distance_to_kth_nbr(k_tune+1)
        if tuning=='self': # scaling depends on sigma_i and sigma_j
            autotune = sigma[row_inds]*sigma[col_inds]
        elif tuning=='solo': # scaling depends on sigma_i only
            autotune = sigma[row_inds]**2
        elif tuning=='median': # scaling is fixed across data points
            autotune = np.median(sigma)**2

    eps = np.finfo(np.float64).eps
    if kernel=='binary':
        K = np.ones(row_inds)
        autotune = None
    elif kernel=='gaussian':
        dist2 = nbrhd_graph.get_data()**2/autotune
        if nbrhd_graph.principal_angles is not None:
            dist2 = dist2 + ((nbrhd_graph.principal_angles.flatten()**2)/(pa_sigma**2))/(dist2+1e-12)
        # if hasattr(nbrhd_graph, 'principal_angles'):
        #     # mask = nbrhd_graph.principal_angles.flatten() < pa_sigma
        #     # max_val = 4
        #     # dist2 = nbrhd_graph.principal_angles.flatten()**2/pa_sigma**2 * mask + (1-mask)*max_val
        #     dist2 = nbrhd_graph.principal_angles.flatten()**2/pa_sigma**2
        # else:
        #     dist2 = nbrhd_graph.get_data()**2/autotune
        K = np.exp(-dist2) + eps
    elif kernel=='laplacian':
        K = np.exp(-nbrhd_graph.get_data()/np.sqrt(autotune)) + eps

    K = csr_matrix(
        (K, (row_inds, col_inds)),
        shape=(n,n)
    )
    ones_like_K = csr_matrix(
        (np.ones(row_inds.shape[0]), (row_inds, col_inds)),
        shape=(n,n)
    )
    # average symmetrization
    K = K + K.T
    ones_like_K = ones_like_K + ones_like_K.T
    K.data /= ones_like_K.data

    if ds_max_iter:
        K = sinkhorn(K.tocoo(), maxiter=ds_max_iter)
        
    if gl_type=='diffusion':
        Dinv = 1/(K.sum(axis=1).reshape((n,1)))
        K = K.multiply(Dinv).multiply(Dinv.transpose())
        gl_type = 'symnorm'

    if gl_type=='symnorm':
        return autotune,\
               laplacian(
                   K,
                   normed=True,
                   return_diag=return_diag,
                   use_out_degree=use_out_degree
                )
    elif gl_type == 'unnorm':
        return autotune,\
               laplacian(
                   K,
                   normed=False,
                   return_diag=return_diag,
                   use_out_degree=use_out_degree
                )

class GL:
    def __init__(self, gl_opts, debug=False, verbose=False, seed=None):
        self.gl_opts = gl_opts
        self.debug = debug
        self.verbose = verbose
        self.seed = seed

        self.local_start_time = time.perf_counter()
        self.global_start_time = time.perf_counter()

    def log(self, s='', log_time=False):
        if self.verbose:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    def fit(self, nbrhd_graph):
        gl_type = self.gl_opts['which']
        tuning = self.gl_opts['tuning']
        ds_max_iter = self.gl_opts['doubly_stochastic_max_iter']
        k_tune = self.gl_opts['k_tune']
        n_eig = self.gl_opts['n_eig']
        n_trivial = self.gl_opts['n_trivial']

        np.random.seed(self.seed)
        n = nbrhd_graph.get_num_nodes()

        #v0 = np.random.uniform(0,1,n)
        v0 = np.ones(n)/np.sqrt(n)
        if gl_type in ['unnorm', 'symnorm']:
            autotune, L = graph_laplacian(nbrhd_graph,
                                          k_tune, gl_type, tuning=tuning,
                                          ds_max_iter=ds_max_iter)
            #TODO: which one to use?
            #lmbda, phi = eigsh(L, k=n_eig+1, v0=v0, which='SM')
            lmbda, phi = eigsh(L, k=n_eig+n_trivial, v0=v0, sigma=-1e-3)
        else:
            if gl_type == 'random_walk':
                gl_type = 'symnorm'
            autotune, L_and_sqrt_D = graph_laplacian(nbrhd_graph, k_tune, gl_type, return_diag=True,
                                                    tuning=tuning, ds_max_iter=ds_max_iter)
            L, sqrt_D = L_and_sqrt_D
            #TODO: which one to use?
            #lmbda, phi = eigsh(L, k=n_eig+n_trivial, v0=v0, which='SM')
            lmbda, phi = eigsh(L, k=n_eig+n_trivial, v0=v0, sigma=-1e-3)
            
            L = L.multiply(1/sqrt_D[:,np.newaxis]).multiply(sqrt_D[np.newaxis,:])
            phi = phi/sqrt_D[:,np.newaxis]
            
            #TODO: Is this normalization needed?
            phi = phi/(np.linalg.norm(phi,axis=0)[np.newaxis,:])

        if self.debug:
            # The trivial eigenvalues and eigenvectors
            self.lmbda0 = lmbda[:n_trivial]
            self.phi0 = phi[:,:n_trivial]
            self.v0 = v0
            self.autotune = autotune
        
        # Remaining eigenvalues and eigenvectors
        self.L = L
        self.lmbda = lmbda[n_trivial:]
        self.phi = phi[:,n_trivial:]