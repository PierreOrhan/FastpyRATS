from types import SimpleNamespace
from .util_ import Param
from .gl_ import graph_laplacian
from .nbrhd_graph_ import NbrhdGraph
from .buml_ import BUML
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, spsolve
from scipy.linalg import svd
import numpy as np

# Hein, Matthias, and Markus Maier. "Manifold denoising." 
# Advances in neural information processing systems 19 (2006).
class ManifoldDenoiser:
    def __init__(self, opts):
        # default opts
        default_opts = {
            'k_nn': 28,
            'k_tune': 7,
            'max_iter': 3,
            'reg': 0.5,
            'debug': False,
            'metric': 'euclidean',
            'tuning': 'self',
            'which': 'unnorm',
            'doubly_stochastic_max_iter': 1000,
            'rank': None, # if non-zero then low-rank approx of laplacian is used
        }
        # override default opts from opts
        for k in opts:
            default_opts[k] = opts[k]

        # convert to dot notation
        self.opts = SimpleNamespace(**default_opts)
    
    def fit(self, X):
        denoised_X_list = [X.copy()]
        for i in range(self.opts.max_iter):
            print('Iter:',i)
            X = denoised_X_list[-1]
            # construct graph Laplacian
            nbrhd_graph = NbrhdGraph(k_nn = self.opts.k_nn, metric=self.opts.metric)
            nbrhd_graph.fit(X=X)
            autotune, L_and_sqrt_D = graph_laplacian(nbrhd_graph, self.opts.k_tune, self.opts.which,
                                                     return_diag = True, tuning=self.opts.tuning,
                                                     ds_max_iter=self.opts.doubly_stochastic_max_iter)
            L, sqrt_D = L_and_sqrt_D
            if self.opts.rank is None:
                L = L.multiply(1/sqrt_D[:,np.newaxis]).multiply(sqrt_D[np.newaxis,:])
                L = L*self.opts.reg
                L.setdiag(L.diagonal() + 1)
                #denoised_X_list.append(sinv(L.tocsc()).dot(X))
                denoised_X_list.append(spsolve(L.tocsc(), X))
            else:
                np.random.seed(42)
                v0 = np.random.uniform(0, 1, L.shape[0])
                print('Computing low rank approx of L of shape', L.shape, flush=True)
                lmbda, phi = eigsh(L, k=self.opts.rank, v0=v0, sigma=-1e-3)
                temp = (phi*(1/(1+self.opts.reg*lmbda[None,:]))).dot(phi.T)
                temp = temp*(1/sqrt_D[:,None])*(sqrt_D[None,:])
                denoised_X_list.append(temp.dot(X))

        self.denoised_X_list = denoised_X_list

# Gong, Dian, Fei Sha, and Gérard Medioni. "Locally linear denoising on image manifolds." 
# Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics.
# JMLR Workshop and Conference Proceedings, 2010.
class LocalLinearDenoiser:
    def __init__(self, opts):
        # default opts
        default_opts = {
            'd': 2,
            'explain_var': 0,
            'k_nn': 28,
            'k': 14,
            'k_tune': 7,
            'reg': 0.1,
            'metric': 'euclidean',
            'tuning': 'self',
            'which': 'unnorm',
            'doubly_stochastic_max_iter': 1000,
            'verbose': False,
            'debug': True,
        }
        # override default opts from opts
        for k in opts:
            default_opts[k] = opts[k]

        # convert to dot notation
        self.opts = SimpleNamespace(**default_opts)

    def fit(self, X):
        # Estimate gradients of Laplacian eigenvectors at each point
        # and choose
        d = self.opts.d
        buml_obj = BUML(d = d, local_opts={'algo':'LPCA', 'k': self.opts.k,
                                          'explain_var': self.opts.explain_var,
                                          'to_postprocess': False},
                        intermed_opts={'eta_min': 1, 'eta_max': 1},
                        verbose=self.opts.verbose, debug=self.opts.debug, exit_at='local_views')
        buml_obj.fit(X=X)

        U = buml_obj.LocalViews.U
        local_param_post = buml_obj.LocalViews.local_param_post
        self.n_pc_dir_chosen = local_param_post.n_pc_dir_chosen
        self.var_explained = local_param_post.var_explained
        # Construct setup transformation
        local_param = Param('LPCA')
        n, p = X.shape
        local_param.X = X
        local_param.b = np.ones(n)
        local_param.Psi = np.zeros((n, p, p))
        local_param.mu = np.zeros((n,p))
        local_param.T = np.zeros((n, p, p))
        for k in range(X.shape[0]):
            U_k = U[k,:].indices
            local_param.Psi[k,:,:d] = local_param_post.Psi[k,:,:d]
            local_param.mu[k,:] = np.mean(X[U_k,:], axis=0)
            local_param.T[k,:d,:] = local_param.Psi[k,:,:d].T
        
        local_param.v = local_param.mu.copy()
        self.denoised_X, self.denoised_nbrhd_graph = consensus_based_reconstruction(X, U, local_param, self.opts)
        self.buml_obj = buml_obj

class LocalKernelDenoiser:
    def __init__(self, opts):
        # default opts
        default_opts = {
            'd': 2,
            'explain_var': 0,
            'k_nn': 28,
            'k_tune': 7,
            'reg': 0.1,
            'metric': 'euclidean',
            'kernel': 'linear',
            'tuning': 'self',
            'which': 'unnorm',
            'doubly_stochastic_max_iter': 1000,
            'verbose': False,
            'debug': True,
        }
        # override default opts from opts
        for k in opts:
            default_opts[k] = opts[k]

        # convert to dot notation
        self.opts = SimpleNamespace(**default_opts)

    def fit(self, X):
        # Estimate gradients of Laplacian eigenvectors at each point
        # and choose
        d = self.opts.d
        buml_obj = BUML(d = d, local_opts={'algo':'LKPCA', 'k': self.opts.k_nn,
                                          'explain_var': self.opts.explain_var,
                                          'lkpca_kernel': self.opts.kernel,
                                           'metric': self.opts.metric,
                                           'metric0': self.opts.metric,
                                           'lkpca_fit_inverse_transform': True,
                                          'to_postprocess': False},
                        intermed_opts={'eta_min': 1, 'eta_max': 1},
                        verbose=self.opts.verbose, debug=self.opts.debug, exit_at='local_views')
        buml_obj.fit(X=X)
        n, p = X.shape
        U = buml_obj.LocalViews.U
        _, L = graph_laplacian(buml_obj.nbrhd_graph, self.opts.k_tune,
                                self.opts.which, return_diag = False, tuning=self.opts.tuning,
                                ds_max_iter=self.opts.doubly_stochastic_max_iter)
        
        L = n*self.opts.reg*L
        L.setdiag(L.diagonal() + np.array(U.sum(axis=0)).flatten())
        SQT = np.zeros(X.shape)
        local_param = buml_obj.LocalViews.local_param_post
        for k in range(X.shape[0]):
            U_k = U[k,:].indices
            temp = local_param.eval_({'view_index': k, 'data_mask': U_k})
            SQT[U_k,:] += local_param.reconstruct_({'view_index': k, 'embeddings': temp})
        
        self.denoised_X = spsolve(L.tocsc(), SQT)

def build_connection_laplacian(L, Psi):
    n = L.shape[0]
    d = Psi.shape[-1]
    row = L.row
    col = L.col
    data = L.data
    CL_row = []
    CL_col = []
    CL_data = []
    for k in range(len(row)):
        i = row[k]
        j = col[k]
        CL_row += np.repeat(np.arange(i*d,(i+1)*d), d).tolist()
        CL_col += np.tile(np.arange(j*d,(j+1)*d), d).tolist()
        if i!=j:
            O_i = Psi[i,:,:]
            O_j = Psi[j,:,:]
            U, _, VT = svd(O_j.T.dot(O_i))
            sigma_ij = U.dot(VT).T
            CL_data += (data[k]*sigma_ij).flatten().tolist()
        else:
            CL_data += (data[k]*np.eye(d)).flatten().tolist()
            
    return coo_matrix((CL_data, (CL_row, CL_col)), shape=(n*d, n*d))

def get_vector_fields(CL, Psi, N=20):
    np.random.seed(42)
    n, p, d = Psi.shape
    v0 = np.random.uniform(0, 1, CL.shape[0])
    W, V = eigsh(CL, k=N, v0=v0, which='LM', sigma=-1e-3)
    VFs = np.zeros((N, n, p))
    for i in range(N):
        Vi = V[:,i].reshape(n, d)
        VFs[i,:,:] = np.sum(Psi*Vi[:,None,:], axis=-1)
    return VFs, W

def consensus_based_reconstruction(X, U, local_param, opts):
    n = X.shape[0]
    nbrhd_graph = NbrhdGraph(k_nn=opts.k_nn, metric=opts.metric)
    nbrhd_graph.fit(X=X)
    # replace nbrhd distances with local param distances
    for k in range(n):
        U_k = nbrhd_graph.neigh_ind[k,:]
        X_k = local_param.eval_({'view_index': k, 'data_mask': U_k})
        nbrhd_graph.neigh_dist[k,:] = np.linalg.norm(X_k - X_k[0:1,:],axis=1)

    _, L = graph_laplacian(nbrhd_graph, opts.k_tune, opts.which,
                            return_diag = False, tuning=opts.tuning,
                            ds_max_iter=opts.doubly_stochastic_max_iter)
    
    L = n*opts.reg*L
    L.setdiag(L.diagonal() + np.array(U.sum(axis=0)).flatten())
    SQT = np.zeros(X.shape)
    for k in range(n):
        U_k = U[k,:].indices
        SQT[U_k,:] += local_param.eval_({'view_index': k, 'data_mask': U_k})
    #self.denoised_X = sinv(L.tocsc()).dot(SQT)
    return spsolve(L.tocsc(), SQT), nbrhd_graph