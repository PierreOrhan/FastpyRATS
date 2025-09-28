import time
import numpy as np

from .common_ import *
from .util_ import print_log, sub_dict, Param
from .local_view_algos import lpca, lkpca, postprocess
from .nbrhd_graph_ import NbrhdGraph

class LocalViews:
    def __init__(
            self,
            exit_at=None,
            verbose=True,
            debug=False,
            seed=None
        ):
        self.exit_at = exit_at
        self.verbose = verbose
        self.debug = debug
        self.seed = seed
        self.local_param_pre = Param()
        self.local_param_post = Param()
        
        self.local_start_time = time.perf_counter()
        self.global_start_time = time.perf_counter()
        
    def log(self, s='', log_time=False):
        if self.verbose:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    # TODO: relax X to be a distance matrix
    def fit(self, d: int, X: np.ndarray, nbrhd_graph: NbrhdGraph, local_opts):
        n = nbrhd_graph.get_num_nodes()
        algo = local_opts['algo']
        self.log('Computing local views using ' + local_opts['algo'])

        # Set the processsing pipeline based on local_opts
        nbrhd_graph.truncate(local_opts['k_for_pca'])

        assert algo in [LPCA, LKPCA]
        # Compute local views
        if algo == LPCA:
            hyperparams = ['lpca_variant', 'explain_var', 'n_proc']
            opts = sub_dict(local_opts, hyperparams)
            local_param_pre = lpca(d, X, nbrhd_graph, opts, verbose=self.verbose)
        elif algo == LKPCA:
            hyperparams = ['lkpca_kernel', 'lkpca_fit_inverse_transform', 'n_proc']
            opts = sub_dict(local_opts, hyperparams)
            local_param_pre = lkpca(d, X, nbrhd_graph, opts, verbose=self.verbose)
        
        self.local_param_pre = local_param_pre
        
        # TODO Pierre: change here
         # Patches on the data
        nbrhd_graph.truncate(local_opts['k'])
        self.U = nbrhd_graph.sparse_matrix(nbr_inds_only=True)
        self.local_param_post = self.local_param_pre
        self.local_param_post.b = np.ones(n)

    def postprocess(self, nbrhd_graph, local_opts):
        if local_opts['to_postprocess']:
            self.local_param_post = postprocess(
                self.U,
                nbrhd_graph.sparse_matrix(symmetrize=True), 
                self.local_param_pre,
                local_opts
            )
    
    def get_state(self):
        state = {
            'exit_at': self.exit_at,
            'debug': self.debug,
            'verbose': self.verbose,
            'seed': self.seed,
            'U': self.U,
            'local_param_pre': self.local_param_pre.get_state(),
            'local_param_post': self.local_param_post.get_state(),
        }
        return state

    def set_state(self, state):
        self.exit_at = state['exit_at']
        self.debug = state['debug']
        self.verbose = state['verbose']
        self.seed = state['seed']
        self.U = state['U']
        self.local_param_pre.set_state(state['local_param_pre'])
        self.local_param_post.set_state(state['local_param_post'])