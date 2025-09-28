import pdb
import numpy as np
import time

from .common_ import *
from . import local_views_
from . import intermed_views_
from . import global_views_
from . import visualize_
from .nbrhd_graph_ import NbrhdGraph
from .util_ import print_log
import multiprocess as mp
import json
import copy

def get_default_local_opts(
        algo='LPCA', metric='euclidean', k=28,
        k_for_pca=None,
        to_postprocess=True,
        local_metric='euclidean',
        explain_var=0, lkpca_kernel='linear',
        lpca_variant='standard',
        lkpca_fit_inverse_transform=False
    ):
    """Sets and returns a dictionary of default_local_opts. 
    
    Parameters
    ----------
    """
    return {
        'algo': algo,
        'metric': metric,
        'k': k,
        'k_for_pca': k_for_pca,
        'explain_var': explain_var,
        'lkpca_kernel': lkpca_kernel,
        'lpca_variant': lpca_variant,
        'to_postprocess': to_postprocess,
        'local_metric': local_metric,
        'lkpca_fit_inverse_transform': lkpca_fit_inverse_transform
    }

def get_default_intermed_opts(algo='best', cost_fn='distortion', n_times=4, eta_min=5, eta_max=25, len_S_thresh=256, c=None, tracker=False):
    """Sets and returns a dictionary of default_intermed_opts, (experimental) options are work in progress.
    
    Parameters
    ----------
    algo : str
        Algo to use. Options are 'mnm' for match and merge,
        'best' for the optimal algorithm (much slower
        but creates intermediate views with lower distortion).
        'mnm' is (experimental).
    cost_fn: str
        Defines the cost function to move/merge a point/cluster into
        another cluster. Options are distortion or alignment_error. 
    n_times : int
        (experimental) Hypereparameter for 'mnm' algo. Number of times to match
        and merge. If n is the #local views then the #intermediate
        views will be approximately n/2^{n_times}.
    eta_min : int
        Hyperparameter for 'best' algo. Minimum allowed size of 
        the clusters underlying the intermediate views.
        The values must be >= 1.
    eta_max : int
        Hyperparameter for 'best' algo. Maximum allowed size of
        the clusters underlying the intermediate views.
        The value must be > eta_min.
    len_S_thresh : int
        Threshold on the number of points for which 
        the costs are to be updated, to invoke
        multiple processors. Used with 'best' algo only.
    c: np.array
        Cluster (partition) labels of each point computed using external procedure.
    """
    return {'algo': algo, 'cost_fn': cost_fn, 'n_times': n_times, 'eta_min': eta_min,
            'eta_max': eta_max, 'len_S_thresh': len_S_thresh, 'c': c, 'tracker': tracker}
    
def get_default_global_opts(align_transform='rigid', to_tear=True, nu=3, max_iter=20, color_tear=True,
                            vis_before_init=False, compute_error=False,
                            init_algo_name='procrustes', align_w_parent_only=True, tree='mst',
                            root_view='center', refine_algo_name='rgd',
                            max_internal_iter=100, alpha=0.3, eps=1e-8,
                            add_dim=False, beta={'align':None, 'repel': 1},
                            repel_by=0., repel_decay=1., n_repel=0,
                            max_var_by=None, max_var_decay=0.9,
                            far_off_points_type='reuse_fixed', patience=5, tol=1e-2,
                            tear_color_method='spectral', tear_color_eig_inds=[1],
                            metric='euclidean', color_cutoff_frac=0.001, color_cutoff_frac2=0.,
                            color_largest_tear_comp_only=False,
                            n_forced_clusters=1):
    """Sets and returns a dictionary of default_global_opts, (experimental) options are work in progress.

    Parameters
    ----------
    align_transform : str
        The algorithm to use for the alignment of intermediate
        views. Options are 'rigid' and 'affine'. (experimental) If 'affine' is
        chosen then none of the following hypermateters are used.
    to_tear : bool
        If True then tear-enabled alignment of views is performed.
    nu : int
        The ratio of the size of local views in the embedding against those
        in the data.
    max_iter : int
        Number of iterations to refine the global embedding for.
    color_tear : bool
        If True, colors the points across the tear with a
        spectral coloring scheme.
    vis_before_init : bool
        If True, plots the global embedding before
        alignment begins. This is same as just plotting
        all the intermediate views without alignment.
    compute_error : bool
        If True the alignment error is computed at each 
        iteration of the refinement, otherwise only at
        the last iteration.
    init_algo_name : str
        The algorithm used to compute initial global embedding
        by aligning the intermediate views. Options are 'procrustes'
        for spanning-tree-based-procrustes alignment,
        'spectral' for spectral alignment (ignores to_tear),
        'sdp' for semi-definite programming based alignment (ignores to_tear).
    align_w_parent_only : bool
        If True, then aligns child views the parent views only
        in the spanning-tree-based-procrustes alignment.
    tree: str
        Type of spanning tree to use. Options are: spt, mst (default).
    root_view: str
        If 'center' then uses center of spanning tree as root view
        otherwise uses the view associated with largest cluster.
        Default if 'largest'.
    refine_algo_name : str
        The algorithm used to refine the initial global embedding
        by refining the alignment between intermediate views.
        Options are 'gpa' for Generalized Procustes Analysis
        (GPA) based alignment, 'rgd' for Riemannian gradient descent
        based alignment, 'spectral' for spectral alignment,
        'gpm' for generalized power method based alignment,
        'sdp' for semi-definite programming based alignment. Note that
        sdp based alignment is very slow. Recommended options are 'rgd'
        with an appropriate step size (alpha) and 'gpm'.
    max_internal_iter : int
        The number of internal iterations used by
        the refinement algorithm, for example, RGD updates.
        This is ignored by 'spectral' refinement.
    alpha : float
        The step size used in the Riemannian gradient descent
        when the refinement algorithm is 'rgd'.
    eps : float
        The tolerance used by sdp solver when the init or refinement
        algorithm is 'sdp'.
    add_dim : bool
        (experimental) add an extra dimension to intermediate views.
    beta : dict
        (experimental) Hyperparameters used for computing the alignment weights and
        the repulsion weights. Form is {'align': float, 'repel': float}.
        Default is {'align': None, 'repel': None} i.e. unweighted.
    repel_by : float
        If positive, the points which are far off are repelled
        away from each other by a force proportional to this parameter.
        Ignored when refinement algorithm is 'gpa'.
    repel_decay : float
        Multiply repel_decay with current value of repel_by after every iteration.
    n_repel : int
        The number of far off points repelled from each other.
    far_off_points_type : 'fixed' or 'random'
        Whether to use the same points for repulsion or 
        randomize over refinement iterations. If 'reuse' is
        in the string, for example 'fixed_reuse', then the points
        to be repelled are the same across iterations.
    patience : int
        The number of iteration to wait for error below tolerance
        to persist before stopping the refinement.
    tol : float
        The tolerance level for the relative change in the alignment error and the
        relative change in the size of the tear.
    tear_color_method : str
        Method to color the tear. Options are 'spectral' or 'heuristic'.
        The latter keeps the coloring of the tear same accross
        the iterations. Recommended option is 'spectral'.
    tear_color_eig_inds : int
        Eigenvectors to be used to color the tear. The value must either
        be a non-negative integer or it must be a list of three non-negative
        integers [R,G,B] representing the indices of eigenvectors to be used
        as RGB channels for coloring the tear. Higher values result in
        more diversity of colors. The diversity saturates after a certain value.
    color_cutoff_frac : float
        If the number of points in a tear component is less than
        (color_cutoff_frac * number of data points), then all the
        points in the component will be colored with the same color.
    color_largest_tear_comp_only : bool
        If True then the largest tear components is colored only.
    metric : str
        metric assumed on the global embedding. Currently only euclidean is supported.
    n_forced_clusters : str
        (experimental) Minimum no. of clusters to force in the embeddings.
    """
    return {'to_tear': to_tear, 'nu': nu, 'max_iter': max_iter,
               'color_tear': color_tear,
               'vis_before_init': vis_before_init,
               'compute_error': compute_error,
               'align_transform': align_transform, 
               'init_algo_name': init_algo_name,
               'root_view': root_view,
               'align_w_parent_only': align_w_parent_only,
               'tree': tree,
               'refine_algo_name': refine_algo_name, 
               'max_internal_iter': max_internal_iter,
               'alpha': alpha, 'eps': eps, 'add_dim': add_dim,
               'beta': beta, 'repel_by': repel_by,
               'repel_decay': repel_decay, 'n_repel': n_repel,
               'max_var_by': max_var_by, 'max_var_decay': max_var_decay,
               'far_off_points_type': far_off_points_type,
               'patience': patience, 'tol': tol,
               'tear_color_method': tear_color_method,
               'tear_color_eig_inds': tear_color_eig_inds,
               'metric': metric, 'color_cutoff_frac': color_cutoff_frac,
               'color_cutoff_frac2': color_cutoff_frac2,
               'color_largest_tear_comp_only': color_largest_tear_comp_only,
               'n_forced_clusters': n_forced_clusters
              }
def get_default_vis_opts(save_dir='', cmap_interior='summer', cmap_boundary='jet', c=None):
    """Sets and returns a dictionary of default_vis_opts.
    
    Parameters
    ----------
    save_dir : str
               The directory to save the plots in.
    cmap_interior : str
                    The colormap to use for the interior of the manifold.
    cmap_boundary : str
                    The colormap to use for the boundary of the manifold.
    c : array shape (n_samples)
        The labels for each point to be used to color them.
    """
    return {'save_dir': save_dir,
             'cmap_interior': cmap_interior,
             'cmap_boundary': cmap_boundary,
             'c': c}

class BUML:
    """Bottom-up manifold learning.
    
    Parameters
    ----------
    d : int
       Intrinsic dimension of the manifold.
    local_opts : dict
                Options for local views construction. The key-value pairs
                provided override the ones in default_local_opts.
    intermed_opts : dict
                    Options for intermediate views construction. The key-value pairs
                    provided override the ones in default_intermed_opts.
    global_opts : dict
                  Options for global views construction. The key-value pairs
                  provided override the ones in default_global_opts.
    vis_opts : dict
               Options for visualization. The key-value pairs
               provided override the ones in default_vis_opts.
    n_proc : int
             The number of processors to use. Defaults to approximately
             3/4th of the available processors. 
    verbose : bool
              print logs if True.
    debug : bool
            saves intermediary objects/data for debugging.
    """
    def __init__(self,
                 d = 2,
                 local_opts = {}, 
                 intermed_opts = {},
                 global_opts = {},
                 vis_opts = {},
                 n_proc = min(32,max(1,int(mp.cpu_count()*0.75))),
                 exit_at = None,
                 verbose = False,
                 debug = False):
        default_local_opts = get_default_local_opts()
        default_intermed_opts = get_default_intermed_opts()
        default_global_opts = get_default_global_opts()
        default_vis_opts = get_default_vis_opts()
        self.verbose = verbose
        self.debug = debug
        self.n_proc = n_proc
        
        self.d = d
        local_opts['n_proc'] = n_proc
        for i in local_opts:
            if i == 'gl_opts':
                for j in local_opts[i]:
                    default_local_opts[i][j] = local_opts[i][j]
            else:
                default_local_opts[i] = local_opts[i]
        self.local_opts = default_local_opts
        #############################################
        intermed_opts['n_proc'] = n_proc
        intermed_opts['local_algo'] = self.local_opts['algo']
        intermed_opts['verbose'] = verbose
        intermed_opts['debug'] = debug
        for i in intermed_opts:
            default_intermed_opts[i] = intermed_opts[i]
        self.intermed_opts = default_intermed_opts
        # Update k_nn
        if self.local_opts['k_for_pca'] is None:
            self.local_opts['k_for_pca'] = self.local_opts['k']
        else:
            self.local_opts['k_for_pca'] = max(self.local_opts['k_for_pca'], self.local_opts['k'])

        if self.intermed_opts['cost_fn'] == 'distortion':
            self.local_opts['k_nn0'] = max(self.local_opts['k_for_pca'],
                                        self.intermed_opts['eta_max']*self.local_opts['k'])
        else:
            self.local_opts['k_nn0'] = self.local_opts['k_for_pca']
        print("local_opts['k_nn0'] =", self.local_opts['k_nn0'], "is created.")

        #############################################
        global_opts['k'] = self.local_opts['k']
        global_opts['n_proc'] = n_proc
        global_opts['verbose'] = verbose
        global_opts['debug'] = debug
        for i in global_opts:
            default_global_opts[i] = global_opts[i]
        if default_global_opts['refine_algo_name'] != 'rgd':
            if 'max_internal_iter' not in global_opts:
                default_global_opts['max_internal_iter'] = 10
                print("Making global_opts['max_internal_iter'] =",
                      default_global_opts['max_internal_iter'])
                print('Supply the argument to use a different value', flush=True)
        self.global_opts = default_global_opts
        #############################################
        for i in vis_opts:
            default_vis_opts[i] = vis_opts[i]
        self.vis_opts = default_vis_opts
        self.vis = visualize_.Visualize(self.vis_opts['save_dir'])
            
        #############################################
        self.exit_at = exit_at
        self.verbose = verbose
        self.debug = debug
        #############################################
        
        # Other useful inits
        self.global_start_time = time.perf_counter()
        self.local_start_time = time.perf_counter()
        
        print('Options provided:')
        print('local_opts:')
        print(json.dumps(self.local_opts, sort_keys=True, indent=4))
        print('intermed_opts:')
        print(json.dumps( self.intermed_opts, sort_keys=True, indent=4))
        print('global_opts:')
        print(json.dumps( self.global_opts, sort_keys=True, indent=4))
        
        # The variables created during the fit
        self.X = None
        self.d_e = None
        self.neigh_dist = None
        self.neigh_ind = None
        self.nbrhd_graph = NbrhdGraph()
        self.LocalViews = local_views_.LocalViews()
        self.IntermedViews = intermed_views_.IntermedViews()
        self.GlobalViews = global_views_.GlobalViews()
        
    def log(self, s='', log_time=False):
        if self.verbose:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    def fit(self, X = None, d_e = None, cond_num=None, local_subspaces=None):
        """Run the algorithm. Either X or d_e must be supplied.
        
        Parameters
        ---------
        X : array shape (n_samples, n_features)
            A 2d array containing data representing a manifold.
        d_e : array shape (n_samples, n_samples)
            A square numpy matrix representing the geodesic distance
            between each pair of points on the manifold.
            
        Returns
        -------
        y : array shape (n_samples, d)
            The embedding of the data in lower dimension.
        """
        assert X is not None or d_e is not None, "Either X or d_e should be provided."
        
        if d_e is None:
            data = X
        else:
            data = d_e.copy()

        self.fit_nbrhd_graph(data, local_subspaces=local_subspaces, cond_num=cond_num)

        # Construct low dimensional local views
        self.fit_local_views(data)
        
        if self.exit_at == 'local_views':
            return
        
        # Construct intermediate views
        self.fit_intermediate_views()
        if self.exit_at == 'intermed_views':
            return
        
        # Construct Global views
        self.fit_global_views()
        return self.GlobalViews.y_final

    def fit_nbrhd_graph(self, data, local_subspaces=None, cond_num=None):
        self.nbrhd_graph = NbrhdGraph(
            k_nn=self.local_opts['k_nn0'],
            metric=self.local_opts['metric']
        )
        self.nbrhd_graph.fit(data)
        if local_subspaces is not None:
            raise Exception("Not tested yet.")
            # self.nbrhd_graph.compute_principal_angles(local_subspaces)

        if cond_num is not None:
            raise Exception("Not tested yet.")
            # self.nbrhd_graph.induce_connections(data, cond_num)
            # old_k = self.local_opts['k']
            # self.local_opts['k'] = self.nbrhd_graph.k_nn
            # self.local_opts['k_for_pca'] = max(self.local_opts['k_for_pca'], self.local_opts['k'])
            # self.global_opts['k'] = self.local_opts['k']

    def fit_local_views(self, data):
        self.LocalViews = local_views_.LocalViews(self.exit_at, self.verbose, self.debug)
        self.LocalViews.fit(self.d, data, copy.deepcopy(self.nbrhd_graph), self.local_opts)
        self.LocalViews.postprocess(self.nbrhd_graph, self.local_opts)
        

    def fit_intermediate_views(self):
        self.IntermedViews = intermed_views_.IntermedViews(self.exit_at, self.verbose, self.debug)
        self.IntermedViews.fit(self.d, self.nbrhd_graph.sparse_matrix(symmetrize=True), self.LocalViews.U,
                              self.LocalViews.local_param_post,
                              self.intermed_opts)
        
    def fit_global_views(self):
        self.GlobalViews = global_views_.GlobalViews(self.exit_at, self.verbose, self.debug)
        intermed_param = copy.deepcopy(self.IntermedViews.intermed_param)
        self.GlobalViews.fit(self.d, self.nbrhd_graph.sparse_matrix(symmetrize=True), self.IntermedViews.Utilde, self.IntermedViews.C, self.IntermedViews.c,
                            self.IntermedViews.n_C, intermed_param,
                            self.global_opts, self.vis, self.vis_opts)
        
    def get_state(self):
        state = {
            'local_opts': self.local_opts,
            'intermed_opts': self.intermed_opts,
            'global_opts': self.global_opts,
            'd': self.d,
            'vis_opts': self.vis_opts,
            'exit_at': self.exit_at,
            'verbose': self.verbose,
            'debug': self.debug,
            'n_proc': self.n_proc,
            'nbrhd_graph': self.nbrhd_graph.get_state(),
            'LocalViews': self.LocalViews.get_state(),
            'IntermedViews': self.IntermedViews.get_state(),
            'GlobalViews': self.GlobalViews.get_state(),
        }
        return state

    def set_state(self, state):
        self.local_opts = state['local_opts']
        self.intermed_opts = state['intermed_opts']
        self.global_opts = state['global_opts']
        self.d = state['d']
        self.n_proc = state['n_proc']
        self.vis_opts = state['vis_opts']
        self.exit_at = state['exit_at']
        self.verbose = state['verbose']
        self.debug = state['debug']
        self.nbrhd_graph.set_state(state['nbrhd_graph'])
        self.LocalViews.set_state(state['LocalViews'])
        self.IntermedViews.set_state(state['IntermedViews'])
        self.GlobalViews.set_state(state['GlobalViews'])

def create_buml_obj(buml_obj_state):
    buml_obj = BUML()
    buml_obj.set_state(buml_obj_state)
    return buml_obj