import pdb
import time
import numpy as np
import copy

from .util_ import print_log, compute_incidence_matrix_in_embedding, Param
from .global_reg_ import spectral_alignment, rgd_alignment, compute_alignment_err, compute_far_off_points

from matplotlib import pyplot as plt
from .global_view_algos import compute_seq_of_views, compute_color_of_pts_on_tear, procrustes_init, procrustes_final

class GlobalViews:
    def __init__(self, exit_at=None, print_logs=True, debug=False):
        self.exit_at = exit_at
        self.print_logs = print_logs
        self.debug = debug
        
        self.y_init = None
        self.color_of_pts_on_tear_init = None
        self.y_final = None
        self.color_of_pts_on_tear_final = None
        self.tracker = {}
        self.intermed_param_final = Param()
        
        self.local_start_time = time.perf_counter()
        self.global_start_time = time.perf_counter()
        
        # saved only when debug is True
        self.n_Utilde_Utilde = None
        self.seq_of_intermed_views_in_cluster = None
        self.parents_of_intermed_views_in_cluster = None
        self.cluster_of_intermed_view = None
        self.y_seq_init = None
        self.y_spec_init = None
        self.y_refined_at = []
        self.y_2_refined_at = []
        self.color_of_pts_on_tear_at = []
        self.it0 = 0
        self.refinement_converged = False

    def get_state(self):
        state = {
            'exit_at': self.exit_at,
            'print_logs': self.print_logs,
            'debug': self.debug,
            'y_init': self.y_init,
            'color_of_pts_on_tear_init': self.color_of_pts_on_tear_init,
            'y_final': self.y_final,
            'color_of_pts_on_tear_final': self.color_of_pts_on_tear_final,
            'tracker': self.tracker,
            'y_refined_at': self.y_refined_at,
            'y_2_refined_at': self.y_2_refined_at,
            'color_of_pts_on_tear_at': self.color_of_pts_on_tear_at,
            'it0': self.it0,
            'refinement_converged': self.refinement_converged,
            'n_Utilde_Utilde': self.n_Utilde_Utilde,
            'seq_of_intermed_views_in_cluster': self.seq_of_intermed_views_in_cluster,
            'parents_of_intermed_views_in_cluster': self.parents_of_intermed_views_in_cluster,
            'cluster_of_intermed_view': self.cluster_of_intermed_view,
            'y_seq_init': self.y_seq_init,
            'y_spec_init': self.y_spec_init,
            'intermed_param_final': self.intermed_param_final.get_state()
        }
        return state
    
    def set_state(self, state):
        self.exit_at = state['exit_at']
        self.print_logs = state['print_logs']
        self.debug = state['debug']
        self.y_init = state['y_init']
        self.color_of_pts_on_tear_init = state['color_of_pts_on_tear_init']
        self.y_final = state['y_final']
        self.color_of_pts_on_tear_final = state['color_of_pts_on_tear_final']
        self.tracker = state['tracker']
        self.y_refined_at = state['y_refined_at']
        self.y_2_refined_at = state['y_2_refined_at']
        self.color_of_pts_on_tear_at = state['color_of_pts_on_tear_at']
        self.it0 = state['it0']
        self.refinement_converged = state['refinement_converged']
        self.n_Utilde_Utilde = state['n_Utilde_Utilde']
        self.seq_of_intermed_views_in_cluster = state['seq_of_intermed_views_in_cluster']
        self.parents_of_intermed_views_in_cluster = state['parents_of_intermed_views_in_cluster']
        self.cluster_of_intermed_view = state['cluster_of_intermed_view']
        self.y_seq_init = state['y_seq_init']
        self.y_spec_init = state['y_spec_init']
        self.intermed_param_final.set_state(state['intermed_param_final'])
        
    def log(self, s='', log_time=False):
        if self.print_logs:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
            
    def fit(self, d, d_e, Utilde, C, c, n_C, intermed_param, global_opts, vis, vis_opts):
        self.intermed_param_final = intermed_param
        print('Using', global_opts['align_transform'], 'transforms for alignment.')
        if global_opts['align_transform'] == 'rigid':
            # Compute |Utilde_{mm'}|
            n_Utilde_Utilde = Utilde.dot(Utilde.transpose())
            n_Utilde_Utilde.setdiag(0)
            
            # Compute sequence of intermedieate views
            seq_of_intermed_views_in_cluster, \
            parents_of_intermed_views_in_cluster, \
            cluster_of_intermed_view = compute_seq_of_views(d, Utilde, n_C, 
                                                            n_Utilde_Utilde,
                                                            intermed_param, global_opts)
            
            if global_opts['add_dim']:
                intermed_param.add_dim = True
                d = d + 1
            # Visualize embedding before init
            if global_opts['vis_before_init']:
                self.vis_embedding_(d, intermed_param, C, Utilde,
                                  n_Utilde_Utilde, global_opts, vis,
                                  vis_opts, title='Before_Init')
            
            # Compute initial embedding
            y_init, color_of_pts_on_tear_init = self.compute_init_embedding(d, d_e, Utilde, n_Utilde_Utilde, intermed_param,
                                                                            seq_of_intermed_views_in_cluster,
                                                                            parents_of_intermed_views_in_cluster,
                                                                            C, c, vis, vis_opts, global_opts)

            self.y_init = y_init
            self.color_of_pts_on_tear_init = color_of_pts_on_tear_init
            
            if global_opts['refine_algo_name']:
                y_final,\
                color_of_pts_on_tear_final = self.compute_final_embedding(y_init, d, d_e, Utilde, C, c, intermed_param,
                                                                          n_Utilde_Utilde, seq_of_intermed_views_in_cluster,
                                                                          global_opts, vis, vis_opts)
                self.y_final = y_final
                self.color_of_pts_on_tear_final = color_of_pts_on_tear_final
            
        elif global_opts['align_transform'] == 'affine':
            print('align_transform == affine, not implemented.')
        
        if self.debug:
            self.n_Utilde_Utilde = n_Utilde_Utilde
            self.seq_of_intermed_views_in_cluster = seq_of_intermed_views_in_cluster
            self.parents_of_intermed_views_in_cluster = parents_of_intermed_views_in_cluster
            self.cluster_of_intermed_view = cluster_of_intermed_view
            
    
    def vis_embedding_(self, y, d, intermed_param, c, C, Utilde,
                      n_Utilde_Utilde, global_opts, vis,
                      vis_opts, title='', color_of_pts_on_tear=None,
                      Utilde_t=None):
        M,n = Utilde.shape
        if global_opts['color_tear']:
            if (color_of_pts_on_tear is None) and global_opts['to_tear']:
                color_of_pts_on_tear = compute_color_of_pts_on_tear(y, Utilde, C, global_opts,
                                                                         n_Utilde_Utilde)
            if global_opts['to_tear']:
                color_of_pts_on_tear = color_of_pts_on_tear[:,global_opts['tear_color_eig_inds']]
        else:
            color_of_pts_on_tear = None
            
        vis.global_embedding(y, vis_opts['c'], vis_opts['cmap_interior'],
                                  color_of_pts_on_tear, vis_opts['cmap_boundary'],
                                  title)
            
        # if color_of_pts_on_tear is not None:
        #     pts_on_tear = np.nonzero(~np.isnan(color_of_pts_on_tear))[0]
        #     y_ = []
        #     ind_ = []
        #     #color_of_pts_on_tear = np.zeros(n)+np.nan
        #     color_of_pts_on_tear_ = []
        #     for i in range(pts_on_tear.shape[0]):
        #         k = pts_on_tear[i]
        #         for m in Utilde[:,k].nonzero()[0].tolist():
        #             if m == c[k]:
        #                 continue
        #             y_.append(intermed_param.eval_({'view_index': m, 'data_mask': np.array([k])}))
        #             ind_.append(k)
        #             color_of_pts_on_tear_.append(color_of_pts_on_tear[k])
        #     ind_ = np.array(ind_)
        #     color_of_pts_on_tear_ = np.array(color_of_pts_on_tear_)
        #     if len(y_):
        #         y_ = np.concatenate(y_, axis=0)
        #         y_ = np.concatenate([y,y_], axis=0)
        #         color_of_pts_on_tear_ = np.concatenate([color_of_pts_on_tear, color_of_pts_on_tear_], axis=0)
        #         if vis_opts['c'] is not None:
        #             c_ = vis_opts['c'][ind_]
        #             c_ = np.concatenate([vis_opts['c'],c_], axis=0)
        #         else:
        #             c_ = None
        #         vis.global_embedding(y_,c_, vis_opts['cmap_interior'],
        #                               color_of_pts_on_tear_, vis_opts['cmap_boundary'],
        #                               title)
        #     else:
        #         vis.global_embedding(y, vis_opts['c'], vis_opts['cmap_interior'],
        #                           color_of_pts_on_tear, vis_opts['cmap_boundary'],
        #                           title)
        # else:
        #     vis.global_embedding(y, vis_opts['c'], vis_opts['cmap_interior'],
        #                           color_of_pts_on_tear, vis_opts['cmap_boundary'],
        #                           title)
        plt.show()
        return color_of_pts_on_tear, y
    
    def vis_embedding(self, y, vis, vis_opts, color_of_pts_on_tear=None, title=''):
        vis.global_embedding(y, vis_opts['c'], vis_opts['cmap_interior'],
                              color_of_pts_on_tear, vis_opts['cmap_boundary'],
                              title)
        plt.show()
        
    def add_spacing_bw_clusters(self, y, d, seq_of_intermed_views_in_cluster,
                                intermed_param, C):
        n_clusters = len(seq_of_intermed_views_in_cluster)
        if n_clusters == 1:
            return
        
        M,n = C.shape
            
        # arrange connected components nicely
        # spaced on horizontal (x) axis
        offset = 0
        self.cluster_label = np.zeros(y.shape[0], dtype=int)-1
        for i in range(n_clusters):
            seq = seq_of_intermed_views_in_cluster[i]
            pts_in_cluster_i = np.where(C[seq,:].sum(axis=0))[1]
            self.cluster_label[pts_in_cluster_i] = i
            # make the x coordinate of the leftmost point
            # of the ith cluster to be equal to the offset
            if i > 0:
                offset_ = np.min(y[pts_in_cluster_i,0])
                intermed_param.v[seq,0] += offset - offset_
                y[pts_in_cluster_i,0] += offset - offset_
            
            # recompute the offset as the x coordinate of
            # rightmost point of the current cluster
            offset = 1.25*np.max(y[pts_in_cluster_i,0])
    
    def compute_init_embedding(self, d, d_e, Utilde, n_Utilde_Utilde, intermed_param,
                               seq_of_intermed_views_in_cluster,
                               parents_of_intermed_views_in_cluster,
                               C, c, vis, vis_opts, global_opts,
                               print_prop = 0.25):
        M,n = Utilde.shape
        print_freq = int(M*print_prop)

        intermed_param.T = np.tile(np.eye(d),[M,1,1])
        intermed_param.v = np.zeros((M,d))
        y = np.zeros((n,d))

        n_clusters = len(seq_of_intermed_views_in_cluster)
        global_opts['far_off_points'] = compute_far_off_points(d_e, global_opts, force_compute=True)

        # Boolean array to keep track of already visited views
        is_visited_view = np.zeros(M, dtype=bool)
        init_algo = global_opts['init_algo_name']
        self.log('Computing initial embedding using: ' + init_algo + ' algorithm', log_time=True)
        if 'procrustes' == init_algo:
            for i in range(n_clusters):
                # First view global embedding is same as intermediate embedding
                seq = seq_of_intermed_views_in_cluster[i]
                rho = parents_of_intermed_views_in_cluster[i]
                seq_0 = seq[0]
                is_visited_view[seq_0] = True
                y[C[seq_0,:].indices,:] = intermed_param.eval_({'view_index': seq_0,
                                                                'data_mask': C[seq_0,:].indices})
                y, is_visited_view = procrustes_init(seq, rho, y, is_visited_view,
                                            d, Utilde, n_Utilde_Utilde,
                                            C, c, intermed_param,
                                            global_opts, print_freq)
            
            if self.debug:
                self.y_seq_init = y
        
        if 'spectral' == init_algo:
            y_2, y = spectral_alignment(y, d, Utilde,
                                         C, intermed_param, global_opts,
                                         seq_of_intermed_views_in_cluster)
            if self.debug:
                self.y_spec_init = y
                self.y_spec_init_2 = y_2
                
        
        self.log('Embedding initialized.', log_time=True)
        self.tracker['init_computed_at'] = time.perf_counter()
        if global_opts['compute_error']:
            self.log('Computing error.')
            err = compute_alignment_err(d, Utilde, intermed_param, Utilde.count_nonzero())
            self.log('Alignment error: %0.3f' % (err/Utilde.nnz), log_time=True)
            self.tracker['init_err'] = err
        
        self.add_spacing_bw_clusters(y, d, seq_of_intermed_views_in_cluster,
                                    intermed_param, C)
        
        # Visualize the initial embedding
        color_of_pts_on_tear, y = self.vis_embedding_(y, d, intermed_param, c, C, Utilde,
                                                  n_Utilde_Utilde, global_opts, vis,
                                                  vis_opts, title='Init')
        if self.debug:
            self.intermed_param_init = copy.deepcopy(intermed_param)
        #intermed_param.y = y
        return y, color_of_pts_on_tear

    def compute_final_embedding(self, y, d, d_e, Utilde, C, c, intermed_param, n_Utilde_Utilde,
                                seq_of_intermed_views_in_cluster,
                                global_opts, vis, vis_opts, reset=True):
        M,n = Utilde.shape
        y = y.copy()
        np.random.seed(42) # for reproducbility

        max_iter0 = global_opts['max_iter']
        max_iter1 = global_opts['max_internal_iter']
        refine_algo = global_opts['refine_algo_name']
        patience_ctr = global_opts['patience']
        tol = global_opts['tol']
        prev_err = None
        prev_edges = None
        Utilde_t = Utilde.copy()
        solver = None
        
        if reset:
            self.y_refined_at = []
            self.y_2_refined_at = []
            self.color_of_pts_on_tear_at = []
            self.tracker['refine_iter_start_at'] = []
            self.tracker['refine_iter_done_at'] = []
            self.tracker['refine_err_at_iter'] = []
            self.tracker['|E(Gamma_t)|'] = []
            self.it0 = 0
            self.refinement_converged = False
        else:
            self.log('Reset is False. Starting from where left off', log_time=True)
            if self.refinement_converged:
                self.log('Refinement had already converged.', log_time=True)
                return self.y_final, self.color_of_pts_on_tear_final
        
        if global_opts['to_tear']:
            Utildeg = compute_incidence_matrix_in_embedding(y, C, global_opts['k'], global_opts['nu'], global_opts['metric'])
        else:
            Utildeg = None
        
        color_of_pts_on_tear = None
        # Refine global embedding y
        for it0 in range(max_iter0):
            self.tracker['refine_iter_start_at'].append(time.perf_counter())
            self.log('Refining with ' + refine_algo + ' algorithm for ' + str(max_iter1) + ' iterations.')
            self.log('repel_by:' + str(global_opts['repel_by']))
            self.log('max_var_by:' + str(global_opts['max_var_by']))
            self.log('Refinement iteration: %d' % self.it0, log_time=True)
            
            global_opts['far_off_points'] = compute_far_off_points(d_e, global_opts)
            
            if global_opts['to_tear']:
                Utilde_t = Utildeg.multiply(Utilde)
                Utilde_t.eliminate_zeros()
            
            if refine_algo == 'procrustes':
                y = procrustes_final(y, d, Utilde_t, C, intermed_param, 
                                     seq_of_intermed_views_in_cluster, global_opts)
            elif refine_algo == 'rgd':
                y_2, y = rgd_alignment(y, d, Utilde_t, C, intermed_param, global_opts)
            elif refine_algo == 'spectral':
                y_2, y = spectral_alignment(y, d, Utilde_t,
                                             C, intermed_param, global_opts,
                                             seq_of_intermed_views_in_cluster)
                
            self.log('Done.', log_time=True)
            self.tracker['refine_iter_done_at'].append(time.perf_counter())
            time_elapsed = self.tracker['refine_iter_done_at'][-1] - self.tracker['refine_iter_start_at'][-1]
            print('### Last iter of refinement took %0.1f seconds.' % (time_elapsed), flush=True)

            if global_opts['compute_error'] or (it0 == max_iter0-1):
                self.log('Computing error.')
                err = compute_alignment_err(d, Utilde_t, intermed_param, Utilde.count_nonzero(),
                                            far_off_points=global_opts['far_off_points'],
                                            repel_by=global_opts['repel_by'],
                                            beta=global_opts['beta'])
                self.tracker['refine_err_at_iter'].append(err)
                E_Gamma_t = Utilde_t.nnz
                self.tracker['|E(Gamma_t)|'].append(E_Gamma_t)
                err = err/E_Gamma_t
                self.log('Alignment error: %0.6f' % (err), log_time=True)
                if prev_err is not None:
                    if (np.abs(err-prev_err)/(prev_err+1e-12) < tol) and\
                        (np.abs(E_Gamma_t-prev_edges)/(prev_edges+1e-12) < tol):
                        patience_ctr -= 1
                    else:
                        patience_ctr = global_opts['patience']
                prev_err = err
                prev_edges = E_Gamma_t
                
            self.add_spacing_bw_clusters(y, d, seq_of_intermed_views_in_cluster,
                                         intermed_param, C)
            
            # If to tear the closed manifolds
            if global_opts['to_tear']:
                # Compute |Utildeg_{mm'}|
                Utildeg = compute_incidence_matrix_in_embedding(y, C, global_opts['k'], global_opts['nu'], global_opts['metric'])
                if global_opts['color_tear']:
                    color_of_pts_on_tear = compute_color_of_pts_on_tear(y, Utilde, C, global_opts,
                                                                            n_Utilde_Utilde,
                                                                            Utildeg)
                
            if self.debug:
                self.y_refined_at.append(y)
                self.y_2_refined_at.append(y_2)
                self.color_of_pts_on_tear_at.append(color_of_pts_on_tear)
            
            #intermed_param.y = y
             
            # Visualize the current embedding
            _, y = self.vis_embedding_(y, d, intermed_param, c, C, Utilde,
                                      n_Utilde_Utilde, global_opts, vis,
                                      vis_opts, title='Iter_%d' % self.it0,
                                      color_of_pts_on_tear=color_of_pts_on_tear,
                                      Utilde_t=Utilde_t)
            
            self.it0 += 1
            if global_opts['compute_error'] and (patience_ctr <= 0):
                self.refinement_converged = True
                break
            
            if (global_opts['repel_by'] is not None):
                global_opts['repel_by'] *= global_opts['repel_decay']
            if (global_opts['max_var_by'] is not None):
                global_opts['max_var_by'] *= global_opts['max_var_decay']
        return y, color_of_pts_on_tear