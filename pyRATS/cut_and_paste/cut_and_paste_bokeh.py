import matplotlib
import numpy as np
import matplotlib
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scipy.spatial import ConvexHull
from ..global_view_algos import compute_color_of_pts_on_tear
from .. import util_
from ..visualize_ import maximize_window

QUIT_KEY = 'q'
ENTER_KEY = 'enter'

def rgba_to_css(rgba):
    r, g, b, a = rgba
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2f})"

def rgba_arr_to_css_arr(rgba_arr):
    css_arr = []
    for i in range(rgba_arr.shape[0]):
        css_arr.append(rgba_to_css(rgba_arr[i,:]))
    return css_arr

class CutAndPaste:
    def __init__(
        self,
        buml_obj,
        metadata_fname=None,
        save_progress_dir='',
        cmap_interior='summer',
        cmap_tear='jet',
        max_refinement_iter=10,
        force_compute=False
    ):
        self.buml_obj = buml_obj
        self.pts = None
        if metadata_fname is None:
            self.metadata = []
        else:
            self.metadata = util_.read(metadata_fname)
        self.save_progress_dir = save_progress_dir
        if self.save_progress_dir:
            util_.makedirs(self.save_progress_dir)
        self.cur_iter = 0
        self.cmap_interior = cmap_interior
        self.cmap_tear = cmap_tear
        self.max_refinement_iter = max_refinement_iter
        self.force_compute = force_compute
        self.selected_pts_to_move = np.array([])

    def init_metadata_for_current_iter(self):
        if len(self.metadata) <= self.cur_iter:
            self.metadata.append({
                'selected_cluster_mask': None,
                'init_polygon': None,
                'final_polygon': None
            })

    def get_current_embedding_data(self):
        buml_obj = self.buml_obj
        cmap_interior = self.cmap_interior
        cmap_tear = self.cmap_tear

        matplotlib.use('Agg')
        y = self.buml_obj.GlobalViews.y_final.copy()
        color_of_pts_on_tear = compute_color_of_pts_on_tear(
            y,
            buml_obj.IntermedViews.Utilde,
            buml_obj.IntermedViews.C,
            buml_obj.global_opts,
            buml_obj.GlobalViews.n_Utilde_Utilde
        )
        pts_on_tear = ~np.isnan(color_of_pts_on_tear[:,-1])

        interior_handle, tear_handle = buml_obj.vis.global_embedding_for_gui(
            y, y[:,0], cmap0=cmap_interior,
            color_of_pts_on_tear=color_of_pts_on_tear[:,-1],
            cmap1=cmap_tear,
            set_title=True,
            figsize=(3,3), s=20
        )
        if self.save_progress_dir:
            save_fn = self.save_progress_dir + '/reg_tear_y_at_iter=' + str(self.cur_iter) + '.png'
            plt.savefig(save_fn, bbox_inches='tight', dpi=400)

        y_face_color = interior_handle._facecolors
        y_face_color[pts_on_tear,:] = tear_handle.get_facecolors()

        self.y_face_color = y_face_color

        source_data_dict = dict(
            x=y[:,0], y=y[:,1],
            color=rgba_arr_to_css_arr(y_face_color)
        )
        return source_data_dict

    def get_encolsing_polygon_for_selected_pts(self, selected_pts_indices):
        selected_cluster_mask = self.select_clusters_to_move(selected_pts_indices)
        points_in_selected_clusters = selected_cluster_mask[self.buml_obj.IntermedViews.c]
        y = self.buml_obj.GlobalViews.y_final.copy()
        init_polyg = get_convex_covering_polygon(y[points_in_selected_clusters,:])
        patch_source_data_dict = dict(x=init_polyg[:,0], y=init_polyg[:,1])
        return patch_source_data_dict
    
    # There is one to one correspondence between
    # clusters/partitions and local views
    def cut_clusters_to_move(self, selected_pts_indices):
        self.init_metadata_for_current_iter()
        if self.force_compute or (self.metadata[self.cur_iter]['selected_cluster_mask'] is None):
            selected_cluster_mask = self.select_clusters_to_move(selected_pts_indices)
            self.metadata[self.cur_iter]['selected_cluster_mask'] = selected_cluster_mask
        else:
            selected_cluster_mask = self.metadata[self.cur_iter]['selected_cluster_mask']
        points_in_selected_clusters = np.where(selected_cluster_mask[self.buml_obj.IntermedViews.c])[0]
        self.selected_pts_to_move = points_in_selected_clusters

        # Find a convex polygon that represents the points in the selected clusters
        y = self.buml_obj.GlobalViews.y_final.copy()
        if self.force_compute or (self.metadata[self.cur_iter]['init_polygon'] is None):
            init_polyg = get_convex_covering_polygon(y[points_in_selected_clusters,:])
            self.metadata[self.cur_iter]['init_polygon'] = init_polyg
        else:
            init_polyg = self.metadata[self.cur_iter]['init_polygon']

        if self.save_progress_dir:
            self.save_polygon(y, self.y_face_color, init_polyg, stage='init')

        patch_source_data_dict = dict(x=init_polyg[:,0], y=init_polyg[:,1])
        return patch_source_data_dict

    def select_clusters_to_move(self, selected_pts_indices):
        cluster_label = self.buml_obj.IntermedViews.c
        selected_pts_mask = util_.create_mask(selected_pts_indices, len(cluster_label))
        avoid_clusters = np.unique(cluster_label[~selected_pts_mask])
        n_clusters = np.max(cluster_label)+1
        avoid_clusters_mask = util_.create_mask(avoid_clusters, n_clusters)
        return ~avoid_clusters_mask

    def paste_clusters(self, final_polyg):
        if self.force_compute or (self.metadata[self.cur_iter]['final_polygon'] is None):
            self.metadata[self.cur_iter]['final_polygon'] = final_polyg
        else:
            final_polyg = self.metadata[self.cur_iter]['final_polygon']

        # Use the final location of the polygon to compute the
        # final location of the points in the selected clusters
        selected_cluster_mask = self.metadata[self.cur_iter]['selected_cluster_mask']
        points_in_selected_clusters = selected_cluster_mask[self.buml_obj.IntermedViews.c]
        y = self.buml_obj.GlobalViews.y_final.copy()
        y_new = self.recompute_embedding(
            y, points_in_selected_clusters,
            self.metadata[self.cur_iter]['init_polygon'],
            final_polyg
        )
        if self.save_progress_dir:
            self.save_polygon(y_new, self.y_face_color, final_polyg, stage='final')
        
        self.finish_pasting(y_new)
        # reset
        self.selected_pts_to_move = np.array([])
        return self.get_current_embedding_data()

    def finish_pasting(self, y_new):
        matplotlib.use('Agg')
        print('Refining...', flush=True)
        buml_obj = self.buml_obj
        buml_obj.global_opts['max_iter'] = self.max_refinement_iter
        y_final, color_of_pts_on_tear_final = buml_obj.GlobalViews.compute_final_embedding(
            y_new, buml_obj.d, buml_obj.d_e, buml_obj.IntermedViews.Utilde,
            buml_obj.IntermedViews.C, buml_obj.IntermedViews.c,
            buml_obj.GlobalViews.intermed_param_final, 
            buml_obj.GlobalViews.n_Utilde_Utilde,
            buml_obj.GlobalViews.seq_of_intermed_views_in_cluster,
            buml_obj.global_opts,
            buml_obj.vis, buml_obj.vis_opts, reset=True
        )
        buml_obj.GlobalViews.y_final = y_final
        buml_obj.GlobalViews.color_of_pts_on_tear_final = color_of_pts_on_tear_final
        plt.close('all')
        
        self.save_buml_obj('buml_obj_and_metadata_after_iter='+str(self.cur_iter)+'.dat')
        self.cur_iter += 1

    def save_buml_obj(self, fname):
        if self.save_progress_dir:
            new_emb_info = {
                'y': self.buml_obj.GlobalViews.y_final,
                'color_of_pts_on_tear': self.buml_obj.GlobalViews.color_of_pts_on_tear_final,
                'buml_obj_state': self.buml_obj.get_state(),
            }
            util_.save(self.save_progress_dir, fname, [new_emb_info, self.metadata])

    def recompute_embedding(self, y, points_in_moving_clusters, init_polyg, final_polyg):
        # init_polyg_centroid = np.mean(init_polyg, axis=0)
        # final_polyg_centroid = np.mean(final_polyg, axis=0)
        # t = final_polyg_centroid - init_polyg_centroid
        # final_polyg_centered = final_polyg - final_polyg_centroid[None,:]
        # init_polyg_centered = init_polyg - init_polyg_centroid[None,:]
        # U, S, VT = np.linalg.svd(final_polyg_centered.T.dot(init_polyg_centered))
        # O = VT.dot(U.T) # TODO: check
        O, t = util_.procrustes(init_polyg, final_polyg)
        y = y.copy()
        y[points_in_moving_clusters,:] = y[points_in_moving_clusters,:].dot(O) + t[None,:]
        return y
    
    def save_polygon(self, y, y_face_color, y_polyg, stage, s=20):
        assert stage in ['init', 'final']
        matplotlib.use('Agg')
        _, ax = plt.subplots()
        maximize_window()
        ax.scatter(y[:,0], y[:,1], c=y_face_color, s=s)
        polyg = Polygon(y_polyg, color=[1,0,0,0.25])
        ax.add_patch(polyg)
        plt.axis('image')
        plt.axis('off')
        plt.tight_layout()
        ax.set_rasterized(True)
        save_fn = self.save_progress_dir + '/' + stage + '_polyg_at_iter=' + str(self.cur_iter) + '.png'
        plt.savefig(save_fn, bbox_inches='tight', dpi=400)

def get_convex_covering_polygon(y):
    hull = ConvexHull(y)
    return y[hull.vertices,:]