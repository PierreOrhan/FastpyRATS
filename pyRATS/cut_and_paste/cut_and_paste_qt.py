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

class CutAndPaste:
    def __init__(
        self,
        buml_obj,
        metadata_fname=None,
        save_progress_dir='',
        cmap_interior='summer',
        cmap_tear='jet',
        max_refinement_iter=10,
        backend='QtAgg',
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
        self.backend = backend

    def fit(self):
        while True:
            # Show current embedding and get face colors of points in embeddings
            print('Current embedding:')
            y = self.buml_obj.GlobalViews.y_final.copy()
            to_quit, y_face_color = self.show_current_embedding()
            if to_quit:
                break

            self.init_metadata_for_current_iter()

            # Cut: Select clusters to move
            print('Cut:')
            if self.metadata[self.cur_iter]['selected_cluster_mask'] is None:
                selected_cluster_mask = self.select_clusters_to_move(y_face_color=y_face_color)
                self.metadata[self.cur_iter]['selected_cluster_mask'] = selected_cluster_mask
            else:
                selected_cluster_mask = self.metadata[self.cur_iter]['selected_cluster_mask']

            # Points in the selected clusters. Moving clusters = moving points in them
            points_in_selected_clusters = selected_cluster_mask[self.buml_obj.IntermedViews.c]

            # Find a convex polygon that represents the points in the selected clusters
            if self.metadata[self.cur_iter]['init_polygon'] is None:
                init_polyg = get_convex_covering_polygon(
                    y[points_in_selected_clusters,:]
                )
                self.metadata[self.cur_iter]['init_polygon'] = init_polyg
            else:
                init_polyg = self.metadata[self.cur_iter]['init_polygon']

            self.show_init_polygon(init_polyg, y_face_color=y_face_color)

            # Move the clusters by moving the polygon
            if self.metadata[self.cur_iter]['final_polygon'] is None:
                final_polyg = self.move_clusters(
                    init_polyg,
                    points_in_selected_clusters,
                    y_face_color=y_face_color
                )
                self.metadata[self.cur_iter]['final_polygon'] = final_polyg
            else:
                final_polyg = self.metadata[self.cur_iter]['final_polygon']

            # Use the final location of the polygon to compute the
            # final location of the points in the selected clusters
            y_new = self.recompute_embedding(y, points_in_selected_clusters, init_polyg, final_polyg)

            self.show_final_polygon_with_embedding(y_new, y_face_color, final_polyg)

            # Paste = Stitch/refine the new embedding 
            matplotlib.use(self.backend)
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
            
            if self.save_progress_dir:
                new_emb_info = {
                    'y': buml_obj.GlobalViews.y_final,
                    'color_of_pts_on_tear': buml_obj.GlobalViews.color_of_pts_on_tear_final,
                    'buml_obj_state': buml_obj.get_state(),
                }
                util_.save(
                    self.save_progress_dir,
                    'buml_obj_and_metadata_after_iter='+str(self.cur_iter)+'.dat',
                    [new_emb_info, self.metadata]
                )
            self.cur_iter += 1
            

    def init_metadata_for_current_iter(self):
        if len(self.metadata) <= self.cur_iter:
            self.metadata.append({
                'selected_cluster_mask': None,
                'init_polygon': None,
                'final_polygon': None
            })

    

    def show_current_embedding(self):
        buml_obj = self.buml_obj
        cmap_interior = self.cmap_interior
        cmap_tear = self.cmap_tear

        matplotlib.use(self.backend)
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

        to_quit = waitforbuttonpress(
            title_msg=' '.join(['Press', ENTER_KEY, 'to close. Press', QUIT_KEY, 'to quit.']),
            button=ENTER_KEY
        )
        y_face_color = interior_handle._facecolors
        y_face_color[pts_on_tear,:] = tear_handle.get_facecolors()
        return to_quit, y_face_color
    
    # There is one to one correspondence between
    # clusters/partitions and local views
    def select_clusters_to_move(self, y_face_color=None):
        buml_obj = self.buml_obj
        y = buml_obj.GlobalViews.y_final.copy()
        selected_pts_indices = select_points(y, y_face_color)
        selected_pts_mask = util_.create_mask(selected_pts_indices, y.shape[0])
        cluster_label = buml_obj.IntermedViews.c
        avoid_clusters = np.unique(cluster_label[~selected_pts_mask])
        n_clusters = np.max(cluster_label)+1
        avoid_clusters_mask = util_.create_mask(avoid_clusters, n_clusters)
        return ~avoid_clusters_mask

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
    
    def show_init_polygon(self, init_polyg, y_face_color=None):
        buml_obj = self.buml_obj
        y = buml_obj.GlobalViews.y_final.copy()

        matplotlib.use(self.backend)
        _, ax = plt.subplots()
        maximize_window()
        ax.scatter(y[:,0], y[:,1], c=y_face_color, s=20)
        polyg = Polygon(init_polyg, color=[1,0,0,0.1])
        ax.add_patch(polyg)
        plt.axis('image')
        plt.axis('off')
        plt.tight_layout()
        plt.gca().set_rasterized(True)
        if self.save_progress_dir:
            save_fn = self.save_progress_dir + '/init_polyg_at_iter=' + str(self.cur_iter) + '.png'
            plt.savefig(save_fn, bbox_inches='tight', dpi=400)

        waitforbuttonpress(
            title_msg='Press Enter to Move Polygon or close the figure.',
            button=ENTER_KEY,
        )

    def move_clusters(self, init_polyg, points_in_selected_clusters, y_face_color=None):
        buml_obj = self.buml_obj
        y = buml_obj.GlobalViews.y_final.copy()

        matplotlib.use(self.backend)
        fig, ax = plt.subplots()
        maximize_window()
        pts_handle = ax.scatter(y[:,0], y[:,1], c=y_face_color, s=20)
        polyg = Polygon(init_polyg, color=[1,0,0,0.25])
        ax.add_patch(polyg)
        plt.axis('image')
        plt.axis('off')
        plt.tight_layout()
        plt.gca().set_rasterized(True)

        drpatch = DraggablePatch(fig, ax, polyg, pts_handle, y, points_in_selected_clusters)
        drpatch.connect()

        waitforbuttonpress(
            'Drag via cursor. Rotate via keys: o or p. Press enter once done.',
            ENTER_KEY
        )

        return polyg.get_xy()[:-1,:]

    def show_final_polygon_with_embedding(self, y_new, y_face_color, final_polyg, s=20):
        matplotlib.use(self.backend)
        fig, ax = plt.subplots()
        maximize_window()

        ax.scatter(y_new[:,0], y_new[:,1], c=y_face_color, s=s)
        polyg = Polygon(final_polyg, color=[1,0,0,0.25])
        ax.add_patch(polyg)
        ax.axis('image')
        ax.axis('off')
        plt.tight_layout()
        ax.set_rasterized(True)
        if self.save_progress_dir:
            save_fn = self.save_progress_dir + '/final_polyg_at_iter=' + str(self.cur_iter) + '.png'
            plt.savefig(save_fn, bbox_inches='tight', dpi=400)

        waitforbuttonpress(
            title_msg='Press Enter to close the figure.',
            button=ENTER_KEY,
        )

class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, fig, ax, collection, alpha_other=0.3):
        self.canvas = fig.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw()

class DraggablePatch:
    def __init__(self, fig, ax, patch, collection, y, sel_pts):
        self.canvas = fig.canvas
        self.ax = ax
        self.patch = patch
        self.press = None
        self.collection = collection
        self.y = y
        self.sel_pts = sel_pts

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidrotate = self.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        
    def on_key_press(self, event):
        r0 = None
        if event.key == 'p':
            r0 = matplotlib.transforms.Affine2D().rotate(np.pi/180)
        elif event.key == 'o':
            r0 = matplotlib.transforms.Affine2D().rotate(-np.pi/180)
            
        if r0 is not None:
            xy0 = self.patch.get_xy()[:-1,:]
            y_new = np.array(self.collection.get_offsets()).copy()
            mtx = r0.get_matrix()
            y_new[self.sel_pts,:] = y_new[self.sel_pts,:].dot(mtx[:2,:2].T) + mtx[-1,:2][None,:]
            self.collection.set_offsets(y_new)
            self.y = y_new.copy()
            self.patch.set_xy(xy0.dot(mtx[:2,:2].T) + mtx[-1,:2][None,:])
            
            self.canvas.draw()
            
    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.patch.axes:
            return
        contains, attrd = self.patch.contains(event)
        if not contains:
            return
        #print('event contains', self.patch.xy)
        self.press = self.patch.xy, (event.xdata, event.ydata)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None or event.inaxes != self.patch.axes:
            return
        xy0, (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        shift = np.array([dx,dy])[None,:]
        self.patch.set_xy(xy0 + shift)
        self.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        xy0, (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        shift = np.array([dx,dy])[None,:]

        y_new = np.array(self.collection.get_offsets().data)
        y_new[self.sel_pts,:] += shift
        self.collection.set_offsets(y_new)
        
        self.press = None
        self.canvas.draw()
        self.y = y_new.copy()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)
        self.canvas.mpl_disconnect(self.cidrotate)

def waitforbuttonpress(title_msg, button):
    plt.gca().set_title(title_msg)
    PRESSED_KEY = None
    def press(event):
        nonlocal PRESSED_KEY
        PRESSED_KEY = event.key

    plt.gcf().canvas.mpl_connect('key_press_event', press)
    while True:
        if plt.waitforbuttonpress(-1):
            if (PRESSED_KEY == button) or (PRESSED_KEY == QUIT_KEY):
                break

    plt.close()
    if PRESSED_KEY == QUIT_KEY:
        return 1
    return 0

def select_points(y, c=None, s=20):
    fig, ax = plt.subplots()
    maximize_window()
    pts_handle = ax.scatter(y[:,0], y[:,1], c=c, s=s)
    ax.axis('image')
    ax.set_title('Press Enter after selecting points by drawing a lasso')

    selected_pts_indices = None
    selector = SelectFromCollection(fig, ax, pts_handle)
    PRESSED_KEY = None
    def accept(event):
        nonlocal PRESSED_KEY
        nonlocal selected_pts_indices
        PRESSED_KEY = event.key
        if PRESSED_KEY == ENTER_KEY:
            selected_pts_indices = selector.ind
            selector.disconnect()
            fig.canvas.draw()

    plt.gcf().canvas.mpl_connect('key_press_event', accept)
    while True:
        if plt.waitforbuttonpress(-1):
            if PRESSED_KEY == ENTER_KEY:
                break

    plt.close()
    return selected_pts_indices

def get_convex_covering_polygon(y):
    hull = ConvexHull(y)
    return y[hull.vertices,:]