#To run the app, execute
# bokeh serve cut_and_paste_app.py --allow-websocket-origin=localhost:5006
# then go to http://localhost:5006

from bokeh.plotting import figure, curdoc
from bokeh.models import LassoSelectTool, TapTool, TextInput, Button, Div, ColumnDataSource, TextAreaInput, Range1d, CustomJS
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.events import PanStart, Pan, PanEnd
import ast

import sys
import numpy as np
from pyRATS.cut_and_paste import cut_and_paste_bokeh
from pyRATS import util_, buml_

REG_TEAR_TAG = 'reg_tear'

# Spinner using CSS
################################################################
spinner = Div(text="""
<div id="spinner" style="display:none;">
  <div class="loader"></div>
</div>

<style>
.loader {
  border: 6px solid #f3f3f3;
  border-top: 6px solid #3498db;
  border-radius: 50%;
  width: 15px;
  height: 15px;
  animation: spin 1s linear infinite;
  margin: 5px auto;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
""", width=25, height=20)
def show_spinner():
    spinner.text = spinner.text.replace('display:none;', 'display:block;')

def hide_spinner():
    spinner.text = spinner.text.replace('display:block;', 'display:none;')

# Loggers
################################################################
class BokehLogger:
    def __init__(self, div_widget):
        self.div = div_widget
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        self.div.text = f"""
        <div style="overflow:auto;">
            <pre>{self.buffer}</pre>
        </div>
        """

    def flush(self):
        pass

def get_scroll_js(div_name):
    scroll_js = CustomJS(code="""
        // Find the element by ID
        const el = document.querySelector('[data-name="{div_name}"]');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    """)
    return scroll_js


DIV_WIDTH = 900
DIV_HEIGHT = 150
log_div = Div(
    text="", width=DIV_WIDTH, height=DIV_HEIGHT,
    styles={'overflow-y': 'scroll', 'border': '1px solid black'},
    name='log_div'
)
log_div.js_on_change("text", get_scroll_js(log_div.name))
sys.stdout = BokehLogger(log_div)
print('<h3>Detailed logs are displayed here.</h3>')

main_log_div = Div(
    text="", width=DIV_WIDTH, height=DIV_HEIGHT,
    styles={'overflow-y': 'scroll', 'border': '1px solid black'},
    name='main_log_div'
)
main_log_div.js_on_change("text", get_scroll_js(main_log_div.name))
main_log = BokehLogger(main_log_div)
def print_main_log(message):
    main_log.write(message + '\n')

print_main_log('<h3>Short logs & instructions are displayed here. Detiled logs are printed at the end of the page.</h3>')
print_main_log('Input algorithm, hyperparameter and options, and then press start.')

# Text inputs
################################################################
gen_data_path_input = TextInput(
    title="Generated data path",
    value="/data2/dhruv/RATS_code/RATS/code/motor_cortex/notebooks/generated_data",
    width=550
)

algo_input = TextInput(title="Algorithm Name: (for ex: rats)", value="rats")
hyp_param_input = TextInput(title="Hyperparameter: (for ex: 40_5)", value="40_5")
options_input = TextAreaInput(
    title="Additional options:",
    value="{'force_compute': True, 'max_refinement_iter': 10, 'cmap_interior': 'summer', 'cmap_tear': 'jet'}",
    rows=5,
)

cut_and_paste_obj = None
rot_deg = 5
lasso_tool = LassoSelectTool()
tap_tool = TapTool()

update_max_fig_height_button = Button(label="Update height", button_type="success", disabled=True)
start_button = Button(label="Start", button_type="success")
cut_button = Button(label="Cut", button_type="success", disabled=True)
paste_button = Button(label="Paste", button_type="success", disabled=True)
save_button = Button(label="Save", button_type="success", disabled=True)
rotate_cw_button = Button(label="Rotate -" + str(rot_deg) + " deg", button_type="success", disabled=True)
rotate_ccw_button = Button(label="Rotate +" + str(rot_deg) + " deg", button_type="success", disabled=True)

source = ColumnDataSource(data=dict(x=[], y=[], color=[]))
patch_source = ColumnDataSource(data=dict(x=[], y=[]))

MAX_FIG_HEIGHT = 900
MAX_FIG_WIDTH = 900

max_fig_height_input = TextInput(title="Max figure height", value=str(MAX_FIG_HEIGHT))
fig = figure(
    title="Cut & Paste operation on the embedding",
    tools=[lasso_tool],
    x_axis_label="x",
    y_axis_label="y",
    width=MAX_FIG_HEIGHT,
    height=MAX_FIG_WIDTH,
    match_aspect=False
)
fig.scatter(
    'x', 'y',
    color='color',
    source=source,
    nonselection_alpha=0.3
)
fig.patch('x', 'y', source=patch_source, alpha=0.4, line_width=2, color='orange')

# Adjust figure size using the scatter x and y coordinates
##############################################################################
def adjust_figure_size():
    x_min = 1.25*np.min(source.data['x'])
    x_max = 1.25*np.max(source.data['x'])
    y_min = 1.25*np.min(source.data['y'])
    y_max = 1.25*np.max(source.data['y'])
    aspect_ratio = (x_max-x_min)/(y_max-y_min)
    print('Aspect ratio (x/y) of the embedding: ' + str(aspect_ratio))
    print('Adjusting figure width based on the aspect ratio.')
    print('MAX_FIG_WIDTH = ' + str(MAX_FIG_WIDTH))
    print('MAX_FIG_HEIGHT = ' + str(MAX_FIG_HEIGHT))
    if aspect_ratio >= 1:
        fig.height = int(1.25*MAX_FIG_WIDTH/aspect_ratio)
        fig.width = int(1.25*MAX_FIG_WIDTH)
    else:
        fig.height = int(1.25*MAX_FIG_HEIGHT)
        fig.width = int(1.25*aspect_ratio*MAX_FIG_HEIGHT)

    fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
    fig.y_range = Range1d(y_min, y_max, bounds=(y_min, y_max))

# Patch dragging logic
##############################################################################
dragging = {
    'active': False, 'start_x': None, 'start_y': None,
    'patch_start_x': None, 'patch_start_y': None
}
def on_pan_start(event):
    global cut_and_paste_obj
    selected_pts_to_move = cut_and_paste_obj.selected_pts_to_move
    if len(selected_pts_to_move) == 0:
        return
    dragging['active'] = True
    dragging['start_x'] = event.x
    dragging['start_y'] = event.y
    dragging['patch_start_x'] = np.array(patch_source.data['x']).copy()
    dragging['patch_start_y'] = np.array(patch_source.data['y']).copy()

def on_pan(event):
    global cut_and_paste_obj
    selected_pts_to_move = cut_and_paste_obj.selected_pts_to_move
    if not dragging['active'] or len(selected_pts_to_move) == 0:
        return

    dx = event.x - dragging['start_x']
    dy = event.y - dragging['start_y']
    new_x = np.array(patch_source.data['x'])+dx
    new_y = np.array(patch_source.data['y'])+dy
    x_out_of_bounds = np.any(new_x < fig.x_range.bounds[0]) | np.any(new_x > fig.x_range.bounds[1])
    y_out_of_bounds = np.any(new_y < fig.y_range.bounds[0]) | np.any(new_y > fig.y_range.bounds[1])
    if not (x_out_of_bounds or y_out_of_bounds):
        patch_source.data.update(
            x=(new_x).tolist(),
            y=(new_y).tolist()
        )
    dragging['start_x'] = event.x
    dragging['start_y'] = event.y

def on_pan_end(event):
    dragging['active'] = False
    global cut_and_paste_obj
    selected_pts_to_move = cut_and_paste_obj.selected_pts_to_move
    if len(selected_pts_to_move) == 0:
        return
    dx = np.mean(np.array(patch_source.data['x']) - dragging['patch_start_x'])
    dy = np.mean(np.array(patch_source.data['y']) - dragging['patch_start_y'])
    new_x = source.data['x']
    new_y = source.data['y']
    for i in selected_pts_to_move:
       new_x[i] += dx
       new_y[i] += dy
    source.data.update(x=new_x, y=new_y)
    
fig.on_event(PanStart, on_pan_start)
fig.on_event(Pan, on_pan)
fig.on_event(PanEnd, on_pan_end)

# Buttons
#############################################################################
# Rotate patch button
#######################################
def rotate_coords(x, y, cx, cy, angle):
    x1 = x - cx
    y1 = y - cy
    x_rot = x1 * np.cos(angle) - y1 * np.sin(angle) + cx
    y_rot = x1 * np.sin(angle) + y1 * np.cos(angle) + cy
    x_out_of_bounds = np.any(x_rot < fig.x_range.bounds[0]) | np.any(x_rot > fig.x_range.bounds[1])
    y_out_of_bounds = np.any(y_rot < fig.y_range.bounds[0]) | np.any(y_rot > fig.y_range.bounds[1])
    if x_out_of_bounds or y_out_of_bounds:
        return x, y
    return x_rot, y_rot

def rotate_patch(angle_degrees):
    global cut_and_paste_obj
    selected_pts_to_move = cut_and_paste_obj.selected_pts_to_move
    indices = selected_pts_to_move
    if len(indices)==0:
        return
    x = np.array(source.data['x'])
    y = np.array(source.data['y'])
    cx = np.mean(x[indices])
    cy = np.mean(y[indices])
    angle = np.deg2rad(angle_degrees)
    x[indices], y[indices] = rotate_coords(x[indices], y[indices], cx, cy, angle)
    source.data.update(x=x.tolist(), y=y.tolist())

    patch_x = np.array(patch_source.data['x'])
    patch_y = np.array(patch_source.data['y'])
    patch_x, patch_y = rotate_coords(patch_x, patch_y, cx, cy, angle)
    patch_source.data.update(x=patch_x.tolist(), y=patch_y.tolist())

rotate_cw_button.on_click(lambda: rotate_patch(-rot_deg))
rotate_ccw_button.on_click(lambda: rotate_patch(rot_deg))

# Start
#######################################
def start_start():
    start_button.disabled = True

def process_additional_options():
    try:
        # Safely evaluate the input string
        user_dict = ast.literal_eval(options_input.value)
        if isinstance(user_dict, dict):
            print(f"Parsed dictionary: {user_dict}")
        else:
            print("Error: Input is not a dictionary.")
    except Exception as e:
        print(f"Invalid input: {e}")
    return user_dict

def start_main():
    gen_data_path = gen_data_path_input.value.strip()
    algo = algo_input.value.strip()
    hyp_param = hyp_param_input.value.strip()
    buml_obj_info_path = gen_data_path + '/' + algo + '/' + hyp_param + '/' + algo + '.dat'
    save_dir = gen_data_path + '/' + algo + '/' + hyp_param + '_' + REG_TEAR_TAG
    options = process_additional_options()
    print('buml_obj_info_path=' + buml_obj_info_path)
    print('save_dir=' + save_dir)
    global cut_and_paste_obj
    emb_info, metadata = util_.read(buml_obj_info_path)
    buml_obj = buml_.create_buml_obj(emb_info['buml_obj_state'])
    cut_and_paste_obj = cut_and_paste_bokeh.CutAndPaste(
        buml_obj=buml_obj,
        save_progress_dir=save_dir,
        max_refinement_iter=options['max_refinement_iter'],
        force_compute=options['force_compute'],
        cmap_interior=options['cmap_interior'],
        cmap_tear=options['cmap_tear'],
    )
    source.data = cut_and_paste_obj.get_current_embedding_data()

def start_end():
    patch_source.data = dict(x=[], y=[])
    start_button.disabled = False
    cut_button.disabled = False
    update_max_fig_height_button.disabled = False
    fig.toolbar.active_inspect = None
    fig.toolbar.active_drag = lasso_tool
    print_main_log('Select region to cut using lasso and then press cut.')

def start():
    curdoc().add_next_tick_callback(show_spinner)
    curdoc().add_next_tick_callback(start_start)
    curdoc().add_next_tick_callback(start_main)
    curdoc().add_next_tick_callback(start_end)
    curdoc().add_next_tick_callback(adjust_figure_size)
    curdoc().add_next_tick_callback(hide_spinner)

def on_start_button_click():
    curdoc().add_next_tick_callback(start)

start_button.on_click(on_start_button_click)

# Cut
#######################################
def cut_start():
    fig.toolbar.active_drag = None
    cut_button.disabled = True
    save_button.disabled = True

def cut_main():
    global cut_and_paste_obj
    patch_source.data = cut_and_paste_obj.cut_clusters_to_move(source.selected.indices)
    source.selected.indices = cut_and_paste_obj.selected_pts_to_move.tolist()
    print('len(source.selected.indices) = ' + str(len(source.selected.indices)))

def cut_end():
    paste_button.disabled = False
    rotate_cw_button.disabled = False
    rotate_ccw_button.disabled = False
    print_main_log('Drag and rotate the polygonal patch, and then press paste.')

def cut():
    curdoc().add_next_tick_callback(show_spinner)
    curdoc().add_next_tick_callback(cut_start)
    curdoc().add_next_tick_callback(cut_main)
    curdoc().add_next_tick_callback(cut_end)
    curdoc().add_next_tick_callback(adjust_figure_size)
    curdoc().add_next_tick_callback(hide_spinner)

def on_cut_button_click():
    curdoc().add_next_tick_callback(cut)

cut_button.on_click(on_cut_button_click)

# Paste
#######################################
def paste_start():
    paste_button.disabled = True
    rotate_cw_button.disabled = True
    rotate_ccw_button.disabled = True

def paste_end():
    cut_button.disabled = False
    save_button.disabled = False
    patch_source.data = dict(x=[], y=[])
    fig.toolbar.active_drag = lasso_tool

def paste_main():
    global cut_and_paste_obj
    patch_x = np.array(patch_source.data['x'])
    patch_y = np.array(patch_source.data['y'])
    final_polyg = np.column_stack([patch_x, patch_y])
    source.data = cut_and_paste_obj.paste_clusters(final_polyg)

def paste():
    curdoc().add_next_tick_callback(show_spinner)
    curdoc().add_next_tick_callback(paste_start)
    curdoc().add_next_tick_callback(paste_main)
    curdoc().add_next_tick_callback(paste_end)
    curdoc().add_next_tick_callback(adjust_figure_size)
    curdoc().add_next_tick_callback(hide_spinner)

def on_paste_button_click():
    curdoc().add_next_tick_callback(paste)

paste_button.on_click(on_paste_button_click)

# Save
#######################################
def save():
    algo = algo_input.value.strip()
    global cut_and_paste_obj
    cut_and_paste_obj.save_buml_obj(algo+'.dat')

def on_save_button_click():
    curdoc().add_next_tick_callback(save)

save_button.on_click(on_save_button_click)

# Update max figure height
#######################################
def update_max_fig_height():
    global MAX_FIG_HEIGHT
    global MAX_FIG_WIDTH
    MAX_FIG_HEIGHT = int(max_fig_height_input.value.strip())
    MAX_FIG_WIDTH = MAX_FIG_HEIGHT

def on_update_max_fig_height_button_click():
    curdoc().add_next_tick_callback(update_max_fig_height)
    curdoc().add_next_tick_callback(adjust_figure_size)

update_max_fig_height_button.on_click(on_update_max_fig_height_button_click)


# Put everythig together in a doc
#########################################################
curdoc().add_root(
    column(
        main_log_div,
        row(gen_data_path_input),
        row(algo_input, hyp_param_input, options_input, max_fig_height_input), 
        row(update_max_fig_height_button, start_button, cut_button, paste_button, save_button),
        row(rotate_cw_button, rotate_ccw_button),
        spinner,
        fig,
        log_div
    )
)