import numpy as np
import time
import argparse
from yaml import load, dump
from art_utils.cairo_painter import CairoPainter
from art_utils.gradient import build_gradient, color_from_hex
from art_utils.interp import interpgrid

# Load the configuration file
parser = argparse.ArgumentParser(description='Generate a random image of a magnetic field')
parser.add_argument('config', type=argparse.FileType('r'))
args = parser.parse_args()
data = load(args.config) or dict()

# Sanitize the configuration, insert defaults as needed
colorlist = data.get('colors', [["fe7f2d","fcca46"],["fcca46","a1c181"],["a1c181","619b8a"],["619b8a","233d4d"]])
colorpoints = data.get('color_points', [[0,0.25],[0.25,0.5],[0.5,0.75],[0.75,1.0]])

if len(colorlist) != len(colorpoints):
    print('Colors and color points are not equal in length')
    quit()

GRADIENT_RES = data.get('gradient_res', 4096)

image = data.get('image', {})
WIDTH = image.get('width', 1080)
HEIGHT = image.get('height', 1080)
XBORDER = image.get('x_border', 0)
YBORDER = image.get('y_border', 0)
BG_COLOR = image.get('bg', 'FFFFFF')
print('Image size {}x{}, borders {}x{}, bg {}'.format(WIDTH,HEIGHT,XBORDER,YBORDER,BG_COLOR))

N_PTS = data.get('n_wires', 16)

field_size = data.get('field_size', {})
XRES = field_size.get('x', 720)
YRES = field_size.get('y', 720)
print('Field size {}x{} with {} wires'.format(XRES,YRES, N_PTS))

stroke_obj = data.get('strokes', {})
STROKE_WIDTH = stroke_obj.get('width', 8)
N_STROKES_X = stroke_obj.get('x', 120)
N_STROKES_Y = stroke_obj.get('y', 120)
TIME_STEP = stroke_obj.get('step_size', 0.5)
TIME_STEPS = stroke_obj.get('n_steps', 40)
TIME_LEN = TIME_STEP * TIME_STEPS
print('Stroke quantity {}x{}, width {}, step size {}, steps {}, length {}'.format(N_STROKES_X,N_STROKES_Y,STROKE_WIDTH, TIME_STEP,TIME_STEPS,TIME_LEN))

SEED = data.get('seed', None)
if SEED is not None:
    print('Using seed {}'.format(SEED))
    np.random.seed(SEED)

ANIMATE = data.get('animate', False)
print('Outputting ' + ('multiple frames' if ANIMATE else 'a single image'))

starttime = time.time()

print('Generating data')
# Set up our gradient map
gradient = build_gradient(colorlist, colorpoints, resolution=GRADIENT_RES)

window_x = np.linspace(0,WIDTH,XRES)
window_y = np.linspace(0,HEIGHT,YRES)

pts = np.random.rand(N_PTS,2)*[WIDTH,HEIGHT]
currents = (np.random.rand(N_PTS)/2+0.5)*np.random.choice([-1,1],N_PTS)
    
X,Y = np.meshgrid(window_x,window_y)

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
cmap = np.zeros_like(X)

for pt, current in zip(pts, currents):
    sq_mag = (X-pt[0])**2 + (Y-pt[1])**2
    mag = np.sqrt(sq_mag)

    Bmag = current/mag
    cmap += Bmag
    By += Bmag * np.cos(np.arctan2(Y-pt[1],X-pt[0]))
    Bx += Bmag * -np.sin(np.arctan2(Y-pt[1],X-pt[0]))

# Clamp it down to a multiple of the mean since the max is theoretically inf
cmapmin, cmapmax = cmap.min(), cmap.max()
cmapmean, cmapstd = cmap.mean(), cmap.std()
cmap = np.clip(cmap, cmapmean-cmapstd/2,cmapmean+cmapstd/2)

# Normalize cmap to [0,1]
cmapmin, cmapmax = cmap.min(), cmap.max()
cmap = (cmap-cmapmin)/(cmapmax-cmapmin)

cmap += (np.random.rand(cmap.shape[0],cmap.shape[1])-0.5)*0.001
cmap = np.clip(cmap, 0, 1)

# Set up Cairo env
painter = CairoPainter(width=WIDTH, height=HEIGHT, bg=color_from_hex(BG_COLOR))
painter.insert_borders(XBORDER, YBORDER)

# Normalize the vector field. This simplifies the streamline drawing process
s = np.ma.sqrt(Bx**2+By**2) # Magnitude
By = By / s
Bx = Bx / s

x_sep = WIDTH/N_STROKES_X
y_sep = HEIGHT/N_STROKES_Y

# Generate a list of start points for the strokes
mesh = np.array(np.meshgrid(range(N_STROKES_X), range(N_STROKES_Y)), dtype=np.float64)
stroke_start_pts = mesh.T.reshape(-1, 2)
stroke_start_pts[:,0] += np.random.rand(N_STROKES_X*N_STROKES_Y)
stroke_start_pts[:,1] += np.random.rand(N_STROKES_X*N_STROKES_Y)
stroke_start_pts[:,0] = np.clip(stroke_start_pts[:,0],0,N_STROKES_X)
stroke_start_pts[:,1] = np.clip(stroke_start_pts[:,1],0,N_STROKES_Y)

# Determine the order in which to draw our strokes
# Ideally, we will sort by color map
cmap_startpoints = interpgrid(cmap, stroke_start_pts[:,0]*(XRES/N_STROKES_X), stroke_start_pts[:,1]*(YRES/N_STROKES_Y))
startpoint_indices_ordered = np.argsort(cmap_startpoints)

cmap += (np.random.rand(cmap.shape[0],cmap.shape[1])-0.5)*0.1
cmap = np.clip(cmap, 0, 1)

print('Painting')
# Plot integrated strokes, in order of starting magnitude
line_pts = np.zeros((TIME_STEPS,2))
for startpoint_index in startpoint_indices_ordered:
    start_x = stroke_start_pts[startpoint_index,0]
    start_y = stroke_start_pts[startpoint_index,1]
    num_line_pts=0
    xi = start_x * x_sep
    yi = start_y * y_sep

    t = 0
    num_line_pts = 0
    line_pts[0] = [xi,yi]
    num_line_pts+=1
    while t < TIME_LEN and num_line_pts < TIME_STEPS:
        last_pt = line_pts[num_line_pts-1]
        grid_x = last_pt[0]/(WIDTH/XRES)
        grid_y = last_pt[1]/(HEIGHT/YRES)
        Bx_i = interpgrid(Bx, grid_x, grid_y)
        By_i = interpgrid(By, grid_x, grid_y)
        dp = np.array([Bx_i,By_i])*TIME_STEP
        stroke_d_delta_sq = dp[0]**2+dp[1]**2
        pt = last_pt + dp
        line_pts[num_line_pts]=pt
        num_line_pts+=1
        if pt[0] < 0 or pt[0] > WIDTH or pt[1] < 0 or pt[1] > HEIGHT:
            break
    
    grid_x = xi/(WIDTH/XRES)
    grid_y = yi/(HEIGHT/YRES)
    c = interpgrid(cmap, grid_x, grid_y)

    # Actual line
    rgb = gradient[int(c*(GRADIENT_RES-1))]
    painter.draw_line(line_pts[:num_line_pts], color=rgb, width=STROKE_WIDTH)
    if ANIMATE and startpoint_index % 10 == 0:
        painter.output_frame()
        print(painter.frame)

if ANIMATE:
    # 5 seconds of dead time at end of sequence
    for i in range(60*5):
        painter.output_frame()
else:
    painter.output_snapshot('./single_frame.png')

print('Finished generating in {}s'.format(time.time()-starttime))