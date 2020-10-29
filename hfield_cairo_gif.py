# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy.integrate import ode

import time

from art_utils.cairo_painter import CairoPainter
from art_utils.gradient import build_gradient

from art_utils.interp import interpgrid

N_PTS = 16
# Linkedin size 1584x396
# WIDTH=1584 # px
# HEIGHT=396 # px
WIDTH=1080 # px
HEIGHT=1080 # px
XRES=8*90 # pts
YRES=8*90 # pts
# XRES = 400
# YRES = 100
XBORDER = 40
YBORDER = 40

GRADIENT_RES = 4096

ANIMATE = False

# colorlist = ["FFC800","C57B57","CBF7ED","5B9279","F24333","BA1B1D"]
#colorlist = ["5B9279","CBF7ED","FFC800","C57B57","F24333","BA1B1D"]
# colorlist = ["2F4B26","85BDA6","85BDA6","0C7797","67697C","484D6D"]
# colorlist = ["2F4B26","85BDA6","85BDA6","0C7797","0C7797","484D6D"]
colorlist = ["2F4B26","85BDA6","85BDA6","0C7797","0C7797","1F2041"]


gradient_pts = [[0,0.33],[0.33,0.66],[0.66,1.0]]

gradient = build_gradient(colorlist, gradient_pts, resolution=GRADIENT_RES)

#print(gradient)
# print(colorlist)
DIPOLE_SEPARATION = 400 # px

# N_STROKES_X = 60*4
# N_STROKES_Y = 60
# N_STROKES_X = 80
# N_STROKES_Y = 45
N_STROKES_X = 120 # pts
N_STROKES_Y = 120 # pts
STROKE_PTS = 20 # pts
MAX_STROKE_DIST = int(3*WIDTH/N_STROKES_X)
MAX_STROKE_STEP_DIST = 4
MAX_STROKE_STEP_DIST_SQ = MAX_STROKE_STEP_DIST**2
TIME_STEP = 2
TIME_STEPS = 20
print('Max step length {} line length {}'.format(MAX_STROKE_STEP_DIST, MAX_STROKE_STEP_DIST*TIME_STEPS))
TIME_LEN = TIME_STEP * TIME_STEPS

starttime = time.time()
np.random.seed(98743)

window_x = np.linspace(0,WIDTH,XRES)
window_y = np.linspace(0,HEIGHT,YRES)

pts = np.random.rand(N_PTS,2)*[WIDTH,HEIGHT]
charges = (np.random.rand(N_PTS)/2+0.5)*np.random.choice([-1,1],N_PTS)
    
X,Y = np.meshgrid(window_x,window_y)

Bx = np.zeros_like(X)# + 0.00001
By = np.zeros_like(Y)# + 0.00001
cmap = np.zeros_like(X)

for pt, q in zip(pts, charges):
    sq_mag = (X-pt[0])**2 + (Y-pt[1])**2
    mag = np.sqrt(sq_mag)

    Bmag = q/mag
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
painter = CairoPainter(width=WIDTH, height=HEIGHT, bg=[0xf4/0xFF,0xf6/0xFF,0xf3/0xFF])
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

frame_num = 0
# Plot integrated strokes, in order of starting magnitude
line_pts = np.zeros((TIME_STEPS,2))
for startpoint_index in startpoint_indices_ordered:
    # print(startpoint_index)
    start_x = stroke_start_pts[startpoint_index,0]
    start_y = stroke_start_pts[startpoint_index,1]
    num_line_pts=0
    xi = start_x * x_sep
    yi = start_y * y_sep

    # Apply dithering
    #xi, yi = xi + (np.random.rand()-0.5)*x_sep, yi + (np.random.rand()-0.5)*y_sep 
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

    # Outline
    #painter.draw_line(line_pts[:num_line_pts], color=[0,0,0,0.2], width=10)

    # Actual line
    rgb = gradient[int(c*(GRADIENT_RES-1))]
    painter.draw_line(line_pts[:num_line_pts], color=rgb, width=8)
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