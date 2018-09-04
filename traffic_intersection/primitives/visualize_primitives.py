# Visualize Primitives
# Tung M. Phan
# August 2nd, 2018
# California Institute of Technology
import os, sys
sys.path.append('..')
import matplotlib.pyplot as plt
from PIL import Image
import primitives.load_primitives as load_primitives
from primitives.load_primitives import get_prim_data

dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"

# prim_id to visualize
prim_id = 132
fig = plt.figure()
plt.axis("on") # turn on/off axes
background = Image.open(intersection_fig)
plt.imshow(background, origin="Lower")
x0 = get_prim_data(prim_id, 'x0')[2:4]
xf = get_prim_data(prim_id, 'x_f')[2:4]
plt.arrow(x=x0[0],y=x0[1],dx=xf[0]-x0[0],dy=xf[1]-x0[1],color='r',head_width=20)
plt.show()

