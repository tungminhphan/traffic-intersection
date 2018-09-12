# Visualize Primitives
# Tung M. Phan
# August 2nd, 2018
# California Institute of Technology
import os
import matplotlib.pyplot as plt
from PIL import Image
if __name__ == '__main__':
    from load_primitives import get_prim_data
else:
    # if this file is not called directly, don't plot
    from primitives.load_primitives import get_prim_data

dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"

# prim_id to visualize
prim_id = 93
fig = plt.figure()
plt.axis("on") # turn on/off axes
background = Image.open(intersection_fig)
plt.imshow(background, origin="Lower")
x0 = get_prim_data(prim_id, 'x0')[2:4]
xf = get_prim_data(prim_id, 'x_f')[2:4]
x0 = [int(x0[0]), int(x0[1])]
xf = [int(xf[0]), int(xf[1])]
plt.arrow(x=x0[0],y=x0[1],dx=xf[0]-x0[0],dy=xf[1]-x0[1],color='r',head_width=20)
print(get_prim_data(prim_id, 'controller_found'))
print(x0)
print(xf)
plt.show()

