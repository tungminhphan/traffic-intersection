# Visualize Primitives
# Tung M. Phan
# August 2nd, 2018
# California Institute of Technology
import os
import matplotlib.pyplot as plt
from PIL import Image
import primitives.load_primitives as load_primitives
from primitives.load_primitives import get_prim_data

dir_path = os.path.dirname(os.path.realpath(__file__))
intersection_fig = os.path.dirname(dir_path) + "/components/imglib/intersection.png"

fig = plt.figure()
plt.axis("on") # turn on/off axes
background = Image.open(intersection_fig)
plt.imshow(background, origin="Lower")

plt.show()

