import blenderproc as bproc
from PIL import Image
import numpy as np
import imageio

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

scene_path = 'assets/free Dimensiva+3Dshaker_FREEpack_01/214 Silla Chair by Thonet/Silla Chair.blend'
camera_path = 'assets/camera_positions'
output_path = 'output'

bproc.init()
objs = bproc.loader.load_blend(scene_path)
bproc.camera.set_resolution(512, 512)

with open(camera_path, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)

# activate normal and depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_normals_output()
bproc.renderer.set_noise_threshold(0.01)  # this is the default value

# render the whole pipeline
data = bproc.renderer.render()

# save images
for key, lis in data.items():
    counter = 0
    for image in lis:
        img_name = str(key) + str(counter) + ".png"
        imageio.imsave(img_name, image)
        counter += 1
