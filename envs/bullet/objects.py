import os
import pybullet_data as pdata

from roboverse.bullet.misc import (
  load_urdf,
  load_urdf_randomize_color,
  load_fixed_urdf,
  deg_to_quat,
)

def loader_randomize_color(*filepath, **defaults):
    filepath = os.path.join(*filepath)
    def new(*args, **kwargs):
        defaults.update(kwargs)
        return load_urdf_randomize_color(filepath, **defaults)
    return new

def loader(*filepath, **defaults):
    filepath = os.path.join(*filepath)
    def new(*args, **kwargs):
        defaults.update(kwargs)

        if 'deg' in defaults:
          assert 'quat' not in defaults
          defaults['quat'] = deg_to_quat(defaults['deg'])
          del defaults['deg']
        return load_urdf(filepath, **defaults)
    return new

#Loading with fixed base
def loader_fixed(*filepath, **defaults):
    filepath = os.path.join(*filepath)
    def new(*args, **kwargs):
        defaults.update(kwargs)
        return load_fixed_urdf(filepath, **defaults)
    return new


cur_path = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join('/home/jang/PycharmProjects/bullet env/roboverse/envs/assets/bulletobjects')
ASSET_PATH_ORI = os.path.join('/home/jang/PycharmProjects/bullet env/roboverse/envs/assets')
PDATA_PATH = pdata.getDataPath()
Robot_path = os.path.join('/home/jang/PycharmProjects/bullet env/roboverse/envs/assets/Panda_robot/urdf/panda.urdf')

## robots
panda_robot= loader_fixed(Robot_path)
panda_base = loader_fixed(ASSET_PATH, 'pandabase/pandabase.urdf',
              pos=[-0.04, 0, 1.05],
              quat=[0, 0, 1, 0],
              scale=1)

marble_floor = loader_fixed(ASSET_PATH_ORI, 'floor/plane.urdf',
               pos=[0, 0, 0.01],
               quat=[0, 0, 1, 0],
               scale=1)

table = loader_fixed(ASSET_PATH, 'table/table.urdf',
               pos=[0.2, 0, -0.57],
               quat=[0, 0, 0.707107, 0.707107],
               scale=2.5)

shelf1 = loader_fixed(ASSET_PATH, 'shelf/shelf1/shelf.urdf',
               pos=[0, -0.85, 0.98],
               quat=[0, 1, 1, 0],
               scale=0.35)

shelf2 = loader_fixed(ASSET_PATH, 'shelf/shelf2/shelf.urdf',
               pos=[0, 0.85, 0.98],
               quat=[1, 0, 0, 1],
               scale=0.35)

shelf3 = loader_fixed(ASSET_PATH, 'shelf/shelf3/shelf.urdf',
               pos=[-0.85, 0, 0.98],
               quat=[1, 1, 1, 1],
               scale=0.35)

cube = loader_randomize_color(ASSET_PATH, os.path.join("cube", "cube.urdf"),
              pos=[.75, -.4, 1.0],
              quat=[0, 0, 0, 1],
              scale=0.05) #0.05

