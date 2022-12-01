import pybullet as p
import roboverse
import numpy as np
import time

physicsClient = p.connect(p.GUI)
p.stepSimulation()

seeing_env = roboverse.make('MURMENV-v0', object_subset='test')

for i in range(100000):
    seeing_env.demo_reset()
    time.sleep(10000)