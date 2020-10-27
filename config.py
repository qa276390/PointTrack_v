"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

project_root = os.path.dirname(os.path.realpath(__file__))
category_embedding = [[0.9479751586914062, 0.4561353325843811, 0.16707628965377808], [0.1,-0.1,0.1], [0.5455077290534973, -0.6193588972091675, -2.629554510116577], [-0.1,0.1,-0.1]]

systemRoot = "/tmp2/vtsai01/"
kittiRoot = os.path.join(systemRoot, "workspace/PointTrack/data/KITTI_MOTS/")
rootDir = os.path.join(systemRoot, 'workspace/PointTrack')
#pythonPath = os.path.join(systemRoot, "PointTrack-torch/bin/python")
pythonPath = "python"
