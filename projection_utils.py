import sys
#sys.path.append('../../../../configs')
#from odt_configs import *

import copy
from configparser import ConfigParser
import numpy as np
from scipy.spatial.transform import Rotation as R

from numba import njit, jit

#@njit
def to_homogeneous(points):
    return np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float64)))

@njit
def dot_prod(mat1, mat2):
    return mat1 @ mat2


@njit
def get_bboxes_vertices(boxes):

    num_verts = 8 * boxes.shape[0]
    vertices = np.zeros((num_verts, 3), dtype=np.float64)

    for i in range(0, num_verts, 8):
        cent = boxes[i//8][0:3]
        dims = boxes[i//8][3:6]

        """
            box.pose.position[x, y, z] 
            box.dimensions[w, d, h]
            v1, v2  ---> top-left aligned vertices
            v3, v4  ---> top-right aligned vertices
            v5, v6  ---> bottom-left aligned vertices
            v7, v8  ---> bottom-right aligned vertices
        """
        vertices[i] = np.array([cent[0]-(dims[0]/2.), cent[1]-(dims[2]/2.), cent[2]+(dims[1]/2.)])
        vertices[i+1] = np.array([cent[0]-(dims[0]/2.), cent[1]+(dims[2]/2.), cent[2]+(dims[1]/2.)])
        vertices[i+2] = np.array([cent[0]+(dims[0]/2.), cent[1]-(dims[2]/2.), cent[2]+(dims[1]/2.)])
        vertices[i+3] = np.array([cent[0]+(dims[0]/2.), cent[1]+(dims[2]/2.), cent[2]+(dims[1]/2.)])
        vertices[i+4] = np.array([cent[0]-(dims[0]/2.), cent[1]-(dims[2]/2.), cent[2]-(dims[1]/2.)])
        vertices[i+5] = np.array([cent[0]-(dims[0]/2.), cent[1]+(dims[2]/2.), cent[2]-(dims[1]/2.)])
        vertices[i+6] = np.array([cent[0]+(dims[0]/2.), cent[1]-(dims[2]/2.), cent[2]-(dims[1]/2.)])
        vertices[i+7] = np.array([cent[0]+(dims[0]/2.), cent[1]+(dims[2]/2.), cent[2]-(dims[1]/2.)])

    return vertices



@njit
def get_outermost_corners(imgbbox):

    num_verts = imgbbox.shape[1]
    num_bbox = imgbbox.shape[0]

    corners = np.zeros((num_bbox, 4, 2))

    for i in range(num_bbox):

        corners[i][0] = np.array([np.min(imgbbox[i][:, 0]), np.min(imgbbox[i][:, 1])])
        corners[i][1] = np.array([np.max(imgbbox[i][:, 0]), np.max(imgbbox[i][:, 1])])
        corners[i][2] = np.array([corners[i][0][0], corners[i][1][1]])
        corners[i][3] = np.array([corners[i][1][0], corners[i][0][1]])

    return corners



# This function compensates the rotation
# of 3d bboxes on x that makes them
# sideways when projected on image
def rotate_vertices(vertices, euler_angles, centroids):

    r = R.from_euler('xyz', euler_angles, degrees=True)

    res = np.empty((0, 3))

    for i in range(0, vertices.shape[0]):

        tmp_verts = np.zeros(3)
        tmp_verts = vertices[i] - centroids[i // 8]
        tmp_res = dot_prod(r.as_dcm(), tmp_verts.T)
        tmp_res = tmp_res.T + centroids[i // 8]

        #tmp_res = tmp_verts + centroids[i // 8]

        res = np.append(res, [tmp_res], axis=0)

    return res


def translate_vertices(vertices, centroids):

    res = np.empty((0, 3))

    for i in range(0, vertices.shape[0]):
        tmp_verts = np.zeros(3)

        tmp_verts[0] = vertices[i][0] + centroids[i // 8][0]
        tmp_verts[1] = vertices[i][1] + centroids[i // 8][1]
        tmp_verts[2] = vertices[i][2] + centroids[i // 8][2]

        res = np.append(res, [tmp_verts], axis=0)

    return res

#@njit
def transform_points(points, translation, rotation, corr_fact=False):

    if corr_fact:
        # Customize any correction factor if needed
        corr_fact = np.array(np.sqrt(np.power(points[:, 0], 2.0) + \
                             np.power(points[:, 1], 2.0) + \
                             np.power(points[:, 2], 2.0)) * np.sin(0.15708))

        points[:, 2] = points[:, 2] + corr_fact

    tf_pts = points.T + translation
    tf_pts = dot_prod(rotation, tf_pts)

    return tf_pts.T
