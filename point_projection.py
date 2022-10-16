from typing import Tuple, Optional, Mapping
import copy
from configparser import ConfigParser

import numpy as np
from scipy.spatial.transform import Rotation as R

from numba import njit, jit
from projection_utils import *

# configs path
ROOT_PATH = './calibration'
CAM_CALIB_PATH = ROOT_PATH + '/SN22892462.conf'
EXT_CALIB_PATH = ROOT_PATH + '/indoor.conf'

# Intrinsic and extrinsic sections and parameters in calibration files
CAM_CALIB_SENSOR = 'RIGHT_CAM_FHD'
EXT_LIDAR_CAMERA = 'EXTRINSICS_LIDAR_CAMERA'
EXT_LIDAR = 'EXTRINSICS_LIDAR'
INTRINSIC_PARAMS = [ 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'p1', 'p2' ]
EXTRINSIC_PARAMS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']


def read_calibration_file(calib_filepath, sensor, checklist=None):
    
    confpar = ConfigParser()
    confpar.read(calib_filepath)

    #print("Found following sections: ")
    #print(list(confpar.keys()))

    assert sensor in list(confpar.keys()), "Calibration mode: "+sensor+" not found in file "+calib_filepath
        
    cparams = {}

    for k, v in confpar[sensor].items():
            
        if checklist != None:
            assert k in checklist, str(k)+": unknown parameter specified in config file"
            
        cparams[k] = float(v)

    return cparams

class PointProjection():

    """
    Project 3D points and bounding boxes to camera image

    Coordinate frames schema:
    
    velodyne:   x, y, z   - right, forward face, up
    camera ref: x, y, z   - right, down, front

    1) Required to transform point cloud to camera frame using extrinsic parameters EXTRINSICS_LIDAR_CAMERA
    2) Project 3D points to 2D pixel coords using intrinsic parameters

    Calibration matrix:

                    [[ fx      0.0     cx  ],
        cameramat =  [ 0.0     fy      cy  ],
                     [ 0.0     0.0     1.0 ]]

    calibration ref: 'https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html'

    """

    def __init__(self, configs=None):
        super().__init__()

        # Read calibration file (intrinsic params)
        cparams = read_calibration_file(CAM_CALIB_PATH, CAM_CALIB_SENSOR, INTRINSIC_PARAMS)

        self.intrinsic_params = np.array([cparams['fx'], cparams['fy'], cparams['cx'], cparams['cy'],
                                 cparams['k1'], cparams['k2'], cparams['k3'], cparams['p1'],
                                 cparams['p2']])

        # Camera matrix
        self.cameramat = np.array([[cparams['fx'], 0.0, cparams['cx']],
                                   [0.0, cparams['fy'], cparams['cy']],
                                   [0.0, 0.0, 1.0]])

        # Read calibration file (extrinsic params)
        lcparams = read_calibration_file(EXT_CALIB_PATH, EXT_LIDAR_CAMERA, EXTRINSIC_PARAMS)

        self.lidar_camera_params = np.array([lcparams['x'], lcparams['y'], lcparams['z'],
                                            lcparams['roll'], lcparams['pitch'], lcparams['yaw']])

        lcrot = R.from_euler('xyz', [lcparams['roll'],
                                   lcparams['pitch'],
                                   lcparams['yaw']], degrees=True)

        # Translation matrix (lidar --> camera)
        self.lidar_to_cam_T = np.array([[lcparams['x']],
                                          [lcparams['y']],
                                          [lcparams['z']]])
        # Rotation matrix (lidar --> camera)
        self.lidar_to_cam_R = lcrot.as_matrix()
        self.lidar_to_cam_RT = np.hstack((self.lidar_to_cam_R, self.lidar_to_cam_T))


        mlparams = read_calibration_file(EXT_CALIB_PATH, EXT_LIDAR, EXTRINSIC_PARAMS)
        self.lidar_camera_params = mlparams


        mlrot = R.from_euler('xyz', [mlparams['roll'],
                                   mlparams['pitch'],
                                   mlparams['yaw']], degrees=True)

        # Translation matrix (world --> lidar)
        self.map_to_lidar_T = np.array([[mlparams['x']],
                                        [mlparams['y']],
                                        [mlparams['z']]])
        # Rotation matrix (world --> lidar)
        self.map_to_lidar_R = mlrot.as_matrix()
        self.map_to_lidar_RT = np.hstack((self.map_to_lidar_R, self.map_to_lidar_T))

    

    def project_pointcloud_to_image(self, points):
        """
            Project lidar points to image

            :param points: lidar points with shape (num_points, 3) e.g: [[x1, y1, z1], [x2, y2, z2], ...]
            :return: pixel coordinates on image e.g: [[u1, v1], ...]
        """
        #print("X")
        # See this function to customize frame transformation
        new_points = self.rigid_transform_points(points)
        return cam_to_img_projection(new_points, self.cameramat, self.intrinsic_params)



    def project_3dbbox_to_image(self, boxes):
        """
            Project 3D boundings boxes to camera image

            :param boxes: bbox (ndarray) shaped as (num_bbox, 6) e.g: [[x1, y1, z1, w1, l1, h1], [...],...]
            :return: pixel coordinates of outermost corners in bbox projection shaped as (num_boxes, 4, 2)

         p1<-@-------------@->p3
             |/|         /|
              ----------- |
             | |        | |
             | ---------- 1
             |/         |/|
         p4<-@-----------_|@->p2
        """

        bbox_cents = boxes[:, 0:3]
        num_bbox = bbox_cents.shape[0]

        # See this function to customize frame transformation
        bbox_cents = self.rigid_transform_points(bbox_cents, corr_fact=True)

        boxes[:, 0:3] = bbox_cents

        vertices = get_bboxes_vertices(boxes)

        # Project vertices
        res_verts = cam_to_img_projection(vertices, self.cameramat, self.intrinsic_params)
        # Project centroids
        res_cents = cam_to_img_projection(bbox_cents, self.cameramat, self.intrinsic_params)

        res_verts = np.reshape(res_verts, (num_bbox, 8, 3)) # 8 vertices, 3 dimensions
        res_cents = np.reshape(res_cents, (num_bbox, 1, 3))

        corners = get_outermost_corners(res_verts)

        return res_verts, res_cents, corners


    def rigid_transform_points(self, points, corr_fact=False):
        if corr_fact:
            # Customize any correction factor if needed
            corr_fact = np.array(np.sqrt(np.power(points[:, 0], 2.0) + \
                                         np.power(points[:, 1], 2.0) + \
                                         np.power(points[:, 2], 2.0)) * np.sin(0.15708))

            points[:, 2] = points[:, 2] + corr_fact

        # PAY ATTENTION HERE:
        # Define here your rigid transform (according to R, T read from calib files)
        # e.g: lidar -> camera (rotation + translation)
        new_points = dot_prod(self.lidar_to_cam_R, points.T)
        new_points = new_points + self.lidar_to_cam_T
        new_points = new_points.T
        # and other rigid transformation here...
        #print(new_points)

        return new_points


@njit
def cam_to_img_projection(img_points, cameramat, int_params=None):

    img_points[:, 0] /= img_points[:, 2]
    img_points[:, 1] /= img_points[:, 2]

    #uncomment this sections if considering lens distorsion #########################
    # r = (img_points[:, 0]**2 + img_points[:, 1]**2)
    # f_rect = (1 + int_params['k1']*(r**2) + int_params['k2']*(r**4) + int_params['k3']*(r**6))
    # x_rect = img_points[:, 0] * f_rect + 2 * int_params['p1'] * img_points[:, 0] * img_points[:, 1] + int_params['p2'] * (r**2 + 2 * img_points[:, 0]**2)
    # y_rect = img_points[:, 1] * f_rect + 2 * int_params['p2'] * img_points[:, 0] * img_points[:, 1] + int_params['p1'] * (r**2 + 2 * img_points[:, 1]**2)
    ######################################################################################à

    # img_pixel_points = dot_prod(cameramat, img_points.T)

    img_pixel_points = np.zeros(img_points.shape)
    # fx * x' + cx
    img_pixel_points[:, 0] = img_points[:, 0] * int_params[0] + int_params[2]
    # fy * y' + cy
    img_pixel_points[:, 1] = img_points[:, 1] * int_params[1] + int_params[3]

    img_pixel_points = np.rint(img_pixel_points)

    return img_pixel_points




