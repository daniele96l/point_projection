import sys
sys.path.append('../../../../../configs')
from odt_configs import *

from utils import *
from Camera import Camera
from Bev import Bev
from point_projection import PointProjection
import darknet

from numba import njit
import numpy as np

import cv2
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image, CompressedImage
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import message_filters


def draw_2dboxes_to_img(detections, image, colors):
    for det in detections:
        bbox = [det.pose.position.x, det.pose.position.y, det.dimensions.x, det.dimensions.y]
        left, top, right, bottom = darknet.bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
        #cv2.putText(image, "{} [{:.2f}]".format(det.label, float(det.value)),
        #            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            (255, 0, 0), 2)
    return image


def draw_3dbboxes_to_img(image, detections, corners, colors):

    for corn in corners:
        # det[0], det[1] - --> top - left aligned vertices (front, rear)
        # det[2], det[3] - --> top - right aligned vertices (front, rear)
        # det[4], det[5] - --> bottom - left aligned vertices (front, rear)
        # det[6], det[7] - --> bottom - right aligned vertices (front, rear)

        corn = corn.astype(int)
        # front rectangle
        #for pt in det:
        #    cv2.circle(image, (pt[0], pt[1]), 1, (255, 0, 0), 1)
        cv2.rectangle(image, (corn[2][0], corn[2][1]), (corn[3][0], corn[3][1]), (0, 255, 0), 1)
        # rear rectangle
        #cv2.rectangle(image, (det[1][0], det[1][1]), (det[7][0], det[7][1]), (255, 0, 0), 1)
        #cv2.line(image, (det[0][0], det[0][1]), (det[1][0], det[1][1]), (255, 0, 0), 1)
        #cv2.line(image, (det[6][0], det[6][1]), (det[7][0], det[7][1]), (255, 0, 0), 1)
        #cv2.line(image, (det[2][0], det[2][1]), (det[3][0], det[3][1]), (255, 0, 0), 1)
        #cv2.line(image, (det[4][0], det[4][1]), (det[5][0], det[5][1]), (255, 0, 0), 1)
    return image


def draw_bbox_centroids_to_img(detections, image, colors):
    for det in detections:
        # det[0], det[1] - --> top - left aligned vertices (front, rear)
        # det[2], det[3] - --> top - right aligned vertices (front, rear)
        # det[4], det[5] - --> bottom - left aligned vertices (front, rear)
        # det[6], det[7] - --> bottom - right aligned vertices (front, rear)

        det = det.astype(int)
        # front rectangle
        for pt in det:
            cv2.circle(image, (pt[0], pt[1]), 2, (0, 0, 255), 1)
    return image

data_file = "./cfg/coco.data"

# metadata = darknet.load_meta(data_file.encode("ascii"))
# class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
# label_colors = darknet.class_colors(class_names)
# class_dict = {}

tal = PointProjection()

image_pub = rospy.Publisher('proj_bbox', Image, queue_size=1)

width = 640
height = 480

old_width = 1280
old_height = 720

cvbridge = CvBridge()


def callback(cam_img, bbox2d_msg, bbox3d_msg):
    try:
        image = cvbridge.compressed_imgmsg_to_cv2(cam_img, "bgr8")
    except CvBridgeError as e:
        print(e)
    image = cv2.rotate(image, cv2.ROTATE_180)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    rospy.loginfo("Input image resized: " + str(image_resized.shape))

    image = draw_2dboxes_to_img(bbox2d_msg.boxes, image_resized, {})
    # print (detections)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_new = cv2.resize(image, (old_width, old_height),
                           interpolation=cv2.INTER_LINEAR)

    bboxes_3d = np.array([np.hstack(([bbox.pose.position.x,
                                      bbox.pose.position.y,
                                      bbox.pose.position.z],
                                     [bbox.dimensions.x,
                                      bbox.dimensions.y,
                                      bbox.dimensions.z])) for bbox in bbox3d_msg.boxes])

    if bboxes_3d.shape[0] != 0:
        img_bbox_verts, img_bbox_cents, corners = tal.project_3dbb_to_image(bboxes_3d)

        draw_3dbboxes_to_img(image_new, img_bbox_verts, corners, {})
        #draw_bbox_centroids_to_img(image_new, img_bbox_cents, {})

    image_message = cvbridge.cv2_to_imgmsg(image_new, encoding="passthrough")
        # image_message.header = data.header
    image_pub.publish(image_message)

    # uncomment for cv_show
    # cv2.imshow("detection", image)
    # cv2.waitKey(3)


if __name__ == "__main__":
    rospy.init_node('yolo_results', anonymous=True)
    rospy.loginfo("Node initialized.")

    img_sub = message_filters.Subscriber("/zed2/zed_node/left/image_rect_color/compressed", CompressedImage)
    rospy.loginfo("Zed2 Camera subscriber initialized.")

    bbox2d_sub = message_filters.Subscriber("/bbox2d_l", BoundingBoxArray)
    rospy.loginfo("2D bbox subsdcriber initialized.")

    bbox3d_sub = message_filters.Subscriber("/bbox3d_world", BoundingBoxArray)
    rospy.loginfo("3D bbox subsdcriber initialized.")

    ts = message_filters.ApproximateTimeSynchronizer([img_sub, bbox2d_sub, bbox3d_sub], 100, 1)
    ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
