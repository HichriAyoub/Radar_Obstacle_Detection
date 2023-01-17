#!/usr/bin/env python3


import roslib
from yolo_detect import yolovs
from yolo_detection.msg import yolo
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import cv2 as cv
import pandas as pd
import numpy as np
PKG = 'yolo_detection'
roslib.load_manifest(PKG)


class camera_classification:

    def __init__(self):
        with np.load('/home/ayoub/catkin_ws/src/yolo_detection/src/tranformation.npz') as X:
            self.x = [X[i] for i in ('M')]  # Load transformation matrix
        self.cam = cv.VideoCapture(2)
        self.pcl_sub = rospy.Subscriber(
            "/clusters", Image, self.cluster_callback, queue_size=1)

        # Publish a custom message containing confidence threshhold and class
        # name
        self.pub = rospy.Publisher("Confidence_Threshhold", yolo, queue_size=1)
        self.x = np.squeeze(np.asarray(self.x))

    def cluster_callback(self, data):
        a = CvBridge().imgmsg_to_cv2(data)
        df_clusters = pd.DataFrame(
            columns=[
                'x',
                'y',
                'z',
                'intensity',
                'velocity',
                'label'],
            data=a)
        _, frame = self.cam.read()

        self.yolo_bounding(frame)
        for i in df_clusters.label.unique():
            if df_clusters[df_clusters.label == i]["velocity"].mean(
            ) > 0.45 or df_clusters[df_clusters.label == i]["velocity"].mean() < -0.45:
                y = df_clusters[df_clusters.label == i]["y"]
                z = df_clusters[df_clusters.label == i]["z"]
                # Merge y and z coordinates into a matrix
                mat = np.column_stack((y, z))

                # Fill pixels matrix with ones so it is the same shape as
                # transformation matrix (x)
                real_coord = np.vstack([mat.T, np.ones(mat.shape[0])]).T

                # Multiply real coordinate matrix with transformation matrix to
                # obtain pixels coordinate
                final = np.dot(real_coord, self.x)

                # Loop to display transformed points in real time with the
                # camera image
                for j in range(len(final)):
                    image2 = cv.circle(
                        frame, (int(
                            final[j][0]), int(
                            final[j][1])), 5, (0, 0, 255), -1)

        cv.imshow("frame", frame)
        cv.waitKey(1)

    def yolo_bounding(self, img):
        msg = yolo()
        detection = yolovs(
            "/home/ayoub/catkin_ws/src/yolo_detection/src/best.pt")
        # x is a tensorflow tensor containing the confidence and class name of
        # each object
        x = detection.detect(image=img)
        conf = []  # list of confidence threshhold for each detected object
        names = []  # list of class names
        for i in range(len(x)):
            if x != []:
                conf.append(x[i][5].item())
                names.append(x[i][0])
        msg.conf = conf
        msg.classe = names
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('camera_classification', anonymous=False)
    cam = camera_classification()
    rospy.spin()
