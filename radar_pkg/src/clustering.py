#!/usr/bin/env python3
import rospy
import sys
import pcl
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from pandas import DataFrame
import pandas as pd
import tensorflow.python.keras as keras
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class cluster_pointcloud:
    def __init__(self):
        self.pcl_sub = rospy.Subscriber(
            "xy_filt_out",
            pc2.PointCloud2,
            self.pcl_callback,
            queue_size=1)  # Subscribe to the filter publisher
        # Publish clusters dataframe as an image numpy array
        self.array_pub = rospy.Publisher("/clusters", Image, queue_size=1)

    # Publisher Function

    def publisher(self, df_grouped):

        df = np.float64(df_grouped)
        msg = CvBridge().cv2_to_imgmsg(df)
        self.array_pub.publish(msg)

    # Clustering Function

    def estimate_number_of_clusters(self, downsampled_list):
        # List of pointcloud data (x,y,z)
        downsampled_list_np = np.array(downsampled_list)
        Data = {'x': downsampled_list_np[:,
                                         0],
                'y': downsampled_list_np[:,
                                         1],
                'z': downsampled_list_np[:,
                                         2]}

        df = DataFrame(Data, columns=['x', 'y', 'z'])

        # Apply DBSCAN to regroup cluster with eps radius of the circle
        # and min_samples the minimum points contained in the cluster
        clustering = DBSCAN(eps=0.35, min_samples=5).fit(df)
        cluster = clustering.labels_
        number_of_clusters = len(set(cluster))

        Data_new = {'x': downsampled_list_np[:,
                                             0],
                    'y': downsampled_list_np[:,
                                             1],
                    'z': downsampled_list_np[:,
                                             2],
                    'intensity': downsampled_list_np[:,
                                                     3],
                    'velocity': downsampled_list_np[:,
                                                    4],
                    'label': cluster}
        df_new = DataFrame(Data_new)  # dataframe containing pointcloud data

        return df_new, number_of_clusters

    def pcl_callback(self, data):
        '''Function used to acquire raw data and publish the cluster dataframe. '''
        downsampled_list = []
        for data in pc2.read_points(
                data,
                field_names=(
                    "x",
                    "y",
                    "z",
                    "intensity",
                    "velocity"),
                skip_nans=True):
            downsampled_list.append(
                [data[0], -data[1], -data[2], data[3], data[4]])

        df_grouped, optimal_number_of_clusters = self.estimate_number_of_clusters(
            downsampled_list)

        self.publisher(df_grouped)


def main(args):
    rospy.init_node('clustering_node', anonymous=True)
    cp = cluster_pointcloud()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv)
