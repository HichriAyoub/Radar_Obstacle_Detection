#!/usr/bin/env python3

import rospy
import pandas as pd
from sensor_msgs.msg import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv_bridge import CvBridge
from matplotlib.patches import Rectangle
from std_msgs.msg import String
import joblib
from math import sqrt
from pandas import DataFrame
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs
import csv
import random


class DBSCAN_plot:

    def __init__(self,):
        self.sub = rospy.Subscriber(
            "/clusters",
            Image,
            self.callback,
            queue_size=1)  # Subscribing to clustering topic
        self.loaded_rf = joblib.load(
            "/home/ayoub/hetbot_ws/src/fusion_pkg/src/newest.joblib")  # Load trained model
        self.pub_person = rospy.Publisher(
            "person", pc2.PointCloud2, queue_size=1)
        self.pub_static = rospy.Publisher(
            "objects", pc2.PointCloud2, queue_size=1)
        self.pub_static_person = rospy.Publisher(
            "static_person", pc2.PointCloud2, queue_size=1)
        # self.yolo_sub = rospy.Subscriber("Confidence_Threshhold",yolo,self.confidence_callback)
        # self.confidence = []
        # self.class_name = []
        self.velocity_mean = None  # Variable used for storing the velocity mean of each cluster

        # Files used in data collection for Random-Forest
        """self.outputfile = open('./Static_Person.csv','a')
        self.outputfile3 = open('./person.csv','a')
        self.outputfile2 = open('./Static_objects.csv', 'a')"""

    # Callback function for retrieving confidence threshold and class_name detected by the camera
    # def confidence_callback(self,msg) :

        # self.confidence = msg.conf
        # self.class_name = msg.classe

    def callback(self, data):
        '''Callback function used to retrieve data from the clustering topic.'''
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
        df = self.random_forest(df_clusters)

        self.populate_point_cloud(df)

    # Classifcation Function using random-forest
    def random_forest(self, df_new):
        '''Classifcation Function using random-forest

        Parameters :
        df_new : Pandas dataframe containing 12 features used for random-forest classification

        returns:
        df_final : Pandas dataframe containing the 12 features + the predicted class

        '''
        df_final = DataFrame()

        for index in df_new.label.unique():
            if index != -1:
                df = df_new[df_new["label"] == index]
                df = df.reset_index()
                num = df.shape[0]
                x_max = df['x'].max()
                x_min = df['x'].min()
                y_max = df['y'].max()
                y_min = df['y'].min()
                width = y_max - y_min
                length = x_max - x_min
                corner = (x_min, y_min)
                S = width * length
                density = num / S
                x_std = df.x.std()
                y_std = df.y.std()
                self.velocity_mean = df.velocity.mean()
                df['range'] = df.apply(
                    lambda a: sqrt(
                        a.x *
                        a.x +
                        a.y *
                        a.y +
                        a.z *
                        a.z),
                    axis=1)
                range_mean = df.range.mean()
                l = df.range.max() - df.range.min()
                range_std = df.range.std()
                i_peak = df.intensity.max()
                df_pred = np.array([num,
                                    width,
                                    length,
                                    S,
                                    density,
                                    x_std,
                                    y_std,
                                    range_mean,
                                    l,
                                    range_std,
                                    i_peak,
                                    self.velocity_mean])
                df_pred = df_pred.reshape(1, -1)
                predictions = self.loaded_rf.predict(df_pred)
                df["prediction"] = predictions[0]
                df_final = pd.concat([df_final, df])

                # Store data collected to the appropriate file
                """if df[df.label == index]["prediction"][3] == 'static_person' :
                    filecsv=csv.writer(self.outputfile2,delimiter=';')
                    filecsv.writerow([num,width,length,S,density,x_std,y_std,range_mean,l,range_std,i_peak,self.velocity_mean])
                elif df[df.label == index]["prediction"][3] == 'objects' :
                    filecsv=csv.writer(self.outputfile2,delimiter=';')
                    filecsv.writerow([num,width,length,S,density,x_std,y_std,range_mean,l,range_std,i_peak,self.velocity_mean])
                elif df[df.label == index]["prediction"][3] == 'person' :
                    filecsv=csv.writer(self.outputfile2,delimiter=';')
                    filecsv.writerow([num,width,length,S,density,x_std,y_std,range_mean,l,range_std,i_peak,self.velocity_mean])"""

        return df_final

    def populate_point_cloud(self, df):
        '''Function used to populate the point after classification
        and publishing the new clusters pointcloud.'''
        array = np.empty((0, 3), float)
        array2 = np.empty((0, 3), float)
        array3 = np.empty((0, 3), float)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ti_mmwave_0"
        lx = []
        ly = []
        for index in df.label.unique():
            if df[df.label == index]["prediction"][3] == 'person':
                array = np.append(
                    array, [df[df.label == index][["x", "y", "z"]].mean().to_numpy()], axis=0)
                points = pc2.create_cloud_xyz32(header, array)
                self.pub_person.publish(points)
            elif df[df.label == index]["prediction"][3] == 'objects':
                """array2 = np.append(array2,[[df[df.label==index][["x"]].values.max(),df[df.label==index][["y"]].values.max(),df[df.label==index][["y"]].values.min()]],axis=0)
                print(array2)"""
                for i in range(0, 100):
                    lx.append([random.uniform(df[df.label == index][["x"]].values.max() +
                                              0.1, df[df.label == index][["x"]].values.min() -
                                              0.1), random.uniform(df[df.label == index][["y"]].values.max() +
                                                                   0.1, df[df.label == index][["y"]].values.min() -
                                                                   0.1), 0])
                array2 = np.append(array2, lx, axis=0)
                points2 = pc2.create_cloud_xyz32(header, array2)
                self.pub_static.publish(points2)
            elif df[df.label == index]["prediction"][3] == 'static_person':
                for i in range(0, 100):
                    ly.append([random.uniform(df[df.label == index][["x"]].values.max(), df[df.label == index][["x"]].values.min(
                    )), random.uniform(df[df.label == index][["y"]].values.max(), df[df.label == index][["y"]].values.min()), 0])
                array3 = np.append(array3, ly, axis=0)
                points3 = pc2.create_cloud_xyz32(header, array3)
                self.pub_static_person.publish(points3)


if __name__ == '__main__':
    rospy.init_node('classification_node')
    plot = DBSCAN_plot()
    rospy.spin()
