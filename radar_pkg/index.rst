.. _Radar_Package:

=================
radar_pkg
=================

This package provides functions for obstacles detection using radars

Global DEscription
===================

The radar package consist of processing the raw data acquired through the radar (IWR6843AOP). The raw data
is acquired through a ros driver developped by Texas instrument. After pre-processing done internally inside the radar the driver
published a PointCloud2 ros message of all the objects detected.

Raw Data
----------

The raw data collection is based on Texas Instrument Out of box demo and the ros driver corresponding to it (see https://github.com/radar-lab/ti_mmwave_rospkg).
The ros driver consists of nodes that reads data from the radar using usb cable and convert this raw data to Pointcloud2 ros data type that can be visuaized in rviz.
The problem with this data is that it's only random points published and visualized on rviz that doesnt necessarly correspond to a single object.

Passthrough Filters
--------------------

The data published by the ros driver might be noisy sometimes and detection of objects in a further distance may create more noise.
This package uses a passthrough filter in order to limit the distance covered by the radar
The filters are defined using a launch file :doc:`/radar_pkg/launch/filt.launch` and uses a predefined nodelet
Four parameters enables to modify this filter in order to obtain desired results.
These can be divided into two groups:

1) **filter_field_name**: The axis to apply the filtering on.
2) **filter_limit_min**: The minimum distance of detection.
3) **filter_limit_max**: The maximum distance of detection.
4) **filter_limit_negative**: Whether to consider negative values.

Laserscan conversion
---------------------

After processing the raw data, a conversion from Pointcloud2 data to laserscan is needed in order to have a better obstacles detection using local_costmap (maybe for data fusion with lidar).

List of nodes
==============

Clustering Node
----------------

The clustering node is the part of code which takes care of acquiring the filtered data and applying DBSCAN algorithm in order to group pointcloud into separte objects.
Due to random reflection of the radar noises and outliers can be detected, using a clustering algorithm will eliminate outliers.
The DBSCAN algorithm has two parameters

1) eps: The diameter of the circle containing all the points of the cluster.
2) min_pts : The minimum points that are contained inside the circle to consider it as a cluster.

After subscribing to the filter topic **xy_filt_out**, the data is saved into a list and then converted into a pandas dataframe.
A clustering algorithm is applied on the dataframe, this dataframe contained the cluster labels and data is then published as a 
``sensor_msgs/Image.msg``.
Funtion used to cluster and publish data:

.. code-block:: python

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

**List of parameters**:

Subscribers : xy_filt_out, message type ``sensor_msgs/PointCloud2.msg``

Publishers : clusters, message type ``sensor_msgs/Image.msg``

Classification Node
--------------------

The role of the classification is to classify each detected cluster and predict when a person is passing in front the robot of a random object is present in the field of view.
Using the cluster dataframe published by the clustering node, a random-forest model is applied on 12 features extracted from the pointcloud data
The data collection is a bit complicated and requires a bit of time investment

- data collection uses the python code below:

.. code-block:: python

    if df[df.label == index]["prediction"][3] == 'static_person' :
        filecsv=csv.writer(self.outputfile2,delimiter=';')
        filecsv.writerow([num,width,length,S,density,x_std,y_std,range_mean,l,range_std,i_peak,self.velocity_mean])
    elif df[df.label == index]["prediction"][3] == 'objects' :
        filecsv=csv.writer(self.outputfile2,delimiter=';')
        filecsv.writerow([num,width,length,S,density,x_std,y_std,range_mean,l,range_std,i_peak,self.velocity_mean])
    elif df[df.label == index]["prediction"][3] == 'person' :
        filecsv=csv.writer(self.outputfile2,delimiter=';')
        filecsv.writerow([num,width,length,S,density,x_std,y_std,range_mean,l,range_std,i_peak,self.velocity_mean])

personally my method of collecting data is as follows : for example if i want to collect data for objects, i place random objects in front of the robot (radar), make sure that the robot detects only objects. 
Then move the robot to another point and save all the objects detected in a csv file specific for the class "objects". For static_person i use the same method
ps: just make sure to isolate the class that you are trying to collect data for and save the data in the corresponding csv file

To train the random-forest model a jupyter notebook is provided in the radar_pkg.

When the classfication model is trained it is loaded to the classification node and a function called random_forest predicts the class of each cluster.
Lastly for each object and ros published is created in order to publish their corresponding processed point-cloud.

**List of parameters**:

Subscribers : clusters, ``sensor_msgs/Image.msg``

Publishers: 

- person, objects, static_person, all of these messages are of type ``sensor_msgs/PointCloud2.msg``

AI model: a trained model.joblib file ``newest.joblib`` is the latest trained model

.. code-block:: python
    header.stamp = rospy.Time.now()
Publishes the classes pointclouds using wall-time and not sim-time which is why in the case of using a remote pc, a synchronisation between the remote pc and the robot pc is needed

Launch the module
===================

To launch the radar obstacle detection module:

1) Launch the ros driver node in the robot pc
2) Launch the filt.launch file found in launch folder
3) run the clustering node and classification node
4) launch pcl-laser.launch

If using a remote pc steps 2 to 4 should be launched on the remote pc.


Radar Configuration
====================

To be able to obtain raw pointcloud data from the radar, a config file is needed. This file is generated using demo visualizer, an application developped by Texas instruments to generate config files
(see https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/).

Two parameters are really important to take care of are:

- Cfar range threshhold, this parameter allows to decrease of increase the cfar algorithm threshhold. The cfar algorithm allows the detection of false detections called also false-alarm.
- Doppler range threshhold, similar to cfar this parameter allows to remvoe false detection and random reflection point detected due to movement generated by other objects detected.

In order to obtain good results these parameters should be lowered but not to the point of removing actual objects from detection, the task of finding a good combination between these parameters is really hard.

Which is why a config file with a combination suitable for indoor environments was generated and can be found in the ti_mmwave_rospkg available on the robot pc

Random-Forest Jupyter Notebook
===============================

The jupyter notebook provided with the package allows to train a random forest model using a set of csv files.
The csv files are loaded and labeled and then merged into one single numpy array.
The numpy array is then divided into train and test dataset.
Finally the model is fitted and saved using joblib.