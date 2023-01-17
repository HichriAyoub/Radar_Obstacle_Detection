import pandas as pd
import numpy as np


df2 = pd.read_csv(
    '/home/ayoub/pcl_to_csv/dbscan/dbscan/image_coord.csv',
    sep=',')  # CSV file containig radar coordinate and camera coordinate
df2.columns = ['x', 'y', 'z', 'image_name', 'u', 'v', '1']
source_points = df2[["y", "z"]]  # Radar coordinate
camera = df2.iloc[:, 4:6]  # Camera coordinate
# add center point of radar which corresponds to
center_radar = pd.DataFrame({'y': [0], 'z': [0]})
# the center point of the camera in pixels (approximately)
center_camera = pd.DataFrame({'u': [320], 'v': [240]})
source_points = pd.concat([source_points, center_radar])
camera = pd.concat([camera, center_camera])
p_radar = np.float64(source_points)  # Convert both pandas dataframes
p_camera = np.float64(camera)       # to numpy array
A = np.vstack([p_radar.T, np.ones(len(p_radar))]).T
x, res, rank, s = np.linalg.lstsq(A, p_camera, rcond=None)

np.savez("tranformation.npz", M=x)
