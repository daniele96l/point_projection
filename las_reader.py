import time
import datetime
import numpy as np
import laspy as lp
import gc
import pandas as pd
#import pptk
#import open3d as o3d
import numpy as np
import laspy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from geopy.distance import geodesic
import geopy
import math
start_time = time.time()
points_arr = []
colors_arr = []

def set_n(n_from_pc,thousants):
    global n
    global t
    n = n_from_pc
    if(n > 49):
        n -= 40
    if(n > 39):
        n -= 30
    if(n > 29):
        n -= 20
    if(n > 19):
        n -= 10
    t = thousants

def return_points(full_map):
  
    df = pd.DataFrame(columns=['NF','X','Y','Z','Heading','Roll','Pitch','Time'])

    print("n=" + str(n))
    print("t=" + str(t))

    if(n < 9):
        file = "/Users/danieleligato/Desktop/Thesis/point_projection/LAS/202107280658_Un_F_"+str(t)+"+"+str(n)+"00_"+str(t)+"+"+str(n+1)+"00.las"
    if(n==9):
        file = "/Users/danieleligato/Desktop/Thesis/point_projection/LAS/202107280658_Un_F_"+str(t)+"+" + str(n) + "00_"+str(t+1)+"+"+ str(0) + "00.las"
    if(n>9):
        file = "/Users/danieleligato/Desktop/Thesis/point_projection/LAS/202107280658_Un_F_"+str(t+1)+"+" + str(n-10) + "00_"+str(t+1)+"+"+str(n+1-10) + "00.las"
    if(n==19):
        file = "/Users/danieleligato/Desktop/Thesis/point_projection/LAS/202107280658_Un_F_"+str(1)+"+" + str(n-10) + "00_"+str(2)+"+"+str(0) + "00.las"
    #print("File LAS usato" + file)

    print(file)
    camera = "/Users/danieleligato/Desktop/Thesis/Data/Processed/Ladybug0_1.ori.txt"
    camera_data = pd.read_csv(camera, header=None, delimiter=r"\s+")
    point_cloud_i = lp.read(file)
    points_i = np.vstack((point_cloud_i.x, point_cloud_i.y, point_cloud_i.z)).transpose()
    colors_i = np.vstack((point_cloud_i.red, point_cloud_i.green, point_cloud_i.blue)).transpose()
    campionamento = 50
    #print("Campionamento usato " + str(campionamento))
    points_i = points_i[0::campionamento]
    colors_i = colors_i[0::campionamento]/65535.
    if(full_map):
        return points_i
    else:
        return points_i, colors_i #COMMENTA QUI PER RUNNARE IL TUTTOOOOOO


'''resticted_points,resticted_points_c = return_points()
d = 0
j = 0

print("3")
# plotting points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(resticted_points[:,0], resticted_points[:,1], resticted_points[:,2], c=resticted_points_c, marker='o')
print("3.5")
plt.show()
print("4")'''
