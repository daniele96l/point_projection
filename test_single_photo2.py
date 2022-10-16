import math
import sys
#sys.path.append('../../../../../configs')
#from odt_configs import *
import pandas as pd
import numpy as np
import math
import cv2
import las_plotter
import las_reader
import matplotlib.pyplot as plt
from point_projection import PointProjection
#import open3d as o3d
from math import atan2, degrees, radians
from scipy.spatial.transform import Rotation as R
from rilevamenti import sorted_list as signal_list
import pandas as pd
pd.options.display.float_format = '{:.5f}'.format
import csv
import open3d as o3d

def angle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle

def shift_coords(fotogramma_n,las_file,t):

    points_las_shift = pd.DataFrame(columns=['X','Y','Z'])

    las_reader.set_n(las_file,t)  #Bad code
    points_las,colors_i = las_reader.return_points(False) #Bad code   , Array of float with 3 columns
    test = []


    points_las_df = pd.DataFrame(points_las, columns = ['X','Y','Z'])
    
    camera = "/Users/danieleligato/Desktop/Thesis/Data/Processed/Ladybug0_1.ori.txt"
    camera_data = pd.read_csv(camera, header=None, delimiter=r"\s+")
    camera_arr = camera_data.to_numpy()
   # curret_frame = pd.DataFrame(columns=['X','Y','Z', 'Heading','Roll', 'Pitch'])
    curret_frame = camera_arr[fotogramma_n][:]#  NF, X, Y, Z, Heading , Roll, pitch
    if(fotogramma_n >= 2):
        previous_frame = camera_arr[fotogramma_n-1][:]

        # Miei radianti = formula(frame_macchina[1] - Frame_macchina[2])
    myradians = math.atan2(float(previous_frame[1]) - float(curret_frame[1]),float(previous_frame[2]) - float(curret_frame[2]))  # boh
    mydegrees = degrees(myradians)  # ci vuole 0 per il frame 50 (170 in termini assoluti)
    #("La mia rotazione rispetto il punto iniziale " + str(mydegrees))

    #Coordinate relative = coordinate assolute nuvola - coordinate assolute macchina
    points_las_shift['X'] = (points_las_df['X'][:] - float(curret_frame[1])) #lato
    points_las_shift['Y'] = (points_las_df['Y'][:] - float(curret_frame[2])) #profondità
    points_las_shift['Z'] = (points_las_df['Z'][:] - float(curret_frame[3])) #altezza

    df = points_las_shift.to_numpy()

    rotation_degrees = float(mydegrees) #MAYBEMAYBE
    rotation_radians = np.radians(rotation_degrees)
    rotation_axis = np.array([0, 0, 1])

    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    #Punti ruotati = rotazione.apply(Coordinate relative)
    rotated_df = rotation.apply(df)

    return rotated_df,points_las,colors_i,curret_frame


def find_las(fotogramma):
    fotogramma_n = fotogramma
    las_file = 0
    if (fotogramma_n < 50):
        las_file = 0
    else:
        las_file = int((fotogramma_n) / 50)

    last_digits = int(str(fotogramma_n)[-2:])
    if (last_digits > 95):  # se siamo vicini alla fine di un point cloud è meglio prendere il prossimo
        las_file += 1
    if (last_digits > 45 and last_digits < 50):
        las_file += 1

    print("Las file: " + str(las_file), "Fotogramma: " + str(fotogramma_n))
    return las_file, fotogramma_n

def find_signal_pos(res,points_las, n,colors_i,curret_frame):
    '''
    n è il fotogramma corrent 
    res sono i punti 2d  nel fotogramma - essa è la trasformazione 2d di points_las , quindi c'è una corrispondenza riga a riga tra i due array
    points_las sono i punti 3d
    
    '''
    global name
    
    position_gps = pd.DataFrame()
    
    risoluzione = [2048,2448]
    #risoluzione = [2448,2048]
    for i in signal_list: #invece di iterare su tutti i segnali io dovrei andare a prendre solo quelli nel mio fotogramma corrente
        Cx = float(i[0])*risoluzione[0]   #centro di ogni i-esimo rilevamento 
        Cy = float(i[1])*risoluzione[1]
        W = float(i[2])*risoluzione[0]
        H = float(i[3])*risoluzione[1]
        
        frame_n = i[5]
        
        index = []

        if(frame_n == n): #itero per ogni frame 
            name = i[4]
            print(name)
            for j in res:
                if(j[0]>Cx-W/2 and j[0]<Cx+W/2 and j[1]>Cy-H/2 and j[1]<Cy+H/2): #vadoa vedere quali punti sono dentro al bounding box
                    index.append(True)
                    print(True)
                else:
                    index.append(False)
                    
            miei_px = res[index]
            point_masked = points_las[index]
            print(point_masked)
            color_naskes = colors_i[index]
            no_outlier = pd.DataFrame(point_masked)
            no_outlier_c = pd.DataFrame(color_naskes)

            if (point_masked.size > 0):
                no_outlier = clustering(point_masked, curret_frame)
                #no_outlier = remove_outlier(no_outlier, 2)
                position_gps = no_outlier.mean(axis=0)
            else:
                position_gps = point_masked.mean(axis=0)

            print(position_gps)

    return position_gps,n,point_masked,no_outlier,no_outlier_c,miei_px


def clustering(point_masked,curret_frame):

   #print(curret_frame[1],curret_frame[2],curret_frame[3])
    print("Con punti > 1m " + str(len(point_masked)))
    df = pd.DataFrame(point_masked)
    df["XCar"] = float(curret_frame[1])
    df["YCar"] = float(curret_frame[2])
    df["ZCar"] = float(curret_frame[3])
    df["Altezza"] = (df.to_numpy()[:, 2]) - (df.to_numpy()[:, 5])
    df = df[ df["Altezza"] < 1]  #i consider points not too high

    print("senza")
    print(len(df.index))
    if(len(df.index) > 0):  #i need to have points inside on the first place
        point_masked = np.asarray(df.drop(columns=["XCar", "YCar", "ZCar", "Altezza"]))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_masked)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=1, min_points=20, print_progress=False))

        max_label = labels.max()
        print(max_label)
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        array = pd.DataFrame(np.asarray(pcd.points))

        print("Clusterizzato")
        print(len(array))
        array["Color"] = labels
        array["XCar"] = float(curret_frame[1])
        array["YCar"] = float(curret_frame[2])
        array["ZCar"] = float(curret_frame[3])
        array_np = np.asarray(array)

        #x_point = array[[0]]
        #x_car  =array[['XCar']]
        #distance_X = x_point.subtract(x_car, axis = 1)

        array["distance"] = np.sqrt(((array_np[:,0]) - (array_np[:,4])) ** 2 + ((array_np[:,1]) - (array_np[:,5])) ** 2 + ((array_np[:,2]) - (array_np[:,6])) ** 2)
        #print(array["distance"].min())

        color_min = array[array["distance"] == array["distance"].min()]
        min = array["distance"].min()
        array = array[array["Color"] == int(color_min["Color"])]

        array = np.asarray(array.drop(columns=['Color',"XCar","YCar","ZCar","distance"]))
        print(len(array))
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(array)
        #o3d.visualization.draw_geometries([pcd_new])

        if (min < 20): #i consider points not too far
            return pd.DataFrame(array)
        else:
            array[:, :] = np.NaN
            return pd.DataFrame(array)
    else:
        array = np.asarray(point_masked)
        array[:, :] = np.NaN
        return pd.DataFrame(array)

def remove_outlier(df_in, col_name):
    df_in = pd.DataFrame(df_in, columns = ['0','1','2'])
    q1 = df_in[col_name].quantile(0.1)
    q3 = df_in[col_name].quantile(0.9)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_in #ops

def iterate_frames():
    to_write = []

    frame_n = fotogramma_scelto
    #651 da problemi
    t = 0

    if (frame_n < 999):
        t = 0
    if (frame_n >= 1000 and frame_n < 1499):
        t = 1
    if (frame_n >= 1500 and frame_n < 2000):
        t = 2

    las_file, fotogramma_n = find_las(frame_n) #minimum 2 , inserisci il numero del tuo fotogramma
    IMG_FILE = "./BoundingBox/202107280658_Rectified_" + str(fotogramma_n) + "_Cam1.jpg"
    tal = PointProjection()
    points_las_shift,points_las,colors_i,curret_frame = shift_coords(fotogramma_n,las_file,t)  #Terzo parametro sono i migliaia

    check = []
    for i in points_las_shift:
        if(i[1]>0):
            check.append(False)
        else:
            check.append(True)

    points_las_shift_correct = points_las_shift[check]
    points_las = points_las[check]
    colors_i = colors_i[check]
        
    #fig = plt.figure()  POTREBBERO SERVIRE QUESTI PLOT
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(points_las_shift_correct[:, 0], points_las_shift_correct[:, 1], points_las_shift_correct[:, 2], marker='o')

    points = np.asarray(points_las_shift_correct)
    res = tal.project_pointcloud_to_image(np.array(points))
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    fotogramma_n = fotogramma_scelto
    
    position_gps,n,point_masked,no_outlier,no_outlier_c,miei_px = find_signal_pos(res,points_las,fotogramma_n,colors_i,curret_frame)

    print("GPS")
    print(position_gps)

    return points, res,IMG_FILE,position_gps,point_masked,no_outlier,no_outlier_c,colors_i,miei_px

def coords_to_px(position_gps):

    tal = PointProjection()
    points = np.asarray(position_gps)
    res = tal.project_pointcloud_to_image(np.array(points))

    return res

if __name__ == "__main__":
   global fotogramma_scelto
   fotogramma_scelto = 528
   #622 mmm
   points,res,IMG_FILE,position_gps ,point_masked,no_outlier,no_outlier_c,colors_i,miei_px=  iterate_frames()

   cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
   frame = cv2.imread(IMG_FILE)
   las_file, fotogramma_n = find_las(fotogramma_scelto)  # minimum 2 , inserisci il numero del tuo fotogramma
   las_plotter.set_n(las_file,0)
   las_plotter.plotta_mappa(position_gps,point_masked,no_outlier,no_outlier_c)  # Bad code   , Array of float with 3 columns
   colors_i[:, [2, 0]] = colors_i[:, [0, 2]]

   for i in range(0, len(res), 1):
        cv2.circle(frame, (int(res[i][0]), int(res[i][1])), 1, (0,0,255), -1)
        #print ("original 3D points: "+str(points[i])+" projected 2D points: ["+str(int(res[i][0]))+" , "+str(int(res[i][1]))+"]" )

   #cv2.imshow("frame", frame)
   cv2.waitKey(0)

