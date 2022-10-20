import math
import sys
#sys.path.append('../../../../../configs')
#from odt_configs import *
import pandas as pd
import numpy as np
import math
import cv2
import las_reader
import matplotlib.pyplot as plt
from point_projection import PointProjection
#import open3d as o3d
from math import atan2, degrees, radians
from scipy.spatial.transform import Rotation as R
from rilevamenti_kaggle import sorted_list as signal_list #----------------------------
import csv
import open3d as o3d
from scipy.stats import zscore
import scipy.stats as stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
min_distance = 20 #meters
min_size = 10 #pixel
max_altezza_segnale = 1
min_altezza_segnale = -2
min_conf = 0.89

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
    full_map = True
    points_las = las_reader.return_points(full_map) #Bad code   , Array of float with 3 columns

    points_las_df = pd.DataFrame(points_las, columns=['X', 'Y', 'Z'])
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

    return rotated_df,points_las,curret_frame

def find_signal_pos(res,points_las,n,curret_frame):
    '''
    n è il fotogramma corrent
    res sono i punti 2d  nel fotogramma - essa è la trasformazione 2d di points_las , quindi c'è una corrispondenza riga a riga tra i due array
    points_las sono i punti 3d
    '''
    global name
    names = []
    position_gps = pd.DataFrame()
    no_outlier_mean = pd.DataFrame()


    risoluzione = [2048, 2448]
    # risoluzione = [2448,2048]
    for i in signal_list:  # invece di iterare su tutti i segnali io dovrei andare a prendre solo quelli nel mio fotogramma corrente
        Cx = float(i[0]) * risoluzione[0]  # centro di ogni i-esimo rilevamento
        Cy = float(i[1]) * risoluzione[1]
        W = float(i[2]) * risoluzione[0]
        H = float(i[3]) * risoluzione[1]

        frame_n = i[5]
        index = []
        tmp = i[4][-4:]
        N = float(tmp.replace(")", ""))

        #if(frame_n == n and W < min_size): #in pixel
            #print(i)
            #print("Segnale troppo piccolo per essere affidabile")

        if (frame_n == n and W > min_size and H > min_size and N > min_conf):  # itero per ogni frame per ogni cartello che non sia troppo piccolo
            name = str(i[4])
            #print(name)
            names.append(name)
            for j in res:
                if (j[0] > Cx - W / 2 and j[0] < Cx + W / 2 and j[1] > Cy - H / 2 and j[1] < Cy + H / 2):  # vadoa vedere quali punti sono dentro al bounding box, segnali troppo piccolo W < 30 sono probabilmente falsi positivi
                    index.append(True)

                else:
                    index.append(False)

            miei_px = res[index]
            point_masked = points_las[index]

            no_outlier = pd.DataFrame(point_masked)

            if (point_masked.size > 0):
                no_outlier = clustering(point_masked, curret_frame)
                #print(no_outlier)
                no_outlier_mean = no_outlier_mean.append(pd.DataFrame(no_outlier.mean(axis=0)))
                # no_outlier = remove_outlier(no_outlier, 2)
                position_gps = no_outlier_mean

            else:
                #print(i)
                print("Zero punti dentro il bounding box, imposisbile trovare la posizione")
                no_outlier_mean = no_outlier_mean.append(pd.DataFrame(no_outlier.mean(axis=0)))
                # no_outlier = remove_outlier(no_outlier, 2)
                position_gps = no_outlier_mean

    return position_gps,names,n

def find_las(fotogramma):
    fotogramma_n = fotogramma
    las_file = 0
    if(fotogramma_n < 50):
        las_file = 0
    else:
        las_file = int((fotogramma_n)/50)
        
    last_digits = int(str(fotogramma_n)[-2:])
    if(last_digits > 95): #se siamo vicini alla fine di un point cloud è meglio prendere il prossimo
        las_file += 1
    if(last_digits > 45 and last_digits<50):
        las_file += 1

        #684 non va

    print("Las file: " + str(las_file),"Fotogramma: "+str(fotogramma_n))
    return las_file,fotogramma_n

def clustering(point_masked,curret_frame):

    df = pd.DataFrame(point_masked)
    df["XCar"] = float(curret_frame[1])
    df["YCar"] = float(curret_frame[2])
    df["ZCar"] = float(curret_frame[3])
    df["Altezza"] = (df.to_numpy()[:, 2]) - (df.to_numpy()[:, 5])
    df = df[ df["Altezza"] < max_altezza_segnale]  #i consider points not too high
    df = df[df["Altezza"] > min_altezza_segnale]

    #print("senza")
    #print(len(df.index))
    if(len(df.index) > 0):  #i need to have points inside on the first place
        point_masked = np.asarray(df.drop(columns=["XCar", "YCar", "ZCar", "Altezza"]))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_masked)
        #o3d.utility.set_verbosity_level(0)

        #with o3d.utility.VerbosityContextManager() as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=5, print_progress=False))

        max_label = labels.max()
        #print(max_label)
        #print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        array = pd.DataFrame(np.asarray(pcd.points))

        #print("Clusterizzato")
        #print(len(array))
        array["Color"] = labels
        array["XCar"] = float(curret_frame[1])
        array["YCar"] = float(curret_frame[2])
        array["ZCar"] = float(curret_frame[3])
        array_np = np.asarray(array)

        array["distance"] = np.sqrt(((array_np[:,0]) - (array_np[:,4])) ** 2 + ((array_np[:,1]) - (array_np[:,5])) ** 2 + ((array_np[:,2]) - (array_np[:,6])) ** 2)

        color_min = array[array["distance"] == array["distance"].min()]

        array = array[array["Color"] == int(color_min["Color"])]
        min = array["distance"].mean()
        array = np.asarray(array.drop(columns=['Color',"XCar","YCar","ZCar","distance"]))
        #print(len(array))
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(array)
        #o3d.visualization.draw_geometries([pcd_new])

        if (min < min_distance): #i consider points not too far
            return pd.DataFrame(array)
        else:
            #print(curret_frame)
            print("Segnale troppo lontano per essere affidabile")
            array[:, :] = np.NaN
            return pd.DataFrame(array)
    else:
        array = np.asarray(point_masked)
        array[:, :] = np.NaN
        return pd.DataFrame(array)

def remove_outlier(df_in, col_name):
    df_in = pd.DataFrame(df_in)
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def iterate_frames():
    to_write = []
    with open('dati_mappa.csv', 'w') as file:
        writer = csv.writer(file)
        tmp = str("X") +"",str("Y") +"",str("Z") +"",str("Name")+"",str("Confidence") +"",str("N frame")
        writer.writerow(tmp)
        starting = 328
        index = starting
        for i in signal_list[starting:]: #QUESTA LISTA PARTE DA 1
            print("Sengale analizzato")
            print(i)
            print("Rilevamento N " + str(index))
            index += 1
            tmp = i[4][-4:]
            N = float(tmp.replace(")", ""))

            if (N > min_conf):  # in pixel

                frame_n = int(i[5])

                if (frame_n < 999):
                    t = 0
                if (frame_n >= 1000 and frame_n < 1499):
                    t = 1
                if (frame_n >= 1500 and frame_n < 2000):
                    t = 2

                las_file, fotogramma_n = find_las(frame_n) #minimum 2 , inserisci il numero del tuo fotogramma
                IMG_FILE = "./BoundingBox/202107280658_Rectified_" + str(fotogramma_n) + "_Cam1.jpg"
                tal = PointProjection()
                points_las_shift,points_las,curret_frame = shift_coords(fotogramma_n,las_file,t)  #Terzo parametro sono i migliaia

                check = []
                for i in points_las_shift:
                    if (i[1] > 0):
                        check.append(False)
                    else:
                        check.append(True)

                points_las_shift = points_las_shift[check]
                points_las = points_las[check]
                    
#                fig = plt.figure()
#                ax = fig.add_subplot(111, projection='3d')
#                ax.scatter(points_las_shift[:, 0], points_las_shift[:, 1], points_las_shift[:, 2], marker='o')
                
                points = np.asarray(points_las_shift)
                res = tal.project_pointcloud_to_image(np.array(points))
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                
                position_gps,names,n = find_signal_pos(res,points_las,fotogramma_n,curret_frame)

                for i in range(int(len(position_gps)/3)):
                    if((len(position_gps)/3)==1):
                        x = position_gps.loc[0][0]
                        y = position_gps.loc[1][0]
                        z = position_gps.loc[2][0]
                    else:
                        x = position_gps.loc[0].iloc[i].values
                        y = position_gps.loc[1].iloc[i].values
                        z = position_gps.loc[2].iloc[i].values
                        x = str(x)
                        y = str(y)
                        z = str(z)
                        x = str(x).replace("[", "")
                        x = str(x).replace("]", "")
                        y = str(y).replace("[", "")
                        y = str(y).replace("]", "")
                        z = str(z).replace("[", "")
                        z = str(z).replace("]", "")

                    name = names[i]
                    name = str(name).replace("[", "")
                    name = str(name).replace("]", "")
                    name = str(name).replace("'", "")
                    confidence = name[-4:]
                    name = name[:-4]
                    name = name.replace(")","")
                    tmp = str(x) +'',str(y) +"" ,str(z) +"",str(name)+"",str(confidence) +"",str(n)

                    if(not math.isnan(float(x))):
                        print("Salvo il segnale")
                        print(tmp)

                        writer.writerow(tmp)
            else:
                print("Confidence troppo piccola")
            break
            print("--------------------------")


    return points, res,IMG_FILE


if __name__ == "__main__":
    
   points,res,IMG_FILE=  iterate_frames()
   cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
   frame = cv2.imread(IMG_FILE)
    
'''   for i in range(0, len(res), 1):
       if points[i][1] < 0: #fatto con marzia per eliminare i punti nel cielo
           cv2.circle(frame, (int(res[i][0]), int(res[i][1])), 1, (0, 0, 255), -1)
            #print ("original 3D points: "+str(points[i])+" projected 2D points: ["+str(int(res[i][0]))+" , "+str(int(res[i][1]))+"]" )

   cv2.imshow("frame", frame)
   cv2.waitKey(0)
'''