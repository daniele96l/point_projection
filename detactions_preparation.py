import pandas as pd
import re
from operator import itemgetter
import numpy as np



sensor_h_mm = 8.6
image_h_px = 2448
real_h_mm =  900  #da prendere da un file che itera e trova il sengale corretto
object_h_px = 0 #in imput dal file generato in output da YOLO
f_m = 4.4

with open ("rilevamenti.txt", "r") as myfile:
    data = myfile.read().splitlines()
    
new_data = []
s = ""
i = 0 
for row in data:
    new_data.append(str(s))
    new_data[i]+= str(row) 
    i+= 1
        

new_listA = [s.replace(" ", "") for s in new_data]
new_listB = [s.replace("[", "") for s in new_listA]
new_list2 = [s.replace("]", ",") for s in new_listB]
new_list3 = [s.replace("'", "") for s in new_list2]



new_list4 = []
for i in new_list3:
    if(len(i) < 20):
        i += ","
        i += ","
        i += ","
        i += ","
        i += ","
        new_list4.append(i.split(","))
    else:
        new_list4.append(i.split(","))
    
    
new_list5 = new_list4
y = 0
    #take the frame number 
for i in new_list4:
    frame = i[5].split("/")[6] 
    num = ''.join(filter(lambda i: i.isdigit(), frame))
    print(num)
    new_list5[y][5] = int(num)
    y+=1
    print(y)
    

#order according to the frame number 
sorted_list = sorted(new_list5, key=itemgetter(5))
sorted_list2 = sorted_list
#calculate the distance
z = 0
for i in sorted_list:
    object_h_px = float(i[2])*image_h_px
   # print(object_h_px)
    distance = (f_m*real_h_mm*image_h_px)/(object_h_px*sensor_h_mm)/real_h_mm
    print(distance)
    sorted_list2[z][6] = distance
    z+= 1

camera_path = "/Users/danieleligato/Desktop/Thesis/Data/Processed/Ladybug0_1.ori.txt"
camera_data = pd.read_csv(camera_path, header=None, delimiter=r"\s+")



camera_data["Signal"] = ""
camera_data["Distances"] = ""

#put inside the dataframe the distance and the name of the signal at the given frame
for i in sorted_list2:
    #print(camera_data["Signal"].iloc[int(i[5])])
    camera_data["Signal"].iloc[int(i[5])] +=  str(i[4])
    camera_data["Signal"].iloc[int(i[5])] += " "
    camera_data["Distances"].iloc[int(i[5])] +=  str(i[6])
    camera_data["Distances"].iloc[int(i[5])] +=  " "


#DENTRO CAMERA DATA CI SONO I SINGOLI FOTOGRAMMI CON OGNI SEGNALE RILEVATO NELLE COLONNE A DESTRA 
