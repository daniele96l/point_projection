import pandas as pd
import re
from operator import itemgetter
import numpy as np

sensor_h_mm = 8.6
image_h_px = 2448
real_h_mm = 900  # da prendere da un file che itera e trova il sengale corretto
object_h_px = 0  # in imput dal file generato in output da YOLO
f_m = 4.4

with open("../point_projection/kaggle3.txt", "r") as myfile:
    data = myfile.read().splitlines()

new_data = []
s = ""
i = 0
for row in data:
    new_data.append(str(s))
    new_data[i] += str(row)
    i += 1

new_listA = [s.replace(" ", "") for s in new_data]
new_listB = [s.replace("[", "") for s in new_listA]
new_list2 = [s.replace("]", ",") for s in new_listB]
new_list3 = [s.replace("'", "") for s in new_list2]

new_list4 = []
for i in new_list3:
    if (len(i) < 20):
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

for i in new_list4:
    #print(i)
    frame = i[5].split("/")[4]
    num = frame.split("_")[2]
    # print(num)
    new_list5[y][5] = int(num)
    y += 1
    # print(y)

# order according to the frame number
sorted_list = sorted(new_list5, key=itemgetter(5))
sorted_list2 = sorted_list
# calculate the distance
z = 0
'''
detected = []
for i in sorted_list2:
    if(i[5] < 1000 and float(i[3]) > 0.0049462890625):
        detected.append(i)
        
'''