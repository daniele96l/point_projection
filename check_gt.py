import pandas as pd
from operator import itemgetter
import os
import csv
import numpy as np
import codecs
from io import StringIO
 
german = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'restriction ends ',
            42:'End no passing veh > 3.5 tons',
            43 : 'parking',
            44 : 'pedestrian crossing',
            45 : 'No stopping',
            46 : 'left - or -right',
            47 : 'No parking',
            48 :' No exit',
            49 : 'duplicated'}

gt_map = {
    
            0: '50km limit',
            1: 'Bike lane',
            2: 'Danger pedestrian crossing',
            3: 'Danger roundabout',
            4: 'Danger slippery road',
            5: 'Keep right',
            6: 'No entry',
            7: 'No parking',
            8: 'No stopping',
            9: 'No trucks',
            10: 'Parking',
            11: 'Pedestrian crossing',
            12: 'Roundabout',
            13: 'Stop',
            14: 'Traffic light',
            15: 'Welcome',
            16: 'Yield',
            17: 'dead end',
            18: 'end bike lane',
            19: 'mee',
            20: 'no entry',
            21: 'no entry 2',
            22: 'right-left',
            23: 'stop',
            24: 'yield'}
sensor_h_mm = 8.6
image_h_px = 2448
real_h_mm = 900  # da prendere da un file che itera e trova il sengale corretto
object_h_px = 0  # in imput dal file generato in output da YOLO
f_m = 4.4

# Folder Path

with open("rilevamenti_test.txt", "r") as myfile:
    data = myfile.read().splitlines()

'''path = "/Users/danieleligato/Desktop/Thesis/point_projection/gt"

os.chdir(path)

with open('gt.txt', 'w') as file:
    writer = csv.writer(file)
    tmp = str("X") + "", str("Y") + "", str("Z") + "", str("Name") + "", str("Confidence") + "", str("N frame")
    writer.writerow(tmp)
    # iterate through all file
    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            x = f.read()
            x += " " +file_path[:4]
            print(x)
            x = x.replace(" ", ",")
            writer.writerow([x])

    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            read_text_file(file)'''


f = codecs.open("/Users/danieleligato/Desktop/Thesis/point_projection/gt1.txt")
contents = f.read()
newcontents = contents.replace('_','')
newcontents = newcontents.replace('"',' ')
newcontents = newcontents.replace(' ','')
StringData = StringIO(newcontents)

df = pd.read_csv(StringData, sep =",")
#gt = pd.read_csv('/Users/danieleligato/Desktop/Thesis/point_projection/gt/gt.csv')  
with open("rilevamenti_test.txt", "r") as myfile:
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

# take the frame number
for i in new_list4:
    frame = i[5].split("/")[4]
    num = frame.split("_")[2]
    # print(num)
    new_list5[y][5] = int(num)
    y += 1
    # print(y)

# order according to the frame number
sorted_list = sorted(new_list5, key=itemgetter(5))
sorted_list2 = pd.DataFrame(sorted_list)
sorted_list2[4] = sorted_list2[4].str[5:-4]
# calculate the distance
