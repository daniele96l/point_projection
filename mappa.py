import plotly.express as px
import pandas as pd
import utm
import numpy as np
import csv

df = pd.read_csv('dati_mappa_test2.csv').dropna(axis = 0, how = 'any').reset_index(drop=True)


totale = df
print(df)
df2 = pd.DataFrame()
df3 = pd.DataFrame(columns = ['X', 'Y', 'Z', 'Name', 'Confidence', 'N frame'])
df3 = [0]

j = 0
for i in range(len(df)-2):

    x1 = df.iloc[i]["X"]
    y1 = df.iloc[i]["Y"]
    z1 = df.iloc[i]["Z"]
    x2 = df.iloc[i+1]["X"]
    y2 = df.iloc[i+1]["Y"]
    z2 = df.iloc[i+1]["Z"]

    dist = np.sqrt((x1- x2)** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2).mean()

    if (((str(df.iloc[[i]]["Name"][i]) == str(df.iloc[[i + 1]]["Name"][i + 1])) and dist < 1) ): #se la distanza è troppo grande se ci sono due segnali al lato opposto della strada in un incrocio li mergia
        df2 = df2.append(df.iloc[[i]])
        df2 = df2.append(df.iloc[[i+1]])
        df2.at[i,"Confidence"] = j  #trovo i segnali duplicati e faccio in modo
        df2.at[i+1,"Confidence"] = j  #che una colonna sia UGUALE (che non sia il nome) in modo da poter usare la group by
    else:
        df2 = df2.append(df3)
        j+=1
        
duplicati = df2.groupby(["Confidence", "Name"]).mean() #faccio la media dei miei segnali che sono duplicati


duplicati.reset_index(level=1, inplace=True)


#df2.drop([0, 'Confidence'], axis = 1, inplace = True) #preparo i due array per essere mergiati eliminando (exclusive join)
df.drop(['Confidence'], axis = 1, inplace = True)  #eliminando quelli presenti vicendevolemnte
unici = pd.merge(df,df2, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
s_unici = unici
tutti= unici.append(duplicati)

tutti_2 = tutti['N frame'].round().astype(str).str[:-3]
tutti_2 = tutti_2.to_frame()
#tutti_2 = tutti_2.reset_index()
tutti_2 = tutti_2.rename(columns={'N frame': 'proximity'})

#mm = triplicati.reset_index().round()

tutti[0] = tutti_2['proximity']

tutti.drop(['Confidence'], axis = 1, inplace = True) #preparo i due array per essere mergiati eliminando (exclusive join)
tutti = tutti.dropna(axis = 0, how ='any')


tutti["Proximity"] = tutti["X"]+tutti["X"]
tutti["Proximity"] = tutti["Proximity"].astype(int) #MAYBE MAYBE MAYBE
triplicati = tutti.groupby(["Proximity", "Name"]).mean() #faccio la media dei miei segnali che sono duplicati

#triplicati = tutti.groupby([0, "Name"]).mean() #faccio la media dei miei segnali che sono duplicati

#---------

df_converted = triplicati
#df_converted = totale
#----------
#TODO, se lo stesso segnale (stesso nome) è in posizione "vicina" ma in fotogrammi diversi va mergiato
df_unstacked = df_converted.index.get_level_values('Name')

df_converted["Name"] = df_unstacked
matrix = np.asarray(df_converted)

with open('mappa_final.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow("x "+"y "+"z " + "F " +"T")
    y = 0
    for i in matrix:
        print(i)
        writer.writerow(i)

df_converted['Name'] = df_converted.Name
c = df_converted.index.get_level_values('Name')

x_offset = 360000
y_offset = 5100000

df_offset = pd.DataFrame()
df_offset["X"] = df_converted["X"] - x_offset
df_offset["Y"] = df_converted["Y"] - y_offset
df_offset["z"] = df_converted["Z"]
df_offset["N frame"] = df_converted["N frame"]

df_offset["Proximity"] = df_offset["X"]+df_offset["X"]+df_offset["z"]
df_offset["Proximity"] = df_offset["Proximity"].astype(int) #MAYBE MAYBE MAYBE

df_offset2 = pd.DataFrame()
df_offset3 = pd.DataFrame()

for i in range(len(df_offset)-1):
    before = df_offset.iloc[i]
    after = df_offset.iloc[i+1]
    before = pd.DataFrame(before)
    after = pd.DataFrame(after)
    before = before.droplevel(level=0, axis = 1)
    after = after.droplevel(level=0, axis = 1)
    distance = abs(before.loc["Proximity"]-after.loc["Proximity"])
    if(before.columns == after.columns and int(distance) < 3):
        print("X")
    else:
        df_offset2 = df_offset2.append(df_offset.iloc[i])
        

for i in range(len(df_offset2)-1):
    before = df_offset2.iloc[i]
    after = df_offset2.iloc[i+1]
    before = pd.DataFrame(before)
    after = pd.DataFrame(after)
    before = before.droplevel(level=0, axis = 1)
    after = after.droplevel(level=0, axis = 1)
    distance = abs(before.loc["Proximity"]-after.loc["Proximity"])
    if(before.columns == after.columns and int(distance) < 3):
        print("x")
    else:
        df_offset3 = df_offset3.append(df_offset2.iloc[i])

df_converted["X"],df_converted["Y"] = utm.to_latlon(df_converted["X"], df_converted["Y"], 33, 'T')


z = df_converted["Z"].fillna(0)
z = np.where(z>200 , 200, z)
#print(df['centroid_lon'])
fig = px.scatter_mapbox(
                        lon=df_converted['Y'],
                        lat=df_converted['X'],
                        color = c,
                        zoom=15,
                        size= z,
                        width=1200,
                        height=900,
                        title='Map')

'''fig = px.scatter_mapbox(
                        lon=df_converted['Y'],
                        lat=df_converted['X'],
                        zoom=15,
                        width=1200,
                        height=900,
                        title='Map')'''

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":10})
fig.show()


    
print("done")


