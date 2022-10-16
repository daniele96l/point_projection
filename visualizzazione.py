import open3d as o3d
import laspy as lp
import numpy as np
import matplotlib.pyplot as plt

file = "LAS/202107280658_Un_F_0+100_0+200.las"
campionamento = 10000
point_cloud_i = lp.read(file)
points_i = np.vstack((point_cloud_i.x, point_cloud_i.y, point_cloud_i.z)).transpose()
colors_i = np.vstack((point_cloud_i.red, point_cloud_i.green, point_cloud_i.blue)).transpose()
points_i = points_i[0::campionamento]
colors_i = colors_i[0::campionamento] / 65535.



geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(points_i)
o3d.visualization.draw_geometries([geom])


'''# plotting points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_i[:, 0], points_i[:, 1], points_i[:, 2], c=colors_i, marker='o')
ax.margins(x=0,y=0,z=0)
plt.show()'''
