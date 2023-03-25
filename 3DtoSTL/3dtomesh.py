import numpy as np
import open3d as o3d

input_path = "C:/Users/Laura/Desktop/OI6/3DtoSTL/"
output_path = "C:/Users/Laura/Desktop/OI6/3DtoSTL/outputs/"
dataname = "teapot_306.xyz"
point_cloud = np.loadtxt(input_path+dataname, delimiter = ',')

#transfer the pointcloud data type from numpy to open3d o3d.geometry.PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([pcd])
#radius determination
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3*avg_dist
#computing the mehs
pcd.normals = o3d.utility.Vector3dVector(np.zeros((1,3)))
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(10)
o3d.visualization.draw_geometries([pcd], point_show_normal = True)
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
o3d.visualization.draw_geometries([bpa_mesh])
o3d.io.write_triangle_mesh(output_path+"bpa_mesh.stl", bpa_mesh)
#computing the mesh
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
o3d.visualization.draw_geometries([poisson_mesh])