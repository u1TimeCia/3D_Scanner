import cv2
import numpy as np
from math import *
import os
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
# import pyvista as pv
import open3d as o3d


output_path = "outputs/"
dataname = "image_coordinates.txt"
# Calculate the total value of green and blue value of the given pixels
# grid_RGB:RGB data set of the origianl picture
# cX:the coordinate of center point of the capturing contour
# cY:the coordinate of center point of the capturing contour
# interval: the scanning matrix width is interval
def GB_total(grid_RGB,cX,cY,interval):
    GB = 0
    for row in grid_RGB[cY:cY+1]:
        for column in row[cX-interval:cX+interval]:
            GB += int(column[1])+int(column[2])
    return GB

# utility function to delete the repeated element in a one dimensional list
def delList(L):
    for i in L:
        if L.count(i) != 1:
            for x in range((L.count(i) - 1)):
                L.remove(i)
    return L

def Get_lightest_pixcel_array(gird_HSV,sort_min_max):
    width = len(grid_HSV)
    lightest_pixcel_array = []
    for key in sort_min_max:
        lightest = 0
        lightest_pixel_x = 0
        lightest_pixcel = []
        if(sort_min_max[key][0] < 0):
            sort_min_max[key][0] = 0
        elif(sort_min_max[key][1] >= 320):
            sort_min_max[key][1] = 319
        for i in range(sort_min_max[key][0],sort_min_max[key][1]+1,1):
            #print("y",key,"x:",i,"light:",gird_HSV[key][i][2])
            if(gird_HSV[key][i][2] > lightest):
                lightest = gird_HSV[key][i][2]
                lightest_pixel_x = i
        lightest_pixcel.append(key)
        lightest_pixcel.append(lightest_pixel_x)
        lightest_pixcel_array.append(lightest_pixcel)
    return lightest_pixcel_array


# triangulation function.
# Takes: a list of coordinates of laser spots; sensor horizontal and vertical resolution;
#     sensor physical width and height;
#     angle alpha, distance D (both set by user);
#     focal length f; radius R (set by user);
#     current rorated angle (deg); current height (mm)
# Returns: a list of the calculated distance and x,y,z coordinates
def triangulation(pList, horizontalP, verticalP, sensorWidth, sensorHeight, alpha, D, f, R, angle, height):
    distList = []
    xList = []
    yList = []
    zList = []
    # get the central pixel's x and y
    centerx = horizontalP / 2
    centery = verticalP / 2

    for i in pList:

        temp = []
        # calculate the horizontal distance of the laser spot to centre of sensor
        deltaD = abs(centerx - i[1]) / horizontalP * sensorWidth
        # calculate the vertical distance of the laser spot to centre of sensor
        h = (i[0] - centery) / verticalP * sensorHeight
        # print("deltaD",deltaD)
        theta = atan(deltaD / f)
        # print("theta",theta)
        H = D / sin(alpha)
        # print("H",H)
        # print("h",h)
        # A is the vertical distance of the laser spot on the object to the current z(height)
        A = h * H / f * (sin(alpha) / sin(alpha + theta))
        z = height + A
        # two cases, depending on if the laser spot is within R or outside of R
        if i[1] < centerx:
            beta = radians(180) - alpha - theta
            deltaR = H * sin(theta) / sin(beta)
            polorCoordinate = R + deltaR
        else:
            beta = alpha - theta
            deltaR = H * sin(theta) / sin(beta)
            polorCoordinate = R - deltaR

        # get the Cartesian coordinates from the angle and the distance
        xList.append(polorCoordinate * cos(angle))
        yList.append(polorCoordinate * sin(angle))
        zList.append(z)
        distList.append(polorCoordinate)

    return distList, xList, yList, zList

def generate_3d_model():

    point_cloud = np.loadtxt(dataname, delimiter=',')

    # transfer the pointcloud data type from numpy to open3d o3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])
    # radius determination
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    # computing the mehs
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(5)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))
    o3d.visualization.draw_geometries([bpa_mesh])
    o3d.io.write_triangle_mesh(output_path + "bpa_mesh.stl", bpa_mesh)
    # computing the mesh
    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    o3d.visualization.draw_geometries([poisson_mesh])

if __name__ == "__main__":

    #   cwd = os.getcwd() + "/Images/"
    #cwd = "Test_2/"  # change this to the image folder if needed
    cwd = "Images/"
    files = sorted(os.listdir(cwd))
    angle = 0
    height = 10
    x_total_list = []
    y_total_list = []
    z_total_list = []

    # --------------------------Spot Detection Part----------------------#
    # looping through iamges
    # files = os.listdir(cwd)
    # files.sort()
    # file_indexes = {file: index for index, file in enumerate(files)}
    #
    # # Print the files and their indexes
    # for file in files:
    #     print(f'{file}: {file_indexes[file]}')
    files = sorted(os.listdir(cwd), key=lambda x: int(x.split('image')[1].split('.jpg')[0]))
    # for file in files:
    #     print(file)
    for file in files:
        if file == ".DS_Store":
            continue
        # This part we get the RGB and HSV data set of the original picture
        print("------" + "Begin to capture " + file + "------" + '\n')
        # print(cwd)
        # print(file)
        img = cv2.imread(cwd + file)
        # colourspace conversion
        grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
        width = len(grid_HSV[0])
        # print(len(grid_HSV[0][0]))
        # HSV: colour (0-179); saturation (0-255); brightness (0-255)

        # set the threshold values and masking
        lower1 = np.array([0, 0, 230])
        upper1 = np.array([30, 255, 255])
        mask1 = cv2.inRange(grid_HSV, lower1, upper1)
        res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

        lower2 = np.array([150, 0, 230])
        upper2 = np.array([180, 160, 255])
        mask2 = cv2.inRange(grid_HSV, lower2, upper2)
        res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

        # combining the two masks together, adn find the red area
        mask3 = mask1 + mask2
        mask3 = cv2.GaussianBlur(mask3, (5, 5), 0)
        ret, binaryMask = cv2.threshold(mask3, 100, 255, cv2.THRESH_BINARY);
        contours, hierarchy = cv2.findContours(binaryMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);
        # cv2.imshow("img1", img)

        # maxX = 0
        # maxY = 0
        num = 0
        interval = 3
        maxGB = 175  # filter for max GB total
        spotList = []
        # traverse the multiple coutours and filter the noise
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            # print("center point", i + 1, ":", cX, cY)
            # print("GB total:", GB_total(grid_RGB, cX, cY, interval))
            # Find the contours of the red laser
            # 175 is the average value of green and blue,
            # first *2 is becuase the scanning matrix's width is interval*2
            # Second *2 is for the average value of green and blue
            # 175 is the threshold
            if GB_total(grid_RGB, cX, cY, interval) / (interval * 2 * 2) < maxGB:
                for j in contours[i]:
                    temp = []
                    temp.append(j[0][0])
                    temp.append(j[0][1])
                    spotList.append(temp)
                # print(spotList)
                # sort the list using the y axis by asceding order

        spotList.sort(key=(lambda x: (x[1], x[0])))
        sort_dict = {}
        for subarray in spotList:
            if subarray[1] in sort_dict:
                sort_dict[subarray[1]].append(subarray[0])
            else:
                sort_dict[subarray[1]] = [subarray[0]]
        sort_min_max = {key: [min(values) - width // 40, max(values) + width // 40] for key, values in
                        sort_dict.items()}
        #print(sort_min_max)
        # print(sort_min_max)

        spotList = Get_lightest_pixcel_array(grid_HSV, sort_min_max)
        # print(spotList)

        #                 lasty = spotList[-1][1]
        #                 #delete the pixel that has the same y coordinate
        #                 for k in range(len(spotList) - 2, -1, -1):
        #                     if lasty == spotList[k][1]:
        #                         del spotList[k]
        #                     else:
        #                         lasty = spotList[k][1]
        # cv2.drawContours(img, [contours[i]], -1, (0,255,0), thickness = -1)
        # print()
        # print("Max:",maxX,maxY)

        #cv2.imshow("mask3", mask3)

        # cv2.imshow("img2", img)

        if (len(spotList) == 0):
            print("No red laser captured!\n")

        # -----------------------Triangulation Part--------------------#
        # triangulation(pList,horizontalP,sensorWidth,alpha,D,f)
        horizontalP = img.shape[1]
        verticalP = img.shape[0]
        sensorWidth = 3
        sensorHeight = 2
        alpha = radians(30)
        D = 100
        f = 5
        R = 10
        dist_list, xList, yList, zList = triangulation(spotList, horizontalP, verticalP, sensorWidth, sensorHeight,
                                                       alpha, D, f, R, angle, height)
        x_total_list += xList
        y_total_list += yList
        z_total_list += zList

        # for i in range(len(dist_list)):
        #     print("x:", xList[i], "y:", yList[i], "z:", zList[i], "distance:", dist_list[i], "x:", spotList[i][1], "y",
        #           spotList[i][0])
        # print(len(deltaR_list))
        green = [0, 255, 0]
        for i in range(len(dist_list)):
            img[spotList[i][0], spotList[i][1]] = green
        #cv2.imshow("img3", img)

        angle += radians(1.8)  # default step size 1.8 for NEMA 17
        # if one rotation finished
        if (angle == radians(360)):
            angle = 0
            height += 40
        print("------" + "End of capturing " + file + "------" + '\n\n')
        cv2.destroyAllWindows()

    xyz_array = np.column_stack((x_total_list, y_total_list, z_total_list))
    #cloud = pv.PolyData(xyz_array)
    # for point in xyz_array:
    #     for xyz in point:
    np.savetxt(dataname, xyz_array, delimiter = ",")
    generate_3d_model()
    # plotter = pv.Plotter()
    # plotter.add_mesh(cloud, point_size=10, render_points_as_spheres=True)
    # plotter.show_grid(show_xaxis=True, show_yaxis=True, show_zaxis=True)
    # plotter.show()

# ax = plt.axes(projection='3d')

#     # Data for a three-dimensional line
#     zline = np.linspace(0, 15, 1000)
#     xline = np.sin(zline)
#     yline = np.cos(zline)
#     ax.plot3D(xline, yline, zline, 'gray')

#     # Data for three-dimensional scattered points
#     ax.scatter3D(x_total_list, y_total_list, z_total_list, c=z_total_list, cmap='Greens');




