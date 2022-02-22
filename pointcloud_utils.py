import os
import csv
import subprocess
import numpy as np
from pathlib import Path
from osgeo import gdal, osr
from skimage.filters import median


# pointcloud_path - relative path of .las files to combine. Ex: 'data/points/'
# output - relative path of output. Ex: 'output/points/'
# fusion_path - absolute path to FUSION. Ex: 'C:/FUSION/'
def merge_pointclouds(pointcloud_path, output_path, fusion_path):

    # Iegust abs. path
    cwd = os.getcwd()
    pointcloud_abs_path = cwd + '/' + pointcloud_path
    output_abs_path = cwd + '/' + output_path

    # Izveido .txt ar apvienojamiem failiem
    las_files = os.listdir(pointcloud_path)
    with open(output_path + '/merge_list.txt', 'w') as file_list:
        for file in las_files:
            if Path(file).suffix == '.las':
                file_list.write(pointcloud_abs_path + file + "\n")

    # Ar FUSION apvieno visus .las failus viena punktu makoni
    subprocess.call(cwd + '/fusion_scripts/merge_pointcloud.bat ' + 
    fusion_path + ' ' + output_abs_path + 'merge_list.txt ' + 
    output_abs_path + 'merged.las')


# pointcloud - pointcloud relative path. Ex: 'output/points/merged.las'
# output_path - relative path of output. Ex: 'output/points/'
# fusion_path - absolute path to FUSION. Ex: 'C:/FUSION/'
# extent -  clipping region [x_min, y_min, x_max, y_max]
# Ex: [352500.0, 365000.0, 355000.0, 362500.0]
def clip_pointcloud(pointcloud, output_path, fusion_path, extent):

    # Iegust abs. path
    cwd = os.getcwd()
    pointcloud_abs_path = cwd + '/' + pointcloud
    output_abs_path = cwd + '/' + output_path

    subprocess.call(cwd + '/fusion_scripts/clip_pointcloud.bat '
    + fusion_path + ' ' + pointcloud_abs_path + ' '
    + output_abs_path + 'clipped '+ str(extent[0]) + ' '
    + str(extent[1]) + ' ' + str(extent[2]) + ' ' + str(extent[3]))


# Clip an in-memory np.array with the given extent.
# pointcloud - np.array([x, y, z, ...])
# extent - clipping region [x_min, y_min, x_max, y_max]
# Ex: [352500.0, 365000.0, 355000.0, 362500.0]
def clip_pointcloud_np(pointcloud, extent):

    pointcloud = pointcloud[pointcloud[:, 0] >= extent[0]]
    pointcloud = pointcloud[pointcloud[:, 0] <= extent[2]]
    pointcloud = pointcloud[pointcloud[:, 1] >= extent[1]]
    pointcloud = pointcloud[pointcloud[:, 1] <= extent[3]]

    return pointcloud


# pointcloud - pointcloud relative path. Ex: 'output/points/merged.las'
# output_file - relative path of output. (no file ext) Ex: 'output/points/ground'
# extent -  clipping region [x_min, y_min, x_max, y_max]
# Ex: [352500.0, 365000.0, 355000.0, 362500.0]
# classes - pointcloud classes to keep. Ex: '(2,3,4)'
# fusion_path - absolute path to FUSION. Ex: 'C:/FUSION/'
def extract_pointcloud_class(pointcloud, output_file, fusion_path, extent, classes):

    # Iegust abs. path
    cwd = os.getcwd()
    pointcloud_abs_path = cwd + '/' + pointcloud
    output_abs_path = cwd + '/' + output_file

    if type(classes) == int:
        subprocess.call(cwd + '/fusion_scripts/extract_pointcloud_class.bat '
        + fusion_path + ' ' + pointcloud_abs_path + ' '
        + output_abs_path + ' ' + str(extent[0]) + ' '
        + str(extent[1]) + ' ' + str(extent[2]) + ' ' + str(extent[3])
        + ' ' + str(classes))
    else:
        subprocess.call(cwd + '/fusion_scripts/extract_pointcloud_class.bat '
        + fusion_path + ' ' + pointcloud_abs_path + ' '
        + output_abs_path + ' ' + str(extent[0]) + ' '
        + str(extent[1]) + ' ' + str(extent[2]) + ' ' + str(extent[3])
        + ' "' + classes + '"')


# pointcloud - pointcloud relative path. Ex: 'output/points/pointcloud.las'
# output_path - relative path of output. Ex: 'output/training_data/'
# fusion_path - absolute path to FUSION. Ex: 'C:/FUSION/'
def las_to_xyz(pointcloud, output_path, fusion_path):

    # Iegust abs. path
    cwd = os.getcwd()
    pointcloud_abs_path = cwd + '/' + pointcloud
    output_abs_path = cwd + '/' + output_path

    subprocess.call(cwd + '/fusion_scripts/las_to_xyz.bat '
    + fusion_path + ' ' + pointcloud_abs_path + ' '
    + output_abs_path)


# extent - region [x_min, y_min, x_max, y_max]
# Ex: [353230.75, 363916.50, 353742.50, 364192.50]
# x_res - spatial x resoltion. Ex: 0.25
# y_res - spatial y resoltion. Ex: 0.25
# points - xyz coordinates np.array. Ex: np.array((1,2,3),(4,5,6))
def geo_to_img(extent, x_res, y_res, points):

    points_img = np.zeros((points.shape[0], 2), dtype=int)
    points_img[:, 0] = np.divide(points[:, 0] - extent[0], x_res)
    points_img[:, 1] = np.divide(points[:, 1] - extent[1], y_res)

    return points_img


# pointcloud - pointcloud relative path. Ex: 'output/points/clipped.las'
# output_path - relative path of output. Ex: 'output/dtm/'
# fusion_path - absolute path to FUSION. Ex: 'C:/FUSION/'
def generate_dtm(pointcloud, output_path, fusion_path):

    cwd = os.getcwd()
    pointcloud_abs_path = cwd + '/' + pointcloud
    output_abs_path = cwd + '/' + output_path

    subprocess.call(cwd + '/fusion_scripts/generate_dtm.bat '
    + fusion_path + ' ' + pointcloud_abs_path + ' '
    + output_abs_path)
    

# Merogo dtm, lai pix. vertibas aprakstitu augstumu metros
# dtm_path - relative path to dtm. Ex: 'output/dtm/dtm_refined.dtm'
# scale_path - relative path to dtm scale csv file. Ex: 'output/dtm/dtm_refined_scale_info.csv'
# output_file - relative path of output. Ex: 'output/dtm/dtm_scaled.tif'
def scale_dtm(dtm_path, scale_path, output_file):

    # Atver dtm un nolasa x, y sakuma pos un telpisko izskirtspeju.
    dtm = gdal.Open(dtm_path)
    x_tl, x_res, _, y_tl, _, y_res = dtm.GetGeoTransform()
    dtm = dtm.ReadAsArray()

    # Nogludinasana ar medianas filtru
    scaled_dtm = median(dtm, np.ones((3, 3)))

    # Meroga nolasisana un merogosana
    with open(scale_path, 'r') as file:
        dtm_scale = list(csv.reader(file, delimiter=','))
        old_min = int(dtm_scale[2][0])
        new_min = float(dtm_scale[2][1])
        old_max = int(dtm_scale[3][0])
        new_max = float(dtm_scale[3][1])

        scaled_dtm = ((new_max - new_min) * (scaled_dtm - old_min)) / (old_max - old_min) + new_min

    # Merogota dtm saglabasana ar GDAL
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(output_file, scaled_dtm.shape[1], scaled_dtm.shape[0], 1, gdal.GDT_Float32)
    out.SetGeoTransform((x_tl, x_res, 0, y_tl, 0, y_res))
    out.GetRasterBand(1).WriteArray(scaled_dtm)
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3059)
    out.SetProjection(crs.ExportToWkt())
    out.FlushCache()

