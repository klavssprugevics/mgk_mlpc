from pointcloud_utils import *


fusion = 'C:/FUSION/'
region = 'vecpilseta'

# x_min, y_min, x_max, y_max
extent = [353230.75, 363916.50, 353742.50, 364192.50]

# Apvieno visus lidara punktu makonus
merge_pointclouds(pointcloud_path='data/' + region +'/points/',
output_path='data/' + region + '/extracted_points/',
fusion_path=fusion)

# Apgriez punktu makoni pec noradita extent
clip_pointcloud(pointcloud='data/'+ region + '/extracted_points/merged.las',
output_path='data/' + region + '/extracted_points/',
extent=extent, fusion_path=fusion)

# Zemes punkti
extract_pointcloud_class(pointcloud='data/' + region + '/extracted_points/clipped.las',
output_file='data/' + region + '/extracted_points/ground',
classes=2, extent=extent, fusion_path=fusion)

# Vegetacijas punkti
extract_pointcloud_class(pointcloud='data/' + region + '/extracted_points/clipped.las',
output_file='data/' + region + '/extracted_points/vegetation',
classes='(3,4,5)', extent=extent, fusion_path=fusion)

# Eku punkti
extract_pointcloud_class(pointcloud='data/' + region + '/extracted_points/clipped.las',
output_file='data/' + region + '/extracted_points/building',
classes=6, extent=extent, fusion_path=fusion)
