import numpy as np
import plotly.express as px
from pointcloud_utils import clip_pointcloud_np

points = np.loadtxt('models/vecpilseta/best_model_ar_izkliedi/predicted_region.txt')
# points_b = np.loadtxt('data/vecpilseta/extracted_points/building.txt')
# points_v = np.loadtxt('data/vecpilseta/extracted_points/vegetation.txt')
# points = np.vstack((points_b, points_v))
# labels = np.zeros((points.shape[0]))
# labels[:points_b.shape[0]] = 1
# points = np.hstack((points, np.expand_dims(labels, axis=1)))
# print(points)

# extent = [353230.75, 363916.50, 353742.50, 364192.50]
# extent = [353230.75, 363716.50, 353542.50, 364100.50]
# points = clip_pointcloud_np(points, extent)

fig = px.scatter_3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], color=points[:, 3],
color_continuous_scale=['green', 'gray'], width=1850, height=900)
fig.update_traces(marker={'size': 2, 'opacity': 1})
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
fig.show()
