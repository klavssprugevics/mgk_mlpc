import numpy as np
from pathlib import Path
from skimage.io import imread
from joblib import Parallel, delayed
from pointcloud_utils import geo_to_img
from sklearn.metrics.pairwise import euclidean_distances


# Funkcija, kas aprekina z izkliedi (standartnovirzi) noteiktam apgabalam
def calculate_z_std(all_points, slice, radius):
    res = np.zeros((slice.shape[0]))
    for i in range(0, slice.shape[0]):
        dist = euclidean_distances(slice[i, 0:2].reshape(1, -1), 
        all_points[:, 0:2]).reshape(all_points.shape[0])

        std = np.std(all_points[dist <= radius, 2])
        res[i] = std
    return res


def generate_features(points, cir, extent, xy_radius, job_count=8):
    points_geo = geo_to_img(extent, 0.25, 0.25, points)
    sample_points = np.zeros((points.shape[0], 5))
    print('Total points: ', sample_points.shape)
    print('Radius: ', xy_radius)

    # Ievieto NIR, R, G vertibas
    sample_points[:, 0] = cir[points_geo[:, 1], points_geo[:, 0], 0]
    sample_points[:, 1] = cir[points_geo[:, 1], points_geo[:, 0], 1]
    sample_points[:, 2] = cir[points_geo[:, 1], points_geo[:, 0], 2]

    # Ievieto z vertibas
    sample_points[:, 3] = points[:, 2]

    # Aprekina z izkliedi
    print('Calculating std on ', job_count, ' jobs.')
    slices = np.array_split(points, job_count)

    results = Parallel(n_jobs=job_count)(delayed(calculate_z_std)
    (points, sl, xy_radius) for sl in slices)

    sample_points[:, 4] = np.concatenate(results)
    return sample_points


region = 'vecpilseta'
radius = 3
extent = [353230.75, 363916.50, 353742.50, 364192.50]
cir = imread('data/' + region + '/apgabals.tif')

Path('data/' + region + '/training_data/').mkdir(parents=True, exist_ok=True)


building_points = np.loadtxt('data/' + region + '/extracted_points/building.txt')
building_sample = generate_features(building_points, cir, extent, radius)
np.savetxt('data/' + region + '/training_data/building_sample_' + str(radius) + 'm.txt',
building_sample, fmt='%1.2f')

building_points = None
building_sample = None
print('Finished calculating building points...')

vegetation_points = np.loadtxt('data/' + region + '/extracted_points/vegetation.txt')
vegetation_sample = generate_features(vegetation_points, cir, extent, radius)
np.savetxt('data/' + region + '/training_data/vegetation_sample_' + str(radius) + 'm.txt',
vegetation_sample, fmt='%1.2f')

vegetation_points = None
vegetation_sample = None
print('Finished calculating vegetation points...')
