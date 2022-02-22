import csv
import time
import joblib
import numpy as np
from pathlib import Path
from validate import calculate_stats
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_data(b_sample, v_sample, train_size=None, test_size=None):

    # Apvieno datus viena un merogo tos
    x_all = np.vstack((b_sample, v_sample))
    y_all = np.zeros((x_all.shape[0]))
    y_all[:b_sample.shape[0]] = 1

    scaler = StandardScaler()
    scaler.fit(x_all)

    if train_size is None or test_size is None:

        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,
        stratify=y_all, random_state=1)
    else:
        x_train_b = b_sample[:train_size, :]
        x_train_v = v_sample[:train_size, :]
        x_train = np.vstack((x_train_b, x_train_v))

        y_train = np.zeros((train_size + train_size))
        y_train[:train_size] = 1

        x_test_b = b_sample[train_size:train_size + test_size, :]
        x_test_v = v_sample[train_size:train_size + test_size, :]
        x_test = np.vstack((x_test_b, x_test_v))

        y_test = np.zeros((test_size + test_size))
        y_test[:test_size] = 1

    x_all = scaler.transform(x_all)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test, x_all, y_all


region = 'vecpilseta'
activation = 'relu'
solver = 'adam'
hidden_layers = (100,200,100,100,100, 100)
train_size = 20000
test_size = 5000
max_iter = 500

# Labels: building = 1, vegetation = 0
b_sample = np.loadtxt('data/' + region + '/training_data/building_sample_2m.txt')
v_sample = np.loadtxt('data/' + region + '/training_data/vegetation_sample_2m.txt')

# Nonem z izkliedi
# b_sample = b_sample[:, :4]
# v_sample = v_sample[:, :4]

x_train, y_train, x_test, y_test, x_region, y_region = split_data(b_sample, v_sample)
# , train_size=train_size,test_size=test_size)

print('Total points: ', x_region.shape[0])
print('Building points: ', b_sample.shape[0])
print('Vegetation points: ', v_sample.shape[0])
print('Total sample points: ', x_train.shape[0] + x_test.shape[0])
print('Train sample: ', x_train.shape[0])
print('Test sample: ', x_test.shape[0])
print('------------------------')

model = MLPClassifier(max_iter=max_iter, random_state=1, solver=solver,
activation=activation, hidden_layer_sizes=hidden_layers)

model.fit(x_train, y_train)

# Saglaba modeli
model_id = time.strftime("%d_%m_%Y-%H_%M")
model_path = 'models/' + region + '/' + model_id + '/'
Path(model_path).mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path + 'model.sav')

acc_test = model.score(x_test, y_test)
acc_region = model.score(x_region, y_region)

print('Accuracy on test sample: ', acc_test)
print('Accuracy on whole region: ', acc_region)

# Predicto apgabalu
pred_region = model.predict(x_region)

# Validacija
TP, FP, FN, TN, accuracy, recall, specificity, precision = calculate_stats(pred_region, b_sample.shape[0])

# Atjauno .csv ar modela rezultatiem
with open('models/' + region + '/results.csv', 'a', encoding='UTF8') as file:
    writer = csv.writer(file)
    writer.writerow([model_id, acc_test, acc_region,
    x_train.shape[0], x_test.shape[0], activation, solver, str(hidden_layers),
    max_iter, TP, FP, FN, TN, accuracy, recall, specificity, precision])

# Ielasa xyz punktus 
b_points = np.loadtxt('data/' + region + '/extracted_points/building.txt')
v_points = np.loadtxt('data/' + region + '/extracted_points/vegetation.txt')

# Izveido klasificetos punktu makonus
region_points = np.vstack((b_points, v_points))
region_points = np.hstack((region_points, np.expand_dims(pred_region, axis=1)))

region_building = region_points[region_points[:, 3] == 1]
region_vegetation = region_points[region_points[:, 3] == 0]

np.savetxt('models/' + region + '/' + model_id + '/predicted_region.txt', region_points)
np.savetxt('models/' + region + '/' + model_id + '/predicted_building.txt', region_building)
np.savetxt('models/' + region + '/' + model_id + '/predicted_vegetation.txt', region_vegetation)
