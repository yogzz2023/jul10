import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.Pp = np.eye(6)
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.Z1 = np.zeros((3, 1))
        self.Z2 = np.zeros((3, 1))
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        self.Meas_Time = current_time
        print("Predicted filter state:")
        print("Sp:", self.Sp)
        print("Pp:", self.Pp)

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def form_clusters(measurements, kalman_filter):
    valid_clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            valid_clusters.append(measurement)
    return valid_clusters

def generate_hypotheses(clusters, targets):
    hypotheses = []
    for cluster in clusters:
        for target in targets:
            hypotheses.append((cluster, target))
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, filter_instance):
    cluster, target = hypothesis
    Z = np.array([[cluster[0]], [cluster[1]], [cluster[2]]])
    Inn = Z - np.dot(filter_instance.H, target)
    S = np.dot(filter_instance.H, np.dot(filter_instance.Pf, filter_instance.H.T)) + filter_instance.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood

def jpda(measurements, targets, kalman_filter):
    valid_clusters = form_clusters(measurements, kalman_filter)
    hypotheses = generate_hypotheses(valid_clusters, targets)
    
    if not hypotheses:
        return None
    
    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, kalman_filter) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)
    
    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]
    
    best_hypothesis_index = np.argmax(marginal_probabilities)
    best_hypothesis = hypotheses[best_hypothesis_index]
    
    return best_hypothesis

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az
    
    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       
        
        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]
    
        if az[i] > 360:
            az[i] = az[i] - 360

    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def form_measurement_groups(measurements, max_time_diff=0.50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]
    
    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]
    
    if current_group:
        measurement_groups.append(current_group)
        
    return measurement_groups

def chi_square_clustering(group, kalman_filter):
    clusters = []
    for measurement in group:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            clusters.append(measurement)
    return clusters

# Create an instance of the Kalman filter
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_52_test.csv'

# Read the measurements from the CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time difference
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_52_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values
A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)



# Initialize lists to store results for plotting
r_list = []
az_list = []
el_list = []
time_list = []

# Process the measurement groups
for group in measurement_groups:
    for i, (x, y, z, mt) in enumerate(group):
        if i == 0:
            kalman_filter.Z1 = np.array([[x], [y], [z]])
            kalman_filter.first_rep_flag = True
        elif i == 1:
            kalman_filter.Z2 = np.array([[x], [y], [z]])
            kalman_filter.second_rep_flag = True
            dt = mt - group[i-1][3]
            vx = (x - kalman_filter.Z1[0, 0]) / dt
            vy = (y - kalman_filter.Z1[1, 0]) / dt
            vz = (z - kalman_filter.Z1[2, 0]) / dt
            kalman_filter.initialize_filter_state(kalman_filter.Z2[0, 0], kalman_filter.Z2[1, 0], kalman_filter.Z2[2, 0], vx, vy, vz, mt)
        else:
            kalman_filter.predict_step(mt)
            Z = np.array([[x], [y], [z]])
            if kalman_filter.gating(Z):
                kalman_filter.update_step(Z)
            r_val, az_val, el_val = cart2sph(x, y, z)
            r_list.append(r_val)
            az_list.append(az_val)
            el_list.append(el_val)
            time_list.append(mt)

# Plot range over time
plt.figure(figsize=(10, 6))
plt.scatter(time_list[2:], r_list[2:], color='b', label='Filtered Values')
plt.scatter(filtered_values_csv[:, 0], A[0], label='Filtered range (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Range')
plt.title('Range over Time')
plt.legend()
mplcursors.cursor()
plt.show()

# Plot azimuth over time
plt.figure(figsize=(10, 6))
plt.scatter(time_list[2:], az_list[2:], color='b', label='Filtered Values')
plt.xlabel('Time')
plt.ylabel('Azimuth')
plt.title('Azimuth over Time')
plt.legend()
mplcursors.cursor()
plt.show()

# Plot elevation over time
plt.figure(figsize=(10, 6))
plt.scatter(time_list[2:], el_list[2:], color='b', label='Filtered Values')
plt.xlabel('Time')
plt.ylabel('Elevation')
plt.title('Elevation over Time')
plt.legend()
mplcursors.cursor()
plt.show()
