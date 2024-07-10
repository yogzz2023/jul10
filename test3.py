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
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = self.Z1[0]
            self.Sf[1] = self.Z1[1]
            self.Sf[2] = self.Z1[2]
            self.Meas_Time = time
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.Meas_Time = time
            print("Initialized filter state:")
            print("Sf:", self.Sf)
            print("Pf:", self.Pf)
            self.second_rep_flag = True
        
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
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sp:", self.Sp)
        print("Pf:", self.Pf)

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

def clusters(measurements, kalman_filter):
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
    valid_clusters = chi_square_clustering(measurements, kalman_filter)
    hypotheses = generate_hypotheses(valid_clusters, targets)
    
    print("Valid clusters:", valid_clusters)
    print("Targets:", targets)
    print("Generated hypotheses:", hypotheses)
    
    if not hypotheses:
        print("No hypotheses generated.")
        return None
    
    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, kalman_filter) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)
    
    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]
    
    best_hypothesis_index = np.argmax(marginal_probabilities)
    best_hypothesis = hypotheses[best_hypothesis_index]
    
    print("Best hypothesis:", best_hypothesis)
    
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
    r_list = []
    az_list = []
    el_list = []
    for i in range(len(filtered_values_csv)):
        r_val, az_val, el_val = cart2sph(x[i], y[i], z[i])
        r_list.append(r_val)
        az_list.append(az_val)
        el_list.append(el_val)
    
    return r_list, az_list, el_list

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

def form_measurement_groups(measurements, max_time_diff=50):
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

def chi_square_clustering(measurements, kalman_filter):
    clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if kalman_filter.gating(Z):
            clusters.append(measurement)
    return clusters

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_52_test.csv'

# Read measurements from CSV
measurements = read_measurements_from_csv(csv_file_path)

# Group measurements based on time proximity
measurement_groups = form_measurement_groups(measurements, max_time_diff=50)

# Define lists to store plotted data
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through measurement groups
for group in measurement_groups:
    for i, (x, y, z, mt) in enumerate(group):
        if i == 0:
            # Initialize filter state with the first measurement in the group
            kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
        else:
            # Perform prediction and update steps
            kalman_filter.predict_step(mt)
            kalman_filter.update_step(np.array([[x], [y], [z]]))
    
    # Perform JPDAF (Joint Probabilistic Data Association Filter)
    clusters = chi_square_clustering(group, kalman_filter)
    targets = [kalman_filter.Sf]  # Use current filter state as the target
    best_hypothesis = jpda(clusters, targets, kalman_filter)

    # Store data for plotting
    time_list.append(mt)
    r_list.append(best_hypothesis[0])
    az_list.append(best_hypothesis[1])
    el_list.append(best_hypothesis[2])

# Convert azimuth and elevation back to spherical coordinates
r_values, az_values, el_values = cart2sph2(np.array(r_list), np.array(az_list), np.array(el_list), filtered_values_csv)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
ax1.plot(time_list, r_values, label='Range')
ax1.set_xlabel('Time')
ax1.set_ylabel('Range')
ax1.legend()

ax2.plot(time_list, az_values, label='Azimuth')
ax2.plot(time_list, el_values, label='Elevation')
ax2.set_xlabel('Time')
ax2.set_ylabel('Angle (degrees)')
ax2.legend()

plt.tight_layout()
plt.show()

