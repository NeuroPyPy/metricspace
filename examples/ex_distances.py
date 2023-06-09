import metricspace as ms
import numpy as np

# Generate random spike trains
spike_train_A = np.sort(np.random.uniform(low=0.0, high=2, size=100))
spike_train_B = np.sort(np.random.uniform(low=0.0, high=2, size=100))

# Input spike trains into a list or array (as many or few as you want)
spike_trains = [spike_train_A, spike_train_B] 

# Make array of cost values to be used in the spike-distance calculation (here we get 0 to 512)
costs = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))

spike_distance = ms.spkd(spike_trains, costs)  # Standard approach
spike_distance_slide = ms.spkd_slide(spike_trains, costs, 10e-3)  # Sliding window approach with search window of 1ms

# Cluster spike trains using spike distance and the number of samples in each class
spike_train_class_labels = np.concatenate((np.zeros(100), np.ones(100))) # 100 samples in each class, randomly generated
_, nsam = np.unique(spike_train_class_labels, return_counts=True)
clustered = ms.distclust(spike_distance, nsam)

# Calculate entropy from the confusion matrix output of distclust
mi = ms.tblxinfo(clustered)
mt = ms.tblxtpbi(clustered)
mit = mi + mt
