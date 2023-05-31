from __future__ import annotations
import numpy as np
from . import dist_engine

def to_numpy_objarray(list_of_lists):
    """
    Converts a list of lists into a numpy object array, where each element is a numpy array.

    Parameters:
        list_of_lists (list): A list of lists, where each inner list is intended to be a column vector.

    Returns:
        numpy.ndarray: A 1D numpy object array, where each element is a numpy array.
    """
    # Convert each sublist to a 2D numpy array with a single column
    array_of_arrays = [np.array(sublist) for sublist in list_of_lists]

    # Create an object array
    obj_arr = np.zeros((len(array_of_arrays),), dtype=object)

    # Assign each numpy array to the corresponding element in the object array
    for i in range(len(array_of_arrays)):
        obj_arr[i] = array_of_arrays[i]

    obj_arr = obj_arr.reshape(-1, 1)

    return obj_arr

class DistanceMatrix:
    """
    A class for computing distance matrices for spike trains.
    """
    qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))
    def __init__(self, data, ):
        self.data = data
        if not hasattr(data, 'intervals'):
            raise ValueError('Data must have intervals attribute.')
        self.events = np.unique(self.data.event_df['event'])
        self.neurons = {neuron: {} for neuron in self.data.neurons }
        self.format_data()
        self.get_distances()

    def __repr__(self):
        return f'DistanceMatrix({self.data.filename})'

    def format_data(self):
        """
            Prep data for distance matrix calculation.
            Final input should be similar to a MATLAB cell array, where each element is a list of spike times.
            Each element in the list is a trial, and each trial is a list of spike times.
        """
        for neuron in self.data.neurons:
            result = []
            for ind, row in self.data.event_df.iterrows():
                event = row['event']
                start = row['start_time']
                end = row['end_time']
                if end-start >= 2:
                    spks = self.data.neurons[neuron][(self.data.neurons[neuron] >= end) & (self.data.neurons[neuron] <= (end + 2))]
                    adjusted_spks = spks - end
                    thisstim = (event, start, adjusted_spks.tolist())  # need to convert to list for MATLAB
                    result.append(thisstim)

            final = {
                'labels': [x[0] for x in result],
                'cspks': [x[2] for x in result],  # list of lists of spike times
            }
            _, counts = np.unique(final['labels'], return_counts=True)
            final['nsam'] = counts.tolist()
            if not np.sum(counts) == len(final['labels']):
                raise ValueError('Something went wrong with the nsam counts') # this shouldn't happen, but check anyway
            self.neurons[neuron] = final

    def get_distances(self):
        for neuron in self.neurons.keys():
            print(f'Calculating distances for {neuron}')
            cspks = to_numpy_objarray(self.neurons[neuron]['cspks'])
            self.neurons[neuron]['Dists'] = dist_engine(cspks)
            print(f'Finished calculating distances for {neuron}')
