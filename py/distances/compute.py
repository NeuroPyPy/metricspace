import os
import scipy.io as sio
import matlab
import matlab.engine

def pairwise_distances(cspks):
    try:
        eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
        print("Connected to existing MATLAB session")
    except IndexError:
        eng = matlab.engine.start_matlab()
        print("Started new MATLAB session")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ))
    eng.addpath(os.path.join(root_path, 'mat'))  # Add the 'mat' folder relative to the current script
    eng.addpath(os.path.join(root_path, 'mat', 'for_spkd_pw'))  # Add the 'for_spkd_pw' subfolder

    mat_file_path = os.path.join(root_path, 'mat', 'for_spkd_pw', 'for_spkd_pw.mat')

    sio.savemat(mat_file_path, {'cspks': cspks})
    Dists = eng.py_to_spkd_pw()
    eng.quit()
    return Dists

