from __future__ import annotations
import matlab.engine
import numpy as np
from py.util import analysis_functions as af

try:
    eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
except IndexError:
    eng = matlab.engine.start_matlab()

file = r"C:\repos\Metric-Space-Analysis\exclude\matfiles\spk.mat"
data = af.loadmat(file)

cspks = data["cspks"]
td = data["data_today"]["labels"]
yd = data["data_yesterday"]["labels"]
qvals = data["qvals"]
cspk_dict = data["cspskslist"]
#
# Dists = eng.spkd_pw(cspks, list(map(len, cspks)), np.array(qvals, dtype=float))