""" Visualize the data from the matlab file. """

# from __future__ import annotations
# import numpy as np
# import seaborn as sns
# from pathlib import Path
#
# from util import analysis_functions as af
#
# file = Path(r"C:\repos\metricspace\data\testers.mat")
#
# # Load data
# data = af.loadmat(file.resolve())
#
# dists = data["Dists"]

# corr, stat = af.get_stats(data, excel=False)
# conf = data["anear"]

# bestbl = np.argmax(stat["bl_info"]) # best information for yesterday x today
# besttr = np.argmax(stat["tr_info"]) # best information for today x yesterday
#
# n = conf.shape[0] // 2
# conf_top_right = conf[:n, n:, :]
# conf_bottom_left = conf[n:, :n, :]
#
# tr_y_true, tr_y_pred = af.get_predictions_arr(conf_top_right[:, :, besttr])
# bl_y_true, bl_y_pred = af.get_predictions_arr(conf_bottom_left[:, :, bestbl])
#
# y_true_labels = [data["today"]["labels"][i] for i in tr_y_true]
# y_pred_labels = [data["today"]["labels"][i] for i in tr_y_pred]
#
# metrics = af.metrics.report(y_true_labels, y_pred_labels, labels=data["today"]["labels"])
# metrics = af.metrics.report(bl_y_true, bl_y_pred, labels=data["yesterday"]["labels"])
#
# # Plot
# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.5)
