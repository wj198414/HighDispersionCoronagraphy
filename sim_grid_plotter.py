#This program will interpolate and plot the grid results, etc.

import numpy as np
import matplotlib.pyplot as plt

res_col = 0
texp_col = 1
contrast_col = 2
zodi_col = 3
ccf_snr_col = 4
ccf_sd_col = 5

sim_results = np.loadtxt("sim_grid_results.dat.old")

detections = np.where(sim_results[:, ccf_snr_col] - sim_results[:, ccf_sd_col] > 3.0)[0]
sim_detections = sim_results[detections]

#print sim_detections[np.argmax(sim_detections[:,contrast_col])]

desired_res = 25000
desired_zodi = 35.9
desired_contrast = 1.e-11
desired_texp = 400*3600

#truth_arr = np.logical_and(np.abs(sim_results[:, res_col] - desired_res) < 10, np.abs(sim_results[:, zodi_col] - desired_zodi) < 1.e-1)
#truth_arr = np.logical_and(truth_arr, np.abs(sim_results[:, contrast_col] - desired_contrast) < 1.e-11)
truth_arr = np.logical_and(np.abs(sim_results[:, zodi_col] - desired_zodi) < 1.e-1, np.abs(sim_results[:, texp_col] - desired_texp) < 10)

relevant_points = sim_results[np.where(truth_arr)[0]]
#print relevant_points
'''texp_vals = np.logspace(np.log10(4*3600), np.log10(400*3600), num=50)

ccf_snr_vs_texp = np.interp(texp_vals, relevant_points[:,texp_col], relevant_points[:,ccf_snr_col])

plt.figure()
plt.plot(texp_vals, ccf_snr_vs_texp)
plt.show(block=False)'''

xx, yy = np.meshgrid(relevant_points[-6:, contrast_col], np.unique(relevant_points[:, res_col]))

#zz = relevant_points[:, ccf_sd_col]
zz = relevant_points[:, ccf_snr_col] - relevant_points[:, ccf_sd_col]
zz = zz.reshape((10,6))

plt.figure()
im = plt.pcolor(xx, yy, zz, cmap=plt.cm.Blues, vmin=0, vmax=3)
plt.colorbar(im)
plt.xscale("log")
plt.yscale("log")
plt.show(block=False)

