# HDC
This is simulation code for high dispersion coronagraphy

The input file is:

ROS128b_J.init

This is the file in which you set up parameters in HDC simulations. Many fields are self explanatory.
1, For young self luminous planets, we use template_type "thermal".
2, pl_template_spec_path is the fits file that are used to generate observed spectrum.
3, template_path_thermal is the fits file that stores the template spectrum in cross correlation. You can also use, e.g, HR8799_H2O.fits, or HR8799_CO.fits or HR8799_CH4.fits
4, all fits file should follow the PHENOEX spectrum format. That is, for any new fits file used by the code, it has to use the PHENOEX spectrum spectrum (e.g., lte012-3.5-0.0a+0.0.BT-Settl.fits) as a template and replace the wavelength and flux information within.

There are large files that I cannot upload to Github, i.e., a folder named modern_cube_zodi_dir, the path of which can be set up in the .init file. The folder can be can be downloaded here:

After set up. you can run:
python  hci_hrs_sim.py

The you will get a few plots:
1, simulated spectra, the information is also store in txt files
e.g.,
atm.txt - telluric transmission
rad.txt - sky emission
pl.txt - planet spectrum
st.txt - stellar spectrum
pl_st.txt - observed spectrum without noise
2, noisy spectrum and template spectrum
3, cross correlation function

The code also spits out the following information:

('number of read = ', 156.59999999999999, 23.0, 0.80000001, 60.0, <Quantity 3394.656182351968>)
('number of read = ', 156.59999999999999, 23.0, 0.80000001, 60.0, 1.2614329685458237)
('number of read = ', 156.59999999999999, 23.0, 0.80000001, 60.0, <Quantity 10343452.93187425>)
('number of read = ', 156.59999999999999, 23.0, 0.80000001, 60.0, <Quantity 355.73614404667796>)
Star flux: 5911923712.0
Leaked star flux: 1773577.25
Planet flux: 210840.65625
Planet flux per pixel: 444.811511076
Sky flux: 6685.15797202
Sky transmission: 0.754592767399
hci_hrs_sim.py:707: RuntimeWarning: invalid value encountered in less
  idx = ((wav < self.hci_hrs_obs.wav_max) & (wav > self.hci_hrs_obs.wav_min))
hci_hrs_sim.py:707: RuntimeWarning: invalid value encountered in greater
  idx = ((wav < self.hci_hrs_obs.wav_max) & (wav > self.hci_hrs_obs.wav_min))
{'SNR_vs_NoiseLess': 2.9887322615269971, 'CCF': <__main__.CrossCorrelationFunction instance at 0x7fb615d17998>, 'Center': -21419.065834811099, 'SNR_RMS': 2.618892293464381}

The above texts include total number of photons from the star, planet, sky, leaded star light in the considered wavelength range. The number of pixels can be calculated by Planet flux
/ Planet flux per pixel. You can use these number for back of envelope order of magnitude calculation.

Another output file is multi_sim_log.dat
which contains info like this:
HR8799e_all                                       ,1.50e+04,3.00e-04, 0.280,2.89e+00,6.98e-01,3.36e+00,8.98e-01
1st column: case name
2nd column: spectral resolution
3rd column: the fraction of detections of 100 simulations
4th column: median value of detection significance
5th column: std of detection significance
6th column: median value of detection significance in photon noise limited case
7th column: std of detection significance in photon noise limited case

