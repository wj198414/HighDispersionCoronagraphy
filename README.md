# HDC
## This is simulation code for high dispersion coronagraphy (HDC)

The methodology is described in the following two papers:

1, [*Observing Exoplanets with High Dispersion Coronagraphy. I. The Scientific Potential of Current and Next-generation Large Ground and Space Telescopes*](https://ui.adsabs.harvard.edu/abs/2017AJ....153..183W/abstract)

2, [*Effects of thermal and exozodiacal background on space telescope observations of exoEarths*](https://ui.adsabs.harvard.edu/abs/2018SPIE10698E..5GC/abstract) 

**In summary**, the code simulates the obeservation and data reduction process using HDC. The observation part considers photons from the star and planet, zodiac light in the system, sky transmission and emission, and the telescope thermal background. The data reduction part includes procedures of removing all contamination but the planet light at the photon-noise limit. The planet is then detected via the cross correlation technique by cross correlating the noisy observed planet spectrum with a template spectrum that considers molecules of interest (e.g., biosignatures).   

The input file is:

*ROS128b_J.init*

```
wav_min:                1.1                             # in um
wav_max:                1.3                             # in um
t_exp:                  18e4                                    # in second
obj_tag:                ROS128b_J_H2O                   # appear in saved pkl file
st_spec_path:           template.star.fits                      # spec fits file for star
st_template_spec_path:  lte035-4.5-0.0a+0.0.BT-Settl.fits       # template spec fits file to generate star fits file
st_inclination_deg:     60.0                                    # in degree
st_rotation_vel:        2.7e3                                   # in m/s
st_radial_vel:          0e3                                     # in m/s positive is blue shift
pl_spec_path:           template.planet.fits                    # spec fits file for planet
pl_template_spec_path:  GeometricA_Earth_LowCloud_UltraRes1.dat # template spec fits file to generate planet fits file
pl_inclination_deg:     60.0                                    # in degree
pl_rotation_vel:        5e2                                     # in m/s
pl_radial_vel:          20.4e3                                  # in m/s positive is blue shift
pl_st_distance:         7.5e9                                   # in m 1 AU = 1.496e11 m
pl_radius:              9e6                             # in m 1 Earth Radius = 6378000.0 m
pl_illumination:        0.3                                     # fraction of illumination for a planet
zodi_spec_path:         ../modern_cube_zodi_dir                 #Exozodi simulation fits file
spec_reso:              1024e2                          # spectral resolution
distance:               3.5                                     # in pc
spec_tran_path:         Earth_atm_trans_NSO_GEMINI_N.txt        # atmosphere transmission file path
spec_radi_path:         mk_skybg_zmnq_16_15_ph.dat              # atmosphere emission file path
telescope_size:         10.0                                    # in m
num_surfaces:           10                      # number of warm optical surfaces in the telescope
temperature:            100                             # temperature of the telescope in K
exozodi_level:          1.0                             # level of exozodi relative to the Solar System
pl_st_contrast:         1e-5                    # star light suppression at fiber position
read_noise:             0.0                                     # in electron / pixel per read
dark_current:           0.0                                     # in elsctron / s / pixel
fiber_size:             1.0                                     # in lambda / D fiber sky projection
pixel_sampling:         3.0                                     # pixels per resolution element
throughput:             0.01                                    # end to end throughput
wfc_residual:           5.0                                     # in nm
template_path_thermal:  lte012-3.5-0.0a+0.0.BT-Settl.fits       # thermal template spec fits file to generate planet fits file
template_path:          template.H2O.fits                       # spec fits file for template
template_tag:           H2O                             # appear in saved pkl file
template_type:          reflection                              # reflection or thermal

```

This is the file in which you set up parameters in HDC simulations. Many fields are self-explanatory.
1, For young self luminous planets, we use template_type "thermal".
2, pl_template_spec_path is the fits file that are used to generate observed spectrum.
3, template_path_thermal is the fits file that stores the template spectrum in cross correlation. You can also use, e.g, HR8799_H2O.fits, or HR8799_CO.fits or HR8799_CH4.fits
4, all fits file should follow the PHENOEX spectrum format. That is, for any new fits file used by the code, it has to use the PHENOEX spectrum spectrum (e.g., lte012-3.5-0.0a+0.0.BT-Settl.fits) as a template and replace the wavelength and flux information within.

There are large files that I cannot upload to Github, i.e., a folder named *modern_cube_zodi_dir*, the path of which can be set up in the .init file. The folder can be can be downloaded [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/wang_12220_osu_edu/EtUnbK-4Yz9KsEWCiEQpc7AB0SAyeUMzVbhk6nJERq2-aw?e=ZAbohR):

After downloading the folder, make sure you set up corretly the path in the .init file, e.g.,:
```zodi_spec_path:         ../modern_cube_zodi_dir                 #Exozodi simulation fits file```

After set up. you can run:
```python  hci_hrs_sim.py```

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

