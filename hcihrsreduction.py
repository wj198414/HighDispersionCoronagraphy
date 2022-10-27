import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.stats
from datetime import datetime
from crosscorrelationfunction import CrossCorrelationFunction
from spectrum import Spectrum

class HCI_HRS_Reduction():
    def __init__(self, hci_hrs_obs, template, save_flag=False, obj_tag="a", template_tag="b", speckle_flag=False, resolution_elements_in_ccf=5e2):
        self.hci_hrs_obs = hci_hrs_obs
        self.template = template
        self.save_flag=save_flag
        self.obj_tag = obj_tag
        self.template_tag = template_tag
        self.speckle_flag = speckle_flag
        self.resolution_elements_in_ccf = np.max([hci_hrs_obs.star.spec_reso/128e2*resolution_elements_in_ccf, hci_hrs_obs.star.spec_reso/64e2*resolution_elements_in_ccf])        
        self.execute()

    def execute(self):
        # get template spectrum for cross correlation
        template_chunk = self.getSpecChunk(self.template.wavelength, self.template.flux)
        self.template.wavelength = template_chunk["Wavelength"]
        self.template.flux = template_chunk["Flux"]
        self.template = self.removeNanInSpecChunk(self.template)
        # rotational and spectral broaden template and resample template to instrument grid
        self.template.resampleSpec(self.hci_hrs_obs.star_spec_chunk.wavelength)
        self.template_R1000 = self.template.copy().spectral_blur(rpower=1e3, quick_blur=False)
        #self.template.rotational_blur(rot_vel=self.hci_hrs_obs.planet.rotation_vel)
        self.template.spectral_blur(rpower=self.template.spec_reso, quick_blur=False)
        self.template_resample = self.template.resampleSpectoSpectrograph(pixel_sampling=self.hci_hrs_obs.instrument.pixel_sampling)
        #self.template_resample.writeSpec(file_name="template_resample.txt")
        self.template_R1000 = self.template_R1000.resampleSpec(self.template_resample.wavelength)
        if self.hci_hrs_obs.atmosphere != None:
            # remove sky emission with spectrum obteined from the sky fiber
            self.obs_emission_removed = self.removeSkyEmission(flag_plot=True)
            # remove star and atmospheric transmission with spectrum obtained from the star fiber
            self.obs_st_at_removed = self.removeSkyTransmissionStar(flag_plot=True)
            #self.obs_st_at_removed = self.obs_emission_removed
            #self.plotObsTemplate()
            # apply high pass filter to remove low frequency component
            self.template_high_pass = self.template_resample.applyHighPassFilter()
            self.obs_high_pass = self.obs_st_at_removed.applyHighPassFilter()
            # write spectra
            self.template_high_pass.writeSpec(file_name="template_high_pass.txt")
            self.obs_high_pass.writeSpec(file_name="obs_high_pass.txt")
            # cross correlate reduced spectrum and template
            self.ccf_noise_less = self.obs_high_pass.crossCorrelation(self.template_high_pass, spec_mask=None, long_array=False, speed_flag=False)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            self.ccf_noise_less = self.ccf_noise_less.getCCFchunk(vmin=-self.resolution_elements_in_ccf*vel_pixel+self.hci_hrs_obs.planet.radial_vel, vmax=self.resolution_elements_in_ccf*vel_pixel+self.hci_hrs_obs.planet.radial_vel)
            self.ccf_peak = self.ccf_noise_less.calcPeak()
            # simulate observation with noise
            result = self.simulateSingleMeasurement(plot_flag=True)
            print(result)
            self.writeLog(result)
            #result = self.simulateMultiMeasurement(num_sim=10, ground_flag=True)
            result = self.simulateMultiMeasurement(num_sim=100, flag_plot=False, ground_flag=True, speckle_flag=self.speckle_flag, spec_mask=None, long_array=False, speed_flag=False)
        else:
            #self.obs_emission_removed = self.hci_hrs_obs.obs_spec_resample.copy()
            self.obs_emission_removed = self.removeSkyEmission(flag_plot=False)
            self.obs_st_at_removed = self.removeSkyTransmissionStar(flag_plot=False)            
            #obs_norm = self.obs_st_at_removed.getSpecNorm(num_chunks=20, poly_order=3)
            print(str(datetime.now()))
            #Normalization is to make sure that the input spectrum for the CCF is flat; 
            #this is especially important for large wavelength coverage.
            obs_norm = self.getStarNorm(self.hci_hrs_obs.obs_st_resample.flux, long_array=True)
            obs_norm = obs_norm.value / np.median(obs_norm.value)# * np.median(self.obs_st_at_removed.flux)
            #obs_norm = np.median(self.obs_st_at_removed.flux) 
            self.obs_st_at_removed.flux = self.obs_st_at_removed.flux / obs_norm
            self.obs_st_at_removed.noise = self.obs_st_at_removed.noise / obs_norm
            if 1 == 0:
                plt.plot(self.hci_hrs_obs.obs_st_resample.flux, label="st")
                plt.plot(self.obs_st_at_removed.flux, label="after")
                plt.legend()
                plt.show()  
            mask_arr = np.where((self.template_R1000.flux / np.nanmedian(self.template_R1000.flux)) > 0.99)
            #mask_arr = np.where((self.template_resample.flux / np.nanmedian(self.template_resample.flux)) > 1e9)
            if 1 == 0:
                #plt.errorbar(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux, yerr=self.obs_st_at_removed.noise)
                plt.figure()
                plt.errorbar(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux, self.obs_st_at_removed.noise)
                #plt.plot(self.template_resample.wavelength, self.template_resample.flux / np.median(self.template_resample.flux))
                #plt.plot(self.template_resample.wavelength[mask_arr], self.template_resample.flux[mask_arr] / np.median(self.template_resample.flux[mask_arr]))
                plt.plot(self.obs_st_at_removed.wavelength[mask_arr], self.obs_st_at_removed.flux[mask_arr], "r.")
                plt.show(block=True)
            if self.speckle_flag:
                #self.cutoff_value = np.min([self.hci_hrs_obs.instrument.spec_reso / 6.0, 100.0])
                self.cutoff_value = 100.0
                #if self.cutoff_value < self.hci_hrs_obs.instrument.spec_reso:
                #    self.cutoff_value = self.cutoff_value 
                #else:
                #    self.cutoff_value = self.hci_hrs_obs.instrument.spec_reso / 2.0
                self.template_resample = self.template_resample.applyHighPassFilter(cutoff=self.cutoff_value)
                self.ccf_noise_less = self.obs_st_at_removed.applyHighPassFilter(cutoff=self.cutoff_value).crossCorrelation(self.template_resample, spec_mask=mask_arr, long_array=False, speed_flag=False)
            else:
                self.ccf_noise_less = self.obs_st_at_removed.crossCorrelation(self.template_resample, spec_mask=mask_arr, long_array=False, speed_flag=False)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            self.ccf_noise_less = self.ccf_noise_less.getCCFchunk(vmin=-self.resolution_elements_in_ccf*vel_pixel+self.hci_hrs_obs.planet.radial_vel, vmax=self.resolution_elements_in_ccf*vel_pixel+self.hci_hrs_obs.planet.radial_vel)
            self.ccf_peak = self.ccf_noise_less.calcPeak()
            result = self.simulateSingleMeasurement(ground_flag=False, plot_flag=False, speckle_flag=self.speckle_flag, spec_mask=mask_arr, long_array=False, speed_flag=False, flag_plot=False)
            print(result)
            self.writeLog(result)
            #result = self.simulateMultiMeasurement_2(num_sim=100, ground_flag=False, speckle_flag=self.speckle_flag, spec_mask=mask_arr, long_array=False, speed_flag=False)
            result = self.simulateMultiMeasurement(num_sim=100, flag_plot=False, ground_flag=False, speckle_flag=self.speckle_flag, spec_mask=mask_arr, long_array=False, speed_flag=False)

        if self.save_flag:
            self.saveObject()            


    def saveObject(self, save_dir="/scr/jwang/hci_hds/OO_hci_hrs/pkl_dir/"):
        hci_hrs_name = "{0:06.3f}_{1:06.3}_{2:06.0f}".format(self.hci_hrs_obs.wav_min, self.hci_hrs_obs.wav_max, self.hci_hrs_obs.t_exp)
        obj_tag = self.obj_tag
        template_tag = self.template_tag
        instrument_name = "{0:05.1f}_{1:08.2e}_{2:08.2e}_{3:04.1f}".format(self.hci_hrs_obs.instrument.telescope_size, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, self.hci_hrs_obs.instrument.read_noise)
        file_name = obj_tag+"_"+template_tag+"_"+hci_hrs_name+"_"+instrument_name+".pkl"
        with open(save_dir+file_name, "wb") as handle:
            pickle.dump(self, handle)

    def writeLog(self, result):
        hci_hrs_name = "{0:06.3f}_{1:06.3}_{2:06.0f}".format(self.hci_hrs_obs.wav_min, self.hci_hrs_obs.wav_max, self.hci_hrs_obs.t_exp)
        obj_tag = self.obj_tag
        template_tag = self.template_tag
        instrument_name = "{0:05.1f}_{1:08.2e}_{2:08.2e}_{3:04.1f}".format(self.hci_hrs_obs.instrument.telescope_size, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, self.hci_hrs_obs.instrument.read_noise)
        log_tag = obj_tag+"_"+template_tag+"_"+hci_hrs_name+"_"+instrument_name # the same format as pkl file name
        time_tag = str(datetime.now())
        vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
        vel_offset_in_pixel = np.abs(result["Center"] - self.hci_hrs_obs.planet.radial_vel) / vel_pixel
        with open("log.dat", "a+") as f:
            f.write("{0:80s},{1:8.2e},{2:8.2e},{3:8.2e},{4:10.1f},{5:10.2f},{6:10.2f},{7:50s}\n".format(log_tag, self.hci_hrs_obs.planet.radial_vel, vel_pixel, result["Center"], vel_offset_in_pixel, result["SNR_RMS"], result["SNR_vs_NoiseLess"], time_tag))

    def getStarNorm(self, spec, num_chunks=20.0, long_array=False):
        if not long_array:
            if int(len(spec) / num_chunks) % 2 != 0:
                obs_norm = scipy.signal.medfilt(spec, kernel_size = int(len(spec) / num_chunks))
            else:
                obs_norm = scipy.signal.medfilt(spec, kernel_size = int(len(spec) / num_chunks) - 1)
        else:
            num_pixels = len(spec)
            num_division = 4
            pix_division = int(np.floor(num_pixels / (num_division + 0.0)))
            for i in np.arange(num_division):
                obs_norm_tmp = self.getStarNorm(spec[i*pix_division:(i+1)*pix_division])
                if i == 0:
                    obs_norm = obs_norm_tmp
                else:
                    obs_norm = np.hstack((obs_norm, obs_norm_tmp))
            obs_norm = np.hstack((obs_norm, spec[(i+1)*pix_division:]))

        return(obs_norm)

    def simulateSingleMeasurement(self, ground_flag=True, plot_flag=False, speckle_flag=False, **kwargs):
        if ground_flag:
            spec = self.obs_st_at_removed.applyHighPassFilter().generateNoisySpec()
            ccf = spec.crossCorrelation(self.template_high_pass, **kwargs)
        else:
            if speckle_flag:
                spec = self.obs_st_at_removed.generateNoisySpec(speckle_noise=True, star_flux=np.median(self.hci_hrs_obs.obs_st_resample.flux)) 
                spec = spec.applyHighPassFilter(cutoff=self.cutoff_value) # self.hci_hrs_obs.obs_st_resample is after starlight suppression
                spec = spec.applyHighPassFilter(cutoff=self.hci_hrs_obs.instrument.spec_reso*1.0, pass_type='low')
                ccf = spec.crossCorrelation(self.template_resample, **kwargs)
            else:
                spec = self.obs_st_at_removed.generateNoisySpec(speckle_noise=False)
                ccf = spec.crossCorrelation(self.template_resample, **kwargs)
        if plot_flag:
            plt.plot(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux / np.median(self.obs_st_at_removed.flux), alpha=0.5, label="obs")
            if ground_flag:
                plt.plot(self.template_high_pass.wavelength, self.template_high_pass.flux / np.max(self.template_high_pass.flux) , alpha=0.5, label="temp")
            else:
                plt.plot(self.template_resample.wavelength, self.template_resample.flux / np.max(self.template_resample.flux) , alpha=0.5, label="temp")
            #plt.plot(spec.wavelength, spec.flux, alpha=0.5, label="obs high pass noise")
            plt.legend()
            plt.show()
        vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
        ccf = ccf.getCCFchunk(vmin=-self.resolution_elements_in_ccf*vel_pixel+self.hci_hrs_obs.planet.radial_vel, vmax=self.resolution_elements_in_ccf*vel_pixel+self.hci_hrs_obs.planet.radial_vel)

        #cen = ccf.calcCentroid()
        cen = self.hci_hrs_obs.planet.radial_vel 

        #peak = ccf.calcPeak()
        dif = np.abs(ccf.vel - cen)
        ind = np.where(dif == np.min(dif))[0][0]
        #pix_search = int(np.round(1e4 / vel_pixel)) # within 10 km/s 
        pix_search = 2 # within 2 vel_pixel 
        peak = np.max(ccf.ccf[ind-pix_search:ind+pix_search+1])

        ccf_orig = CrossCorrelationFunction(ccf.vel, ccf.ccf)
        #ccf.ccf = ccf.ccf - self.ccf_noise_less.ccf
        #ccf.ccf[ind-pix_search:ind+pix_search+1] = ccf_orig.ccf[ind-pix_search:ind+pix_search+1]

        if plot_flag:
            plt.plot(ccf.vel, ccf.ccf, "bo-", alpha=0.5)
            plt.plot(self.ccf_noise_less.vel, self.ccf_noise_less.ccf, "ro-", alpha=0.5)
            plt.plot(ccf.vel, ccf.ccf - self.ccf_noise_less.ccf, "go-", alpha=0.5)
            plt.show()

        #snr_rms = ccf.calcSNRrms(peak=peak)
        snr_rms = ccf.calcSNRrmsNoiseless(self.ccf_noise_less, peak=peak)
        snr_vs_noise_less = ccf.calcSNRnoiseLess(self.ccf_noise_less)

        return({"CCF":ccf, "Center":cen, "SNR_RMS":snr_rms, "SNR_vs_NoiseLess":snr_vs_noise_less, "CCF_peak":peak})

    def simulateMultiMeasurement(self, num_sim=10, flag_plot=False, **kwargs):
        info_arr = np.zeros((3, num_sim))
        for i in np.arange(num_sim):
            result = self.simulateSingleMeasurement(**kwargs)
            print("now at ", i, result["SNR_RMS"])
            self.writeLog(result)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            info_arr[:, i] = [result["SNR_RMS"], result["CCF_peak"], 1]
        # peak_correction_rate is always 100% because in simulateSingleMeasurement result["Center"] is always the planet vel.
        peak_correction_rate = (len(info_arr[2,:][np.where(info_arr[2,:] == 1)]) + 0.0) / (num_sim + 0.0)
        if peak_correction_rate > 0.68:
            idx = np.where(info_arr[2,:] == 1)
            SNR_RMS_mean = 0.0
            SNR_RMS_std = 0.0
            if 1 == 0:
                SNR = np.transpose(info_arr[0,idx]).flatten() # SNR_RMS
            else:
                SNR = np.transpose(info_arr[1,idx]).flatten() # CCF _peak
                SNR = SNR / (scipy.stats.iqr(SNR, rng=(16,84)) / 2.0) # 16 to 84 percentile covers 2-sigma 
            if flag_plot:
                plt.hist(SNR, alpha=0.3, label="R ={0:4.0f}".format(self.hci_hrs_obs.instrument.spec_reso))
                plt.show()
            SNR_vs_NoiseLess_mean = np.sort(SNR)[int(0.32*len(SNR))]
            SNR_vs_NoiseLess_std = np.mean(SNR) / np.std(SNR)
        else:
            SNR_RMS_mean = 0.0
            SNR_RMS_std = 0.0
            SNR_vs_NoiseLess_mean = 0.0
            SNR_vs_NoiseLess_std = 0.0
        with open("multi_sim_log.dat", "a+") as f:
            f.write("{0:50s},{2:8.3e},{1:8.3e},{3:6.3f},{4:8.2e},{5:8.2e},{6:8.2e},{7:8.2e}\n".format(self.obj_tag, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, peak_correction_rate, vel_pixel, np.sort(SNR)[int(0.05*len(SNR))], np.sort(SNR)[int(0.32*len(SNR))], np.sort(SNR)[int(0.50*len(SNR))]))
        return([peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std])

    def simulateMultiMeasurement_2(self, num_sim=10, **kwargs):
        info_arr = np.zeros((3, num_sim))
        for i in np.arange(num_sim):
            result = self.simulateSingleMeasurement(**kwargs)
            self.writeLog(result)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            vel_offset_in_pixel = np.abs(result["Center"] - self.hci_hrs_obs.planet.radial_vel) / vel_pixel
            #result["CCF"].pltCCF()
            if vel_offset_in_pixel <= 10.0: # this may only be relavant to ground based observation
            #if vel_offset_in_pixel <= 3e5:
                info_arr[:, i] = [result["SNR_RMS"], result["CCF_peak"], 1]
            else:
                info_arr[:, i] = [0.0, 0.0, 0]
        peak_correction_rate = (len(info_arr[2,:][np.where(info_arr[2,:] == 1)]) + 0.0) / (num_sim + 0.0)
        if peak_correction_rate > 0.5:
            idx = np.where(info_arr[2,:] == 1)
            SNR_RMS_mean = np.median(info_arr[0,idx])
            SNR_RMS_std = np.std(np.sort(np.transpose(info_arr[0,idx]))[1:-1])
            CCF_peak_mean = np.median(info_arr[1,idx])
            CCF_peak_std = np.std(np.sort(np.transpose(info_arr[1,idx]))[2:-2])
            n, bins = np.histogram(np.transpose(info_arr[1,idx]), bins=np.linspace(0, np.max(info_arr[1,idx]), 10))
            #plt.plot(bins[0:-1], n, "b")
            #plt.plot([self.ccf_peak, self.ccf_peak], [0,num_sim/2.0],"r--")
            #plt.show()
            if (n[0] == 0.0) & (np.sort(np.transpose(info_arr[1,idx]))[int(np.floor((num_sim-1.0)*0.15))] < self.ccf_peak):
                SNR_vs_NoiseLess_mean = CCF_peak_mean / CCF_peak_std
            else:
                SNR_vs_NoiseLess_mean = 0.0
            SNR_vs_NoiseLess_std = CCF_peak_mean / CCF_peak_std
        else:
            SNR_RMS_mean = 0.0
            SNR_RMS_std = 0.0
            SNR_vs_NoiseLess_mean = 0.0
            SNR_vs_NoiseLess_std = 0.0
        with open("multi_sim_log.dat", "a+") as f:
            f.write("{0:50s},{2:8.2e},{1:8.2e},{3:6.3f},{4:8.2e},{5:8.2e},{6:8.2e},{7:8.2e}\n".format(self.obj_tag, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std))
        return([peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std])

    def removeSkyEmission(self, flag_plot=False):
        if self.hci_hrs_obs.atmosphere != None: 
            spec = self.hci_hrs_obs.obs_spec_resample.copy()
            spec.wavelength = self.hci_hrs_obs.atm_radi_spec_chunk_resample.wavelength
            spec.flux = self.hci_hrs_obs.obs_spec_resample.flux - self.hci_hrs_obs.atm_radi_spec_chunk_resample.flux - self.hci_hrs_obs.obs_therm_resample.flux
            spec.noise = None
            spec.addNoise(np.sqrt(self.hci_hrs_obs.obs_spec_resample.noise**2 + self.hci_hrs_obs.atm_radi_spec_chunk_resample.noise**2 + self.hci_hrs_obs.obs_therm_resample.noise**2))
            if flag_plot:
                plt.plot(self.hci_hrs_obs.obs_spec_resample.wavelength, self.hci_hrs_obs.obs_spec_resample.flux, label="obs")
                plt.plot(self.hci_hrs_obs.atm_radi_spec_chunk_resample.wavelength, self.hci_hrs_obs.atm_radi_spec_chunk_resample.flux, label="sky")
                plt.plot(self.hci_hrs_obs.obs_therm_resample.wavelength, self.hci_hrs_obs.obs_therm_resample.flux, label="therm")
                plt.plot(spec.wavelength, spec.flux, label="after")
                plt.plot(spec.wavelength, spec.noise, label="noise")
                plt.yscale("log")
                plt.legend()
                plt.show()
        else:
            spec = self.hci_hrs_obs.obs_spec_resample.copy()
            spec.wavelength = self.hci_hrs_obs.obs_spec_resample.wavelength
            spec.flux = self.hci_hrs_obs.obs_spec_resample.flux - self.hci_hrs_obs.obs_therm_resample.flux
            spec.noise = None
            spec.addNoise(np.sqrt(self.hci_hrs_obs.obs_spec_resample.noise**2 + self.hci_hrs_obs.obs_therm_resample.noise**2))
            if flag_plot:
                plt.plot(self.hci_hrs_obs.obs_spec_resample.wavelength, self.hci_hrs_obs.obs_spec_resample.flux, label="obs")
                plt.plot(self.hci_hrs_obs.obs_therm_resample.wavelength, self.hci_hrs_obs.obs_therm_resample.flux, label="therm")
                plt.plot(spec.wavelength, spec.flux, label="after")
                plt.plot(spec.wavelength, spec.noise, label="noise")
                plt.yscale("log")
                plt.legend()
                plt.show()
        return(spec)

    #Noise model and final observed spectrum made more realistic.  Can't easily remove thermal+zodi backgrounds!
        
    def removeSkyTransmissionStar(self, flag_plot=False):
        flx_st_atm = self.hci_hrs_obs.obs_st_resample.flux
        flx_obs = self.obs_emission_removed.flux
        wav_obs = self.obs_emission_removed.wavelength
        flx_st_atm_norm = flx_st_atm / np.median(flx_st_atm)
        obs_st_at_removed = self.hci_hrs_obs.obs_st_resample.copy()
        obs_st_at_removed.wavelength = wav_obs
        if self.hci_hrs_obs.atmosphere != None:
            self.hci_hrs_obs.obs_st_resample.flux *= self.hci_hrs_obs.instrument.pl_st_contrast # obs_st_resample is now attenuated by coronagraph
            noise = np.sqrt(self.obs_emission_removed.noise**2 + self.hci_hrs_obs.calNoise(self.hci_hrs_obs.obs_st_resample)**2)
            atm_tran = self.hci_hrs_obs.obs_atm_tran_resample.copy()
            st_tran_free_flux = self.hci_hrs_obs.obs_st_resample.flux / atm_tran.flux
            obs_st_at_removed.flux = (self.obs_emission_removed.flux / atm_tran.flux) - st_tran_free_flux
            # set deep transmission region to median 
            idx = np.where(atm_tran.flux < 1e-1)
            obs_st_at_removed.flux[idx] = np.median(obs_st_at_removed.flux)
            obs_st_at_removed.noise = None
            obs_st_at_removed.addNoise(np.abs(noise))
        else:
            self.hci_hrs_obs.obs_st_resample.flux *= self.hci_hrs_obs.instrument.pl_st_contrast # obs_st_resample is now attenuated by coronagraph
            noise = np.sqrt(self.obs_emission_removed.noise**2 + self.hci_hrs_obs.calNoise(self.hci_hrs_obs.obs_st_resample)**2)
            atm_tran = self.hci_hrs_obs.obs_st_resample.copy()
            atm_tran.flux[:] = 1.0
            st_tran_free_flux = self.hci_hrs_obs.obs_st_resample.flux / atm_tran.flux
            obs_st_at_removed.flux = (self.obs_emission_removed.flux / atm_tran.flux) - st_tran_free_flux 
            obs_st_at_removed.noise = None
            obs_st_at_removed.addNoise(np.abs(noise))
        if flag_plot:
            plt.plot(self.obs_emission_removed.wavelength, self.obs_emission_removed.flux, label="before")
            plt.plot(self.hci_hrs_obs.obs_pl_resample.wavelength, self.hci_hrs_obs.obs_pl_resample.flux, label="pl")
            plt.plot(self.hci_hrs_obs.obs_st_resample.wavelength, self.hci_hrs_obs.obs_st_resample.flux, label="st+atm")
            plt.plot(obs_st_at_removed.wavelength, obs_st_at_removed.flux, label="after")
            plt.plot(obs_st_at_removed.wavelength, obs_st_at_removed.noise, label="noise")
            plt.plot(atm_tran.wavelength, atm_tran.flux, label="tran")
            plt.plot(obs_st_at_removed.wavelength, obs_st_at_removed.flux / self.hci_hrs_obs.obs_pl_resample.flux, label="after/pl")
            plt.plot(self.hci_hrs_obs.obs_st_resample.wavelength, st_tran_free_flux, label="st")
            #plt.plot(self.hci_hrs_obs.obs_st_resample.wavelength, self.obs_emission_removed.flux / atm_tran.flux, label="pl+st w/o atm")
            plt.yscale("log")
            plt.legend()
            plt.show()
        return(obs_st_at_removed)

    def plotObsTemplate(self, plotSkyStAtmRemoved=True, plotTemplate=True, plotStAtm=True, plotObs=False, plotObsSkyRemoved=True):
        flx_obs = self.hci_hrs_obs.obs_spec_resample.flux
        flx_obs_emission_removed = self.obs_emission_removed.flux
        flx_template = self.template.flux
        flx_st_atm = self.hci_hrs_obs.obs_st_resample.flux
        flx_obs_st_at_removed = self.obs_st_at_removed.flux
        wav_obs = self.hci_hrs_obs.obs_spec_resample.wavelength
        wav_template = self.template.wavelength
        plt.figure()
        if plotTemplate:
            plt.plot(wav_template, flx_template / np.median(flx_template), label="Template")
        if plotStAtm:
            plt.plot(wav_obs, flx_st_atm / np.median(flx_st_atm), label="Star Atm only")
        if plotObs:
            plt.plot(wav_obs, flx_obs / np.median(flx_obs), label="Observed")
        if plotObsSkyRemoved:
            plt.plot(wav_obs, flx_obs_emission_removed / np.median(flx_obs_emission_removed), label="Sky removed")
        if plotSkyStAtmRemoved:
            plt.plot(wav_obs, flx_obs_st_at_removed / np.median(flx_obs_st_at_removed), label="Sky Star removed")
        plt.ylim(np.min(flx_st_atm / np.median(flx_st_atm)), 2.0 * np.max(flx_st_atm / np.median(flx_st_atm)))
        plt.legend()
        plt.show(block=True)

    def getSpecChunk(self, wav, flx):
        # get spectrum within wavelength range
        idx = ((wav < self.hci_hrs_obs.wav_max) & (wav > self.hci_hrs_obs.wav_min))
        return {'Wavelength':wav[idx],'Flux':flx[idx]}

    def removeNanInSpecChunk(self, spectrum):
        idx = np.isnan(spectrum.flux)
        spectrum.flux[idx] = np.nanmedian(spectrum.flux)
        return(spectrum)

def readInit(init_file="MdwarfPlanet.init"):
    initDict = {}
    with open(init_file, 'r') as f:
        for line in f:
            key_value = line.split('#')[0]
            key = key_value.split(':')[0].strip(' \t\n\r')
            value = key_value.split(':')[1].strip(' \t\n\r')
            initDict[key] = value
    return(initDict)
