import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import constants as c
from spectrum import Spectrum

class HCI_HRS_Observation():
    def __init__(self, wav_min, wav_max, t_exp, target_pl, target_st, instrument, thermal_background, zodi, atmosphere=None):
        self.wav_min = wav_min
        self.wav_max = wav_max
        self.t_exp = t_exp
        self.planet = target_pl
        self.star = target_st
        self.instrument = instrument
        self.thermbg = thermal_background
        self.zodi = zodi
        self.atmosphere = atmosphere
        self.execute()

    def execute(self):

        # Construct HCI+HRS spectrum

        # get star spectrum within wavelength range, remove NaNs, and calculate total star flux
        
        spec_star = self.getSpecChunk(self.star.wavelength, self.star.flux)
        self.star_spec_chunk = Spectrum(spec_star["Wavelength"], spec_star["Flux"], spec_reso=self.star.spec_reso)
        self.star_spec_chunk = self.removeNanInSpecChunk(self.star_spec_chunk)
        self.star_spec_chunk.evenSampling()
        self.star_spec_chunk.flux = self.specToPhoton(self.star_spec_chunk.wavelength, self.star_spec_chunk.flux, self.star.distance, self.star.PHXREFF, self.instrument.telescope_size, self.instrument.throughput, self.t_exp)
        #Chop the one of the ends of the spectrum off because we have no delta lambda for it
        self.star_spec_chunk.wavelength = self.star_spec_chunk.wavelength[1:]
        self.star_total_flux = self.getTotalFlux(self.star_spec_chunk.flux)

        # get planet spectrum within wavelength range, remove NaNs, and calculate total planet flux
        
        spec_planet = self.getSpecChunk(self.planet.wavelength, self.planet.flux)
        self.planet_spec_chunk = Spectrum(spec_planet["Wavelength"], spec_planet["Flux"], spec_reso=self.planet.spec_reso)
        self.planet_spec_chunk = self.removeNanInSpecChunk(self.planet_spec_chunk)
        self.planet_spec_chunk.evenSampling() #Does this actually badly affect the total flux?
        self.planet_spec_chunk.flux = self.specToPhoton(self.planet_spec_chunk.wavelength, self.planet_spec_chunk.flux, self.planet.distance, self.planet.PHXREFF, self.instrument.telescope_size, self.instrument.throughput, self.t_exp)
        self.planet_spec_chunk.wavelength = self.planet_spec_chunk.wavelength[1:]
        self.planet_total_flux = self.getTotalFlux(self.planet_spec_chunk.flux)
        # resample planet spectrum to star wavelength scale
        self.planet_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)

        #Get thermal background spectrum within wavelength range, remove NaNs, calculate total thermal flux, and resample to stellar wavelength scale
   
        spec_therm = self.getSpecChunk(self.thermbg.wavelength, self.thermbg.flux)
        self.therm_spec_chunk = Spectrum(spec_therm["Wavelength"], spec_therm["Flux"], spec_reso=self.thermbg.spec_reso)
        self.therm_spec_chunk = self.removeNanInSpecChunk(self.therm_spec_chunk)
        self.therm_spec_chunk.evenSampling() #Does this actually badly affect the total flux?
        self.therm_spec_chunk.flux = self.thermToPhoton(self.therm_spec_chunk.wavelength, self.therm_spec_chunk.flux, self.t_exp)
        self.therm_spec_chunk.wavelength = self.therm_spec_chunk.wavelength[1:]
        self.therm_total_flux = self.getTotalFlux(self.therm_spec_chunk.flux)
        self.therm_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)
        
        #Get exozodiacal spectrum within wavelength range, calculate total zodi flux, and resample to stellar wavelength scale

        spec_zodi = self.getSpecChunk(self.zodi.wavelength, self.zodi.flux)
        self.zodi_spec_chunk = Spectrum(spec_zodi["Wavelength"], spec_zodi["Flux"], spec_reso=self.zodi.spec_reso)
        self.zodi_spec_chunk = self.removeNanInSpecChunk(self.zodi_spec_chunk)
        self.zodi_spec_chunk.flux = self.zodiToPhoton(self.zodi_spec_chunk.wavelength, self.zodi_spec_chunk.flux, self.instrument.telescope_size, self.instrument.throughput, self.t_exp)
        self.zodi_spec_chunk.wavelength = self.zodi_spec_chunk.wavelength[1:]
        self.zodi_total_flux = self.getTotalFlux(self.zodi_spec_chunk.flux)
        self.zodi_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)

        #Calculate exozodi background (add later)
      
        # Considering Earth's atmosphere, e.g., ground-based observation
        if self.atmosphere != None:
            # get the transmission spectrum 
            spec_atm_tran = self.getSpecChunk(self.atmosphere.spec_tran_wav, self.atmosphere.spec_tran_flx)
            self.atm_tran_spec_chunk = Spectrum(spec_atm_tran["Wavelength"], spec_atm_tran["Flux"], spec_reso=self.star.spec_reso)
            self.atm_tran_spec_chunk = self.removeNanInSpecChunk(self.atm_tran_spec_chunk)
            self.atm_tran_spec_chunk.flux[np.where(self.atm_tran_spec_chunk.flux < 1e-9)] = 1e-9
            # get the emission spectrum
            spec_atm_radi = self.getSpecChunk(self.atmosphere.spec_radi_wav, self.atmosphere.spec_radi_flx)
            self.atm_radi_spec_chunk = Spectrum(spec_atm_radi["Wavelength"], spec_atm_radi["Flux"], spec_reso=self.star.spec_reso)
            self.atm_radi_spec_chunk = self.removeNanInSpecChunk(self.atm_radi_spec_chunk)
            self.atm_radi_spec_chunk.evenSampling()
            # calculate total flux from sky emission
            self.sky_total_flux = self.atmosphere.getTotalSkyFlux(self.wav_min, self.wav_max, tel_size=self.instrument.telescope_size, multiple_lambda_D=self.instrument.fiber_size, t_exp=self.t_exp, eta_ins=self.instrument.throughput)
            # resample transmission and emission spectra to star wavelength scale
            self.atm_tran_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)
            self.atm_radi_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)

            # doppler shift and rotationally broaden planet and star spectra
            self.planet_spec_chunk.dopplerShift(rv_shift=self.planet.radial_vel)
            #self.planet_spec_chunk.rotational_blur(rot_vel=self.planet.rotation_vel)
            self.star_spec_chunk.dopplerShift(rv_shift=self.star.radial_vel)
            #self.star_spec_chunk.rotational_blur(rot_vel=self.star.rotation_vel)
            # doppler shift transmission and emission spectra
            self.atm_tran_spec_chunk.dopplerShift(rv_shift=self.atmosphere.radial_vel)
            self.atm_radi_spec_chunk.dopplerShift(rv_shift=self.atmosphere.radial_vel)  

            # calculate sky transmission rate
            self.sky_transmission = np.sum(self.atm_tran_spec_chunk.flux) / (0.0 + len(self.atm_tran_spec_chunk.flux))

            # construct spectrum with planet, star and atmospheric transmission
            # pl_st spectrum = (planet + star * contrast) * transmission 
            # pl_st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_pl_st = ((self.planet_spec_chunk.flux + self.star_spec_chunk.flux * self.instrument.pl_st_contrast) * self.atm_tran_spec_chunk.flux)
            self.obs_pl_st = Spectrum(obs_spec_wav, obs_spec_pl_st, spec_reso=self.star.spec_reso) 
            self.obs_pl_st.spectral_blur(rpower=self.star.spec_reso)
            self.obs_pl_st_resample = self.obs_pl_st.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            # pl_st spectrum is scaled by total flux from the star and the planet and the atmosphere transmission
            self.obs_pl_st_resample.scaleSpec(total_flux=self.sky_transmission * (self.planet_total_flux + self.star_total_flux * self.instrument.pl_st_contrast))

            # construct spectrum of atmospheric transmission only
            # atm spectrum = transmission 
            # atm spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_atm_tran = self.atm_tran_spec_chunk.flux
            self.obs_atm_tran = Spectrum(obs_spec_wav, obs_spec_atm_tran, spec_reso=self.star.spec_reso)
            self.obs_atm_tran.spectral_blur(rpower=self.star.spec_reso)
            self.obs_atm_tran_resample = self.obs_atm_tran.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)

            # construct spectrum with star and atmospheric transmission
            # st spectrum = star * transmission 
            # st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_st = self.star_spec_chunk.flux * self.atm_tran_spec_chunk.flux
            self.obs_st = Spectrum(obs_spec_wav, obs_spec_st, spec_reso=self.star.spec_reso)
            self.obs_st.spectral_blur(rpower=self.star.spec_reso)
            self.obs_st_resample = self.obs_st.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            # st spectrum is scaled by total flux from the star and the atmosphere transmission
            self.obs_st_resample.scaleSpec(total_flux=self.sky_transmission * self.star_total_flux)

            # construct spectrum with planet 
            # pl spectrum = planet  
            # pl spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_pl = self.planet_spec_chunk.flux 
            self.obs_pl = Spectrum(obs_spec_wav, obs_spec_pl, spec_reso=self.planet.spec_reso)
            self.obs_pl.spectral_blur(rpower=self.planet.spec_reso)
            self.obs_pl_resample = self.obs_pl.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            # pl spectrum is scaled by total flux from the planet and the atmosphere transmission
            self.obs_pl_resample.scaleSpec(total_flux=self.sky_transmission * self.planet_total_flux)

            # construct spectrum with atmospheric emission, which is independent (addable) with pl_st spectrum
            self.atm_radi_spec_chunk.spectral_blur(rpower=self.star.spec_reso)
            self.atm_radi_spec_chunk_resample = self.atm_radi_spec_chunk.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.atm_radi_spec_chunk_resample.scaleSpec(total_flux=self.sky_total_flux)

            #Construct thermal background spectrum
            #Thermal spectrum = thermbg
            #Thermal spectrum is then spectrally blurred and resampled to spectrograph wavelength grid

            self.obs_therm = self.therm_spec_chunk.copy()
            self.obs_therm.spectral_blur(rpower=self.thermbg.spec_reso, quick_blur=False)
            self.obs_therm_resample = self.obs_therm.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_therm_resample.scaleSpec(total_flux=self.therm_total_flux)

            self.obs_zodi = self.zodi_spec_chunk.copy()
            self.obs_zodi.spectral_blur(rpower=self.zodi.spec_reso, quick_blur=False)
            self.obs_zodi_resample = self.obs_zodi.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_zodi_resample.scaleSpec(total_flux=self.zodi_total_flux)

            # construct final spectrum with pl_st and radi
            self.obs_spec_resample = Spectrum(self.atm_radi_spec_chunk_resample.wavelength, self.atm_radi_spec_chunk_resample.flux + self.obs_pl_st_resample.flux + self.obs_zodi_resample.flux + self.obs_therm_resample.flux, spec_reso=self.star.spec_reso)

            if 1 == 0:
                plt.figure()
                plt.plot(self.obs_st_resample.wavelength, self.obs_st_resample.flux*self.instrument.pl_st_contrast, color="b", label="star+atm")
                plt.plot(self.obs_pl_resample.wavelength, self.obs_pl_resample.flux, color="orange", label="planet")
                plt.plot(self.obs_therm_resample.wavelength, self.obs_therm_resample.flux, color="g", label="thermal")
                plt.plot(self.obs_zodi_resample.wavelength, self.obs_zodi_resample.flux, color="r", label="zodi")
                plt.plot(self.obs_spec_resample.wavelength, self.obs_spec_resample.flux, color="yellow", label="obs")
                plt.plot(self.atm_radi_spec_chunk_resample.wavelength, self.atm_radi_spec_chunk_resample.flux, color="black", label="sky")
                plt.plot(self.obs_atm_tran_resample.wavelength, self.obs_atm_tran_resample.flux, label="tran")
                plt.yscale("log")
                plt.legend()
                plt.show(block=True)

            # calculate noise for obs_spec_resample, atm_radi_spec_chunk_resample, and obs_st_resample
            noise = self.calNoise(self.obs_spec_resample)
            self.obs_spec_resample.addNoise(noise)
            noise = self.calNoise(self.atm_radi_spec_chunk_resample)
            self.atm_radi_spec_chunk_resample.addNoise(noise)
            noise = self.calNoise(self.obs_st_resample)
            self.obs_st_resample.addNoise(noise)
            noise = self.calNoise(self.obs_pl_resample)
            self.obs_pl_resample.addNoise(noise)
            noise = self.calNoise(self.obs_therm_resample)
            self.obs_therm_resample.addNoise(noise)
            noise = self.calNoise(self.obs_zodi_resample)
            self.obs_zodi_resample.addNoise(noise)

            # write to file 
            self.atm_tran_spec_chunk.writeSpec(file_name="atm.txt",flag_python2=False,flag_append=False)
            self.atm_radi_spec_chunk_resample.writeSpec(file_name="rad.txt",flag_python2=False,flag_append=False)
            self.obs_pl_resample.writeSpec(file_name="pl.txt",flag_python2=False,flag_append=False)
            self.obs_st_resample.writeSpec(file_name="st.txt",flag_python2=False,flag_append=False)
            self.obs_pl_st_resample.writeSpec(file_name="pl_st.txt",flag_python2=False,flag_append=False)

        # Excluding Earth's atmosphere, e.g., space-based observation  
        else:
            self.sky_total_flux = 0.0
            self.sky_transmission = 1.0
            # doppler shift and rotationally broaden planet and star spectra
         
            self.planet_spec_chunk.dopplerShift(rv_shift=self.planet.radial_vel)
            #self.planet_spec_chunk.rotational_blur(rot_vel=self.planet.rotation_vel)
            self.star_spec_chunk.dopplerShift(rv_shift=self.star.radial_vel)
            #self.star_spec_chunk.rotational_blur(rot_vel=self.star.rotation_vel)
            
            # construct spectrum with star only
            # st spectrum = star  
            # st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            
            self.obs_st = self.star_spec_chunk.copy()
            self.obs_st.spectral_blur(rpower=self.star.spec_reso, quick_blur=False)
            self.obs_st_resample = self.obs_st.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_st_resample.scaleSpec(total_flux=self.star_total_flux)
            
            # construct spectrum with planet only
            # pl spectrum = planet  
            # pl spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            
            self.obs_pl = self.planet_spec_chunk.copy()
            self.obs_pl.spectral_blur(rpower=self.planet.spec_reso, quick_blur=False)
            self.obs_pl_resample = self.obs_pl.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_pl_resample.scaleSpec(total_flux=self.planet_total_flux)
            
            #Construct thermal background spectrum
            #Thermal spectrum = thermbg
            #Thermal spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
         
            self.obs_therm = self.therm_spec_chunk.copy()
            self.obs_therm.spectral_blur(rpower=self.thermbg.spec_reso, quick_blur=False)
            self.obs_therm_resample = self.obs_therm.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_therm_resample.scaleSpec(total_flux=self.therm_total_flux)
        
            self.obs_zodi = self.zodi_spec_chunk.copy()
            self.obs_zodi.spectral_blur(rpower=self.zodi.spec_reso, quick_blur=False)
            self.obs_zodi_resample = self.obs_zodi.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_zodi_resample.scaleSpec(total_flux=self.zodi_total_flux)

            if 1 == 0:
                plt.figure()
                plt.plot(self.obs_st_resample.wavelength, self.obs_st_resample.flux*self.instrument.pl_st_contrast, "b")
                plt.plot(self.obs_pl_resample.wavelength, self.obs_pl_resample.flux, "orange")
                plt.plot(self.obs_therm_resample.wavelength, self.obs_therm_resample.flux, "g")
                plt.plot(self.obs_zodi_resample.wavelength, self.obs_zodi_resample.flux, "r")
                plt.show(block=False)

            # construct spectrum with planet, star, and thermal background 
            # obs = (planet + star * contrast + thermbg) 
            # pl_st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            
            obs_spec_wav = self.obs_st_resample.wavelength
            obs_spec_flx = self.obs_st_resample.flux*self.instrument.pl_st_contrast + self.obs_pl_resample.flux + self.obs_therm_resample.flux + self.obs_zodi_resample.flux
            self.obs_spec_resample = Spectrum(obs_spec_wav, obs_spec_flx, spec_reso=self.star.spec_reso)

            '''obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_flx = self.planet_spec_chunk.flux + self.star_spec_chunk.flux * self.instrument.pl_st_contrast
            self.obs_spec = Spectrum(obs_spec_wav, obs_spec_flx, spec_reso=self.star.spec_reso)            
            self.obs_spec.spectral_blur(rpower=self.star.spec_reso, quick_blur=True)
            self.obs_spec_resample = self.obs_spec.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_spec_resample.scaleSpec(total_flux=self.planet_total_flux + self.star_total_flux * self.instrument.pl_st_contrast)'''

            # calculate noise for obs_spec_resample, atm_radi_spec_chunk_resample, and obs_st_resample
            
            noise = self.calNoise(self.obs_spec_resample)
            self.obs_spec_resample.addNoise(noise)
            noise = self.calNoise(self.obs_st_resample)
            self.obs_st_resample.addNoise(noise)
            noise = self.calNoise(self.obs_pl_resample)
            self.obs_pl_resample.addNoise(noise)
            noise = self.calNoise(self.obs_therm_resample)
            self.obs_therm_resample.addNoise(noise)
            noise = self.calNoise(self.obs_zodi_resample)
            self.obs_zodi_resample.addNoise(noise)

    def calNoise(self, spec):
        flx = spec.flux
        num_read = np.max([np.round(np.sort(flx)[int(0.9 * len(flx))] / 1e6), 1]) # 1e6 is linearity/persitence range
        #print("number of read = ", num_read)
        var = (flx + self.instrument.dark_current * self.t_exp + self.instrument.read_noise**2 * num_read)
        noise = np.sqrt(np.abs(var))
        idx = np.where(noise < 1.0)
        noise[idx] = 1.0
        return(noise)

    def getSpecChunk(self, wav, flx):
        # get spectrum within wavelength range
        idx = ((wav < self.wav_max) & (wav > self.wav_min))
        if np.size(wav[idx]) != 0 :
            return {'Wavelength':wav[idx],'Flux':flx[idx]}
        else:
            wave = np.arange(self.wav_min, self.wav_max, 1e-4)
            flux = np.zeros(np.shape(wave)) + 1e-99
            return {'Wavelength':wave,'Flux':flux}
               
    def removeNanInSpecChunk(self, spectrum):
        idx = np.isnan(spectrum.flux)
        spectrum.flux[idx] = np.nanmedian(spectrum.flux)
        return(spectrum)

    #Calculate the total number of photons in each pixel for the stellar and planetary
    #spectra.

    def specToPhoton(self, wav, flx, dis, PHXREFF, tel_size, eta_ins, t_exp):
        #dis in pc
        #PHXREFF in m, found in PHOENIX header
        #wav in um
        #flx in W/m^2/um/sr
        #tel_size in m
        #eta_ins considers both telescope and instrument throughput

        wav_u = wav * u.micron
        flx_u = flx * u.si.watt / u.meter / u.meter / u.micron
        dis_u = dis * u.pc
        PHXREFF_u = PHXREFF * u.meter
        tel_size_u = tel_size * u.meter
        t_exp_u = t_exp * u.second
        wav_u_inc = wav_u[1:] - wav_u[0:-1]

        photon_energy_u = (c.h*c.c/wav_u).decompose()

        flx_u_photon = (flx_u[1:] * np.pi * (tel_size_u / 2.0)**2 * wav_u_inc * (PHXREFF_u / dis_u)**2 * t_exp_u / photon_energy_u[1:]).decompose() * eta_ins

        return(flx_u_photon)

    #Calculate the photons per pixel from the thermal background of the instrument.  
    #This assumes diffraction-limited imaging (A*Omega ~ lambda^2).

    def thermToPhoton(self, wav, flx, t_exp):
        #wav in um
        #flx in W/um
        #t_exp in s
        #Coronagraph throughput doesn't matter - thermal background is even!
      
        wav_u = wav * u.micron
        flx_u = flx * u.si.watt / u.micron
        wav_u_inc = wav_u[1:] - wav_u[0:-1]
        t_exp_u = t_exp * u.second

        photon_energy_u = (c.h*c.c/wav_u).decompose()
        
        flx_u_photon = (flx_u[1:] * wav_u_inc * t_exp_u / photon_energy_u[1:]).decompose()

        return(flx_u_photon)

    #Calculate the photons per pixel from the exozodiacal background.

    def zodiToPhoton(self, wav, flx, tel_size, eta_ins, t_exp):
        #wav in um
        #flx in Jy
        #dis in pc
        #tel_size in m
        #eta_ins considers both telescope and instrument throughput, though as exozodi is not a point source,
        #it'll need to be more complicated eventually
        #t_exp in s

        wav_u = wav*u.um
        flx_u = flx*u.Jy
        wav_u_inc = wav_u[1:] - wav_u[0:-1]
        tel_size_u = tel_size*u.m
        t_exp_u = t_exp*u.s

        photon_energy_u = (c.c*c.h/wav_u).decompose()

        flx_u_photon = (flx_u[1:] * np.pi * (tel_size_u/2.)**2 * (c.c/wav_u[1:]**2) * wav_u_inc * eta_ins * t_exp_u / photon_energy_u[1:]).decompose()

        return(flx_u_photon)

    def getTotalFlux(self, flx_u_photon):
        return(np.sum(flx_u_photon)) 
