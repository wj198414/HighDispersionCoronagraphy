import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import scipy.constants
import pickle
from scipy import signal
from astropy import units as u
import numpy.fft as fft
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
import time
import scipy.interpolate
import scipy.fftpack as fp
from datetime import datetime
import astropy.io.ascii

class Target():
    def __init__(self, distance=10.0, spec_path=None, inclination_deg=90.0, rotation_vel=5e3, radial_vel=1e4, spec_reso=1e5):
        self.distance = distance
        self.spec_path =  spec_path
        self.spec_reso = spec_reso
        if self.spec_path != None:
            self.spec_data = pyfits.open(self.spec_path)[1].data
            self.wavelength = self.spec_data["Wavelength"]
            self.flux = self.spec_data["Flux"]
            self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
            self.spec_header = pyfits.open(self.spec_path)[1].header
            self.PHXREFF = self.spec_header["PHXREFF"]
        self.inclination_deg = inclination_deg
        self.rotation_vel = rotation_vel
        self.radial_vel = radial_vel

    def calStrehl(self):
        strehl = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / self.wav_med))**2) 
        return(strehl)

    def calRelativeStrehl(self):
        strehl_K = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / 2.0))**2)
        strehl = self.calStrehl()
        return(strehl / strehl_K)

class Spectrum():
    def __init__(self, wavelength, flux, spec_reso=1e5, norm_flag=False):
        self.wavelength = wavelength
        self.flux = flux
        self.spec_reso = spec_reso
        self.norm_flag = norm_flag
        self.noise = None

    def getSpecChunk(self, wav_min, wav_max):
        idx = ((self.wavelength < wav_max) & (self.wavelength > wav_min))
        return(Spectrum(self.wavelength[idx], self.flux[idx], spec_reso=self.spec_reso))

    def addNoise(self, noise):
        if self.noise == None:
            self.noise = noise
        else:
            print("Warning: spectrum noise already added")

    def generateNoisySpec(self):
        spec = self.copy()
        flx = self.flux
        flx_new = np.zeros(np.shape(flx))
        num = len(flx)
        i = 0
        while i < num:
            #flx_new[i] = np.max([np.random.poisson(np.round(flx[i]), 1)+0.0, np.random.normal(flx[i], self.noise[i], 1)])
            flx_new[i] = np.random.normal(flx[i], self.noise[i], 1)
            i = i + 1
        spec.flux = flx_new
        return(spec)

    def applyHighPassFilter(self, order = 5, cutoff = 100.0):
        # cutoff is number of sampling per 1 micron, so 100 means 0.01 micron resolution, about R = 100 at 1 micron
        x = self.wavelength
        y = self.flux
        fs = 1.0 / np.median(x[1:-1] - x[0:-2])
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        yy = scipy.signal.filtfilt(b, a, y)
        return(Spectrum(x, yy, spec_reso=self.spec_reso))

    def copy(self):
        # make a copy of a spectrum object
        spectrum_new = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
        return(spectrum_new)

    def pltSpec(self, **kwargs):
        # plot wav vs. flx 
        # **kwargs accepted by plot
        # image is stored as tmp.png
        fig, ax = plt.subplots()
        ax.plot(self.wavelength, self.flux, **kwargs)
        plt.show()
        #fig.savefig("./tmp.png")

    def scaleSpec(self, total_flux=1e4):
        # scale spectrum so that summed flux from each pixel is equal to total_flux
        num_pixels = len(self.wavelength)
        spec_total_flux = np.sum(self.flux)
        flx = self.flux / spec_total_flux * total_flux
        self.flux = flx

    def resampleSpec(self, wav_new):
        # resample a spectrum to a new wavelength grid
        flx_new = np.interp(wav_new, self.wavelength, self.flux)
        self.wavelength = wav_new
        self.flux = flx_new
        return self

    def resampleSpectoSpectrograph(self, pixel_sampling=3.0):
        # resample a spectrum to a wavelength grid that is determed by spectral resolution and pixel sampling rate
        # num_pixel_new = wavelength coverage range / wavelength per resolution element * pixel sampling rate
        num_pixel = len(self.wavelength)
        num_pixel_new = (np.nanmax(self.wavelength) - np.nanmin(self.wavelength)) / (np.nanmedian(self.wavelength) / self.spec_reso) * pixel_sampling
        wav_new = np.linspace(np.nanmin(self.wavelength), np.nanmax(self.wavelength), num = num_pixel_new)
        flx_new = np.interp(wav_new, self.wavelength, self.flux)
        return(Spectrum(wav_new, flx_new, spec_reso=self.spec_reso))

    def dopplerShift(self, rv_shift=0e0):
        #positive number means blue shift and vice versa
        beta = rv_shift / scipy.constants.c
        wav_shifted = self.wavelength * np.sqrt((1 - beta)/(1 + beta))
        flx = np.interp(self.wavelength, wav_shifted, self.flux, left=np.nanmedian(self.flux), right=np.nanmedian(self.flux))
        flx[np.isnan(flx)] = np.nanmedian(flx)
        self.flux = flx
        return self

    def crossCorrelation(self, template, spec_mask=None):
        # positive peak means spectrum is blue shifted with respect to template
        wav = self.wavelength
        flx = self.flux
        wav_temp = template.wavelength
        flx_temp = template.flux
        flx_temp = np.interp(wav, wav_temp, flx_temp)
        flx = flx - np.nanmedian(flx)
        flx_temp = flx_temp - np.nanmedian(flx_temp)
        if spec_mask != None:
            flx[spec_mask] = np.nanmedian(flx)
            flx[spec_mask] = np.nanmedian(flx_temp)
        cc = fp.ifft(fp.fft(flx_temp)*np.conj(fp.fft(flx)))
        ccf = fp.fftshift(cc)
        ccf = ccf - np.median(ccf)
        ccf = ccf.real

        vel_int = np.nanmedian(np.abs(wav[1:-1] - wav[0:-2])) / np.nanmedian(wav) * scipy.constants.c
        nx = len(ccf)
        vel = (np.arange(nx)-(nx-1)/2.0) * vel_int

        return(CrossCorrelationFunction(vel, ccf))

    def spectral_blur(self, rpower=1e5):
        # broaden a spectrum given its spectral resolving power
        wave = self.wavelength
        tran = self.flux

        wmin = wave.min()
        wmax = wave.max()

        nx = wave.size
        x  = np.arange(nx)

        A = wmin
        B = np.log(wmax/wmin)/nx
        wave_constfwhm = A*np.exp(B*x)
        tran_constfwhm = np.interp(wave_constfwhm, wave, tran)
        dwdx_constfwhm = np.diff(wave_constfwhm)
        fwhm_pix = wave_constfwhm[1:]/rpower/dwdx_constfwhm

        fwhm_pix  = fwhm_pix[0]
        sigma_pix = fwhm_pix/2.3548
        kx = np.arange(nx)-(nx-1)/2.
        kernel = 1./(sigma_pix*np.sqrt(2.*np.pi))*np.exp(-kx**2/(2.*sigma_pix**2))

        tran_conv = fft.ifft(fft.fft(tran_constfwhm)*np.conj(fft.fft(kernel)))
        tran_conv = fft.fftshift(tran_conv).real
        tran_oldsampling = np.interp(wave,wave_constfwhm,tran_conv)

        self.wavelength = wave
        self.flux = tran_oldsampling
        
        return self

    def rotational_blur(self, rot_vel=3e4):
        # broaden a spectrum given the rotation of a target
        # kernel is a cosine function with only [-pi/2, pi/2] phase
        # -pi/2 phase corresponds to fwhm_pix for rpower of c / rot_vel
        wave = self.wavelength
        tran = self.flux

        wmin = wave.min()
        wmax = wave.max()

        nx = wave.size
        x  = np.arange(nx)

        A = wmin
        B = np.log(wmax/wmin)/nx
        wave_constfwhm = A*np.exp(B*x)
        tran_constfwhm = np.interp(wave_constfwhm, wave, tran)
        dwdx_constfwhm = np.diff(wave_constfwhm)
        rpower = scipy.constants.c / rot_vel
        fwhm_pix = wave_constfwhm[1:]/rpower/dwdx_constfwhm

        fwhm_pix  = fwhm_pix[0]
        sigma_pix = fwhm_pix/2.3548
        kx = np.arange(nx)-(nx-1)/2.
        kernel = np.cos(2.0 * np.pi * kx / (4.0 * fwhm_pix))
        idx = ((kx < -fwhm_pix) | (kx > fwhm_pix))
        kernel[idx] = 0.0
        kernel = kernel / np.sum(kernel)

        tran_conv = fft.ifft(fft.fft(tran_constfwhm)*np.conj(fft.fft(kernel)))
        tran_conv = fft.fftshift(tran_conv).real
        tran_oldsampling = np.interp(wave,wave_constfwhm,tran_conv)

        self.wavelength = wave
        self.flux = tran_oldsampling

        return self

def readInit(init_file="MdwarfPlanet.init"):
    initDict = {}
    with open(init_file, 'r') as f:
        for line in f:
            key_value = line.split('#')[0]
            key = key_value.split(':')[0].strip(' \t\n\r')
            value = key_value.split(':')[1].strip(' \t\n\r')
            initDict[key] = value
    return(initDict)

def __main__():
    #initDict = readInit(init_file="MdwarfPlanet.init")
    initDict = readInit(init_file="ROS128b_J.init")
    if initDict["template_type"] == "thermal":
        print("Thermal Spectrum")
        target_st = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["st_template_spec_path"], inclination_deg=np.float32(initDict["st_inclination_deg"]), rotation_vel=np.float32(initDict["st_rotation_vel"]), radial_vel=np.float32(initDict["st_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
        target_pl = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["pl_template_spec_path"], inclination_deg=np.float32(initDict["pl_inclination_deg"]), rotation_vel=np.float32(initDict["pl_rotation_vel"]), radial_vel=np.float32(initDict["pl_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
        wav_min, wav_max = np.float32(initDict["wav_min"]), np.float32(initDict["wav_max"])
        spec_st = target_st.spectrum.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["st_rotation_vel"]) * np.sin(np.float32(initDict["st_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_st.rotational_blur(rot_vel=np.float32(initDict["st_rotation_vel"]) * np.sin(np.float32(initDict["st_inclination_deg"])/180.0*np.pi))
        spec_pl = target_pl.spectrum.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_pl.rotational_blur(rot_vel=np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi))
        spec = pyfits.open(initDict["template_path_thermal"])
        spec_mol = Spectrum(spec[1].data["Wavelength"], spec[1].data["Flux"], spec_reso=np.float32(initDict["spec_reso"]))
        spec_mol = spec_mol.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_mol.rotational_blur(rot_vel=np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi))

        spec_pl.resampleSpec(spec_st.wavelength)
        spec_mol.resampleSpec(spec_st.wavelength)

        hdu = pyfits.open(initDict["pl_template_spec_path"])
        hdu[1].data["Wavelength"][0:-1] = np.nan
        hdu[1].data["Flux"][0:-1] = np.nan

        hdu[1].data["Wavelength"][0:len(spec_pl.wavelength)] = spec_pl.wavelength
        hdu[1].data["Flux"][0:len(spec_pl.wavelength)] = spec_pl.flux
        hdu.writeto("template.planet.fits", clobber=True)

        hdu = pyfits.open(initDict["st_template_spec_path"])
        hdu[1].data["Wavelength"][0:-1] = np.nan
        hdu[1].data["Flux"][0:-1] = np.nan

        hdu[1].data["Wavelength"][0:len(spec_st.wavelength)] = spec_st.wavelength
        hdu[1].data["Flux"][0:len(spec_st.wavelength)] = spec_st.flux 
        hdu.writeto("template.star.fits", clobber=True)

        hdu[1].data["Wavelength"][0:len(spec_mol.wavelength)] = spec_mol.wavelength
        hdu[1].data["Flux"][0:len(spec_mol.wavelength)] = spec_mol.flux 
        hdu.writeto("template."+initDict["template_tag"]+".fits", clobber=True)

    elif initDict["template_type"] == "reflection":
        print("Reflection Spectrum")
        target_st = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["st_template_spec_path"], inclination_deg=np.float32(initDict["st_inclination_deg"]), rotation_vel=np.float32(initDict["st_rotation_vel"]), radial_vel=np.float32(initDict["st_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
        target_pl = Target(distance=np.float32(initDict["distance"]), spec_path=None, inclination_deg=np.float32(initDict["pl_inclination_deg"]), rotation_vel=np.float32(initDict["pl_rotation_vel"]), radial_vel=np.float32(initDict["pl_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
        wav_min, wav_max = np.float32(initDict["wav_min"]), np.float32(initDict["wav_max"])
        spec_st = target_st.spectrum.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["st_rotation_vel"]) * np.sin(np.float32(initDict["st_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_st.rotational_blur(rot_vel=np.float32(initDict["st_rotation_vel"]) * np.sin(np.float32(initDict["st_inclination_deg"])/180.0*np.pi))
        template_files = initDict["pl_template_spec_path"].split(",")
        flx = 0.0
        flx_mol = 0.0
        search_dic = {"all":1, "H2O":2, "CO2":3, "N2O":4, "CH4":5, "O2":6, "O3":7}
        field_num = int(search_dic[initDict["template_tag"]])
        for file in template_files:
            spec = astropy.io.ascii.read(file.strip(" \t\n\r"))
            wav = spec.field(0) / 1e3
            flx = spec.field(1) + flx
            flx_mol = spec.field(field_num) + flx_mol
        flx = flx / (len(template_files) + 0.0)
        flx_mol = flx_mol / (len(template_files) + 0.0)

        template_pl = Spectrum(wav, flx, spec_reso=np.float32(initDict["spec_reso"]))
        template_mol = Spectrum(wav, flx_mol, spec_reso=np.float32(initDict["spec_reso"])) 
        spec_pl = template_pl.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_pl.rotational_blur(rot_vel=np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi))
        spec_pl.resampleSpec(spec_st.wavelength)
        spec_mol = template_mol.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_mol.rotational_blur(rot_vel=np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi))
        spec_mol.resampleSpec(spec_st.wavelength)
        rp = np.float32(initDict["pl_radius"])
        sma = np.float32(initDict["pl_st_distance"])
        pl_illumination = np.float32(initDict["pl_illumination"])
        hci_hrs_name = "{0:06.3f}_{1:06.3}".format(wav_min, wav_max)
        obj_tag = initDict["obj_tag"]
        fits_dir = "../fits_dir/"
        hdu = pyfits.open(initDict["st_template_spec_path"])
        hdu[1].data["Wavelength"][0:-1] = np.nan
        hdu[1].data["Flux"][0:-1] = np.nan

        hdu[1].data["Wavelength"][0:len(spec_st.wavelength)] = spec_st.wavelength
        hdu[1].data["Flux"][0:len(spec_st.wavelength)] = spec_st.flux * spec_pl.flux * pl_illumination * (rp / sma)**2 / 1.0
        hdu.writeto(fits_dir+obj_tag+"."+hci_hrs_name+"."+"template.planet.fits", clobber=True)
        hdu.writeto("template.planet.fits", clobber=True)

        hdu[1].data["Wavelength"][0:len(spec_st.wavelength)] = spec_st.wavelength
        hdu[1].data["Flux"][0:len(spec_st.wavelength)] = spec_st.flux 
        hdu.writeto(fits_dir+obj_tag+"."+hci_hrs_name+"."+"template.star.fits", clobber=True)
        hdu.writeto("template.star.fits", clobber=True)

        hdu[1].data["Wavelength"][0:len(spec_mol.wavelength)] = spec_mol.wavelength
        hdu[1].data["Flux"][0:len(spec_mol.wavelength)] = spec_mol.flux 
        hdu.writeto(fits_dir+obj_tag+"."+hci_hrs_name+"."+"template."+initDict["template_tag"]+".fits", clobber=True)
        hdu.writeto("template."+initDict["template_tag"]+".fits", clobber=True)
    elif initDict["template_type"] == "reflection_giants":
        print("Reflection Spectrum For Giants")
        target_st = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["st_template_spec_path"], inclination_deg=np.float32(initDict["st_inclination_deg"]), rotation_vel=np.float32(initDict["st_rotation_vel"]), radial_vel=np.float32(initDict["st_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
        target_pl = Target(distance=np.float32(initDict["distance"]), spec_path=None, inclination_deg=np.float32(initDict["pl_inclination_deg"]), rotation_vel=np.float32(initDict["pl_rotation_vel"]), radial_vel=np.float32(initDict["pl_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
        wav_min, wav_max = np.float32(initDict["wav_min"]), np.float32(initDict["wav_max"])
        spec_st = target_st.spectrum.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["st_rotation_vel"]) * np.sin(np.float32(initDict["st_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_st.rotational_blur(rot_vel=np.float32(initDict["st_rotation_vel"]) * np.sin(np.float32(initDict["st_inclination_deg"])/180.0*np.pi))
        template_files = initDict["pl_template_spec_path"].split(",")
        flx = 0.0
        flx_mol = 0.0
        search_dic = {"all":1, "CH4":2, "NH3":3, "H2O":4}
        field_num = int(search_dic[initDict["template_tag"]])
        for file in template_files:
            spec = astropy.io.ascii.read(file.strip(" \t\n\r"))
            wav = spec.field(0) / 1e3
            flx = spec.field(1) + flx
            flx_mol = spec.field(field_num) + flx_mol
        flx = flx / (len(template_files) + 0.0)
        flx_mol = flx_mol / (len(template_files) + 0.0)

        template_pl = Spectrum(wav, flx, spec_reso=np.float32(initDict["spec_reso"]))
        template_mol = Spectrum(wav, flx_mol, spec_reso=np.float32(initDict["spec_reso"])) 
        spec_pl = template_pl.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_pl.rotational_blur(rot_vel=np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi))
        spec_pl.resampleSpec(spec_st.wavelength)
        spec_mol = template_mol.getSpecChunk(wav_min, wav_max)
        if np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi) > 2e3:
            spec_mol.rotational_blur(rot_vel=np.float32(initDict["pl_rotation_vel"]) * np.sin(np.float32(initDict["pl_inclination_deg"])/180.0*np.pi))
        spec_mol.resampleSpec(spec_st.wavelength)
        rp = np.float32(initDict["pl_radius"])
        sma = np.float32(initDict["pl_st_distance"])
        pl_illumination = np.float32(initDict["pl_illumination"])
        hci_hrs_name = "{0:06.3f}_{1:06.3}".format(wav_min, wav_max)
        obj_tag = initDict["obj_tag"]
        fits_dir = "./fits_dir/"
        hdu = pyfits.open(initDict["st_template_spec_path"])
        hdu[1].data["Wavelength"][0:-1] = np.nan
        hdu[1].data["Flux"][0:-1] = np.nan

        hdu[1].data["Wavelength"][0:len(spec_st.wavelength)] = spec_st.wavelength
        hdu[1].data["Flux"][0:len(spec_st.wavelength)] = spec_st.flux * spec_pl.flux * pl_illumination * (rp / sma)**2 / 1.0
        hdu.writeto(fits_dir+obj_tag+"."+hci_hrs_name+"."+"template.planet.fits", clobber=True)
        hdu.writeto("template.planet.fits", clobber=True)

        hdu[1].data["Wavelength"][0:len(spec_st.wavelength)] = spec_st.wavelength
        hdu[1].data["Flux"][0:len(spec_st.wavelength)] = spec_st.flux 
        hdu.writeto(fits_dir+obj_tag+"."+hci_hrs_name+"."+"template.star.fits", clobber=True)
        hdu.writeto("template.star.fits", clobber=True)

        hdu[1].data["Wavelength"][0:len(spec_mol.wavelength)] = spec_mol.wavelength
        hdu[1].data["Flux"][0:len(spec_mol.wavelength)] = spec_mol.flux 
        hdu.writeto(fits_dir+obj_tag+"."+hci_hrs_name+"."+"template."+initDict["template_tag"]+".fits", clobber=True)
    else:
        print("template_type in init file has to be thermal or reflection")

__main__()



