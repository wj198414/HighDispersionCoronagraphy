import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import pickle
from scipy import signal
import numpy.fft as fft
from scipy import interpolate
import scipy.interpolate
import scipy.fftpack as fp
from crosscorrelationfunction import CrossCorrelationFunction

class Spectrum():
    def __init__(self, wavelength, flux, spec_reso=1e5, norm_flag=False):
        self.wavelength = wavelength
        self.flux = flux
        self.spec_reso = spec_reso
        self.norm_flag = norm_flag
        self.noise = None

    def addNoise(self, noise):
        if self.noise == None:
            self.noise = noise
        else:
            print("Warning: spectrum noise already added")

    def writeSpec(self, file_name="tmp.dat",flag_python2=False,flag_append=False):
        if not flag_append:
            write_code = "wb"
        else:
            write_code = "ab"
        with open(file_name, write_code) as f:
            for i in np.arange(len(self.wavelength)):
                if flag_python2 == True:
                    if np.size(self.noise) == 1:
                        f.write("{0:20.8f}{1:20.8e}\n".format(self.wavelength[i], self.flux[i]))
                    else:
                        f.write("{0:20.8f}{1:20.8e}{2:20.8e}\n".format(self.wavelength[i], self.flux[i], self.noise[i]))
                else:
                    if np.size(self.noise) == 1:
                        f.write(bytes("{0:20.8f}{1:20.8e}\n".format(self.wavelength[i], self.flux[i]), 'UTF-8'))
                    else:
                        f.write(bytes("{0:20.8f}{1:20.8e}{2:20.8e}\n".format(self.wavelength[i], self.flux[i], self.noise[i]), 'UTF-8'))

    def getSpecNorm(self, num_chunks=10, poly_order=2, emission=False):
        wav = self.wavelength
        flx = self.flux
        num_pixels = len(wav)
        pix_chunk = int(np.floor(num_pixels / num_chunks))
        wav_chunk = np.zeros((num_chunks,))
        flx_chunk = np.zeros((num_chunks,))
        for i in np.arange(num_chunks):
            wav_chunk[i] = np.nanmedian(wav[i*pix_chunk:(i+1)*pix_chunk])
            if not emission:
                flx_chunk[i] = np.nanmax(flx[i*pix_chunk:(i+1)*pix_chunk])
            else:
                flx_chunk[i] = np.nanmin(flx[i*pix_chunk:(i+1)*pix_chunk]) 
        coeff = np.polyfit(wav_chunk, flx_chunk, poly_order)
        p = np.poly1d(coeff)
        flx_norm = p(wav)
        return(flx_norm)

    def combineSpec(self, spec):
        spec_new = self.copy()
        idx = np.argsort(np.hstack((self.wavelength, spec.wavelength)))
        spec_new.wavelength = np.hstack((self.wavelength, spec.wavelength))[idx]
        spec_new.flux = np.hstack((self.flux, spec.flux))[idx]
        if spec_new.noise != None:
            print("Combining spectrum may cause trouble for attribute Noise")
        return(spec_new)

    def getChunk(self, wav_min, wav_max):
        spec_new = self.copy()
        idx = np.where((self.wavelength <= wav_max) & (self.wavelength > wav_min))
        spec_new.wavelength = self.wavelength[idx]
        spec_new.flux = self.flux[idx]
        if spec_new.noise != None:
            spec_new.noise = self.noise[idx]
        return(spec_new)

    def saveSpec(self, file_name="tmp.pkl"):
        with open(file_name, "wb") as handle:
            pickle.dump(self, handle)

    def simSpeckleNoise(self, wav_min, wav_max, wav_int, wav_new):
        wav = np.arange(wav_min, wav_max + wav_int / 2.0, wav_int)
        wav_arr = np.array([])
        flx_arr = np.array([])
        for i, wav_tmp in enumerate(wav[:-1]):
            wav_mid = np.random.normal(wav_tmp + wav_int * 0.5, wav_int / 10.0, size=(1,))
            flx_mid = np.random.random(size=(1,)) * 0.5 + 0.5 
            flx_tmp = np.random.random(size=(1,)) * 0.5 + 1.0
            wav_arr = np.hstack((wav_arr, [wav_tmp, wav_mid[0]]))
            flx_arr = np.hstack((flx_arr, [flx_tmp, flx_mid[0]]))
        wav_arr = np.hstack((wav_arr, [wav[-1]]))
        flx_arr = np.hstack((flx_arr, [np.random.random(1)[0] + 1.0]))
        f = scipy.interpolate.interp1d(wav_arr, flx_arr, kind="cubic")
        flx_new = f(wav_new)
        idx = np.where(flx_new < 0.1)
        flx_new[idx] = 0.1
        return(flx_new)

    def generateNoisySpec(self, speckle_noise=False, star_flux=1.0):
        spec = self.copy()
        flx = self.flux
        flx_new = np.zeros(np.shape(flx))
        num = len(flx)
        i = 0
        if hasattr(flx[0], 'value'):
            while i < num:
                #flx_new[i] = np.max([np.random.poisson(np.round(flx[i]), 1)+0.0, np.random.normal(flx[i], self.noise[i], 1)])
                flx_new[i] = np.random.normal(flx[i].value, self.noise[i].value, 1)
                i = i + 1
        else:
            while i < num:
                #flx_new[i] = np.max([np.random.poisson(np.round(flx[i]), 1)+0.0, np.random.normal(flx[i], self.noise[i], 1)])
                flx_new[i] = np.random.normal(flx[i], self.noise[i], 1)
                i = i + 1        
        spec.flux = flx_new

        if speckle_noise:
            flx_speckle = self.simSpeckleNoise(np.min(spec.wavelength), np.max(spec.wavelength), 0.1, spec.wavelength)
            #spec.flux = spec.flux * flx_speckle
            spec.flux = spec.flux + star_flux * flx_speckle

        return(spec)

    def evenSampling(self):
        wav = self.wavelength
        flx = self.flux
        wav_int = np.median(np.abs(np.diff(wav)))
        wav_min = np.min(wav)
        wav_max = np.max(wav)
        wav_new = np.arange(wav_min, wav_max, wav_int)
        flx_new = np.interp(wav_new, wav, flx)
        self.wavelength = wav_new
        self.flux = flx_new

        return(self)

    def applyHighPassFilter(self, order = 5, cutoff = 100.0, pass_type="high", fourier_flag=True, plot_flag=False):
        if not fourier_flag:
            # cutoff is number of sampling per 1 micron, so 100 means 0.01 micron resolution, about R = 100 at 1 micron
            x = self.wavelength
            y = self.flux
            n = self.noise
            fs = 1.0 / np.median(x[1:-1] - x[0:-2])
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            #print("normal_cutoff = ", normal_cutoff)
            b, a = scipy.signal.butter(order, normal_cutoff, btype=pass_type, analog=False)
            yy = scipy.signal.filtfilt(b, a, y)
            spec = Spectrum(x, yy, spec_reso=self.spec_reso)
            if n is not None:
                spec.addNoise(n)
            return(spec)
        else:
            x = self.wavelength
            y = self.flux
            n = self.noise
            fy = fp.fft(y)
            fy = fp.fftshift(fy)
            delta_x = np.median(np.abs(np.diff(x)))
            N = len(x)
            fx = np.linspace(-0.5 / delta_x, 0.5 / delta_x - 0.5 / delta_x / N, num=N)
            if plot_flag:
                plt.plot(fx, np.abs(fy))
                plt.yscale("log")
            delta_x = np.median(np.abs(np.diff(x)))
            if pass_type == "high":
                filter_envelope = 1.0 - 1.0 * np.exp(-1.0 * fx**2 / (2.0 * cutoff**2))
            else:
                filter_envelope = 1.0 * np.exp(-1.0 * fx**2 / (2.0 * cutoff**2))
            if plot_flag:
                plt.plot(fx, filter_envelope * np.max(fy))
                plt.show()
            ffy = fp.ifft(fp.ifftshift(fy* filter_envelope))
            if plot_flag:
                plt.plot(x,y)
                plt.plot(x, ffy)
                plt.show()
            spec = Spectrum(x, ffy, spec_reso=self.spec_reso)
            if n is not None:
                spec.addNoise(n)
            return(spec)

    def copy(self):
        # make a copy of a spectrum object
        spectrum_new = Spectrum(self.wavelength.copy(), self.flux.copy(), spec_reso=self.spec_reso+0.0)
        if np.size(self.noise) != 1:
            spectrum_new.noise = self.noise.copy()
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
        if total_flux != 0.0:
            flx = self.flux / spec_total_flux * total_flux
            self.flux = flx
        else:
            self.flux = np.zeros(np.shape(self.flux)) + 1e-99

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
        wav_new = np.linspace(np.nanmin(self.wavelength), np.nanmax(self.wavelength), num = int(np.round(num_pixel_new)))
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

    def crossCorrelation(self, template, spec_mask=None, long_array=False, speed_flag=False, flag_plot=False, flag_crop=True):
        # positive peak means spectrum is blue shifted with respect to template
        # do not recommend long_array option. It does not produce the same SNR as the non-long_array option. 
        if not long_array:
            wav = self.wavelength
            flx = self.flux
            wav_temp = template.wavelength
            flx_temp = template.flux
            flx_temp = np.interp(wav, wav_temp, flx_temp)
            flx = flx - np.nanmedian(flx)
            flx_temp = flx_temp - np.nanmedian(flx_temp)
            if spec_mask != None:
                flx[spec_mask] = np.nanmedian(flx)
                flx_temp[spec_mask] = np.nanmedian(flx_temp)
            if flag_crop:
                num_crop = int(len(wav) * 0.05)
                wav = wav[num_crop:-num_crop]
                flx = flx[num_crop:-num_crop]
                flx_temp = flx_temp[num_crop:-num_crop]
            if speed_flag:
                num_pixels = len(wav)
                power_2 = np.ceil(np.log10(num_pixels + 0.0) / np.log10(2.0))
                num_pixels_new = 2.0**power_2
                wav_new = np.linspace(np.min(wav), np.max(wav), num_pixels_new)
                flx_new = np.interp(wav_new, wav, flx)
                flx_temp_new = np.interp(wav_new, wav, flx_temp)
                flx_temp = flx_temp_new
                flx = flx_new
                wav = wav_new

            if flag_plot:
                plt.plot(wav, flx / (np.max(flx) - np.min(flx)))
                plt.plot(wav, flx_temp / (np.max(flx_temp) - np.min(flx_temp)))
                plt.show()

            cc = fp.ifft(fp.fft(flx_temp)*np.conj(fp.fft(flx)))
            ccf = fp.fftshift(cc)
            ccf = ccf - np.median(ccf)
            ccf = ccf.real 
    
            vel_int = np.nanmedian(np.abs(wav[1:-1] - wav[0:-2])) / np.nanmedian(wav) * scipy.constants.c
            nx = len(ccf)
            ccf = ccf / (nx + 0.0)
            vel = (np.arange(nx)-(nx-1)/2.0) * vel_int
        else:
            num_chunks = 4
            num_pixels = len(self.wavelength) 
            pix_chunk = int(np.floor(num_pixels / (num_chunks + 0.0)))
            for i in np.arange(num_chunks):
                spec_tmp = Spectrum(self.wavelength[i*pix_chunk:(i+1)*pix_chunk], self.flux[i*pix_chunk:(i+1)*pix_chunk])
                template_tmp = Spectrum(template.wavelength[i*pix_chunk:(i+1)*pix_chunk], template.flux[i*pix_chunk:(i+1)*pix_chunk])
                ccf_tmp = spec_tmp.crossCorrelation(template_tmp)
                if i == 0:
                    ccf_total = ccf_tmp
                else:
                    ccf_total = CrossCorrelationFunction(ccf_tmp.vel, ccf_tmp.ccf + ccf_total.ccf)
                vel = ccf_total.vel
                ccf = ccf_total.ccf             

        return(CrossCorrelationFunction(vel, ccf))

    def spectral_blur(self, rpower=1e5, quick_blur=False):
        # broaden a spectrum given its spectral resolving power
        if not quick_blur:
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
        else:
            pixel_to_sum = int(102400.0 / rpower)
            if pixel_to_sum >= 1.5:
                num_pixels = len(self.wavelength)
                num_pixels_new = int(np.floor((num_pixels + 0.0) / pixel_to_sum))
                wav = np.zeros((num_pixels_new,))
                flx = np.zeros((num_pixels_new,))
                for i in np.arange(num_pixels_new):
                    wav[i] = np.mean(self.wavelength[i*pixel_to_sum:(i+1)*pixel_to_sum])       
                    flx[i] = np.mean(self.flux[i*pixel_to_sum:(i+1)*pixel_to_sum])
                self.wavelength = wav
                self.flux = flx
        self.spec_reso = rpower
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
