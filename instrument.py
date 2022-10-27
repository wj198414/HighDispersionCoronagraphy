import numpy as np
import astropy.io.fits as pyfits
import scipy.constants

class Instrument():
    def __init__(self, wav_med, telescope_size=10.0, pl_st_contrast=1e-10, spec_reso=1e5, read_noise=2.0, dark_current=1e-3, fiber_size=1.0, pixel_sampling=3.0, throughput=0.1, wfc_residual=200.0, num_surfaces=10, temperature=280):   
        self.telescope_size = telescope_size
        self.pl_st_contrast = pl_st_contrast
        self.spec_reso = spec_reso
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.fiber_size = fiber_size
        self.pixel_sampling = pixel_sampling
        self.wfc_residual = wfc_residual # in nm
        self.wav_med = wav_med # in micron
        self.strehl = self.calStrehl()
        self.relative_strehl = self.calRelativeStrehl()
        self.throughput = throughput * self.relative_strehl
        self.num_surfaces = num_surfaces #Number of warm optical surfaces in the instrument for calculating thermal background
        self.temperature = temperature #Temperature of the instrument, in K
        self.thermal_background = self.calcThermBackground()

    def calStrehl(self):
        strehl = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / self.wav_med))**2) 
        return(strehl)

    def calRelativeStrehl(self):
        strehl_K = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / 2.0))**2)
        strehl = self.calStrehl()
        return(strehl / strehl_K)

    #Calculate the thermal background of the instrument.  Diffraction-limited imaging is assumed (A*Omega ~ lambda^2).
    #Returns a FITS binary table.  This is done to match the format expected by Target() and Spectrum().
    #Columns of the table are 0:  wavelength in um.  1:  Thermal flux in one diffraction-limited pixel in W/um.

    def calcThermBackground(self):

        #Complex refractive index data (n+i*k) for gold.  
        #Columns are:  0:  wavel in um; 1: n; 2: k.
        gold_refrac_data = np.genfromtxt("goldrefrac.txt", dtype=float)

        wavelidx = 0
        nidx = 1
        kidx = 2

        #Assume normal incidence for all light.  Not necessarily true, but good enough for an estimate.  
        #Any deviation from normal incidence only lowers the background, in any case.
        gold_norm_reflec = np.abs((1.-(gold_refrac_data[:,nidx]+1j*gold_refrac_data[:,kidx]))/\
                          (1.+gold_refrac_data[:,nidx]+1j*gold_refrac_data[:,kidx]))**2.

        mask_avg_reflec = 0.5 #Average reflectivity of the coronagraph mask
        emiss = 1. - mask_avg_reflec*(gold_norm_reflec**self.num_surfaces) #Effective emissivity of the instrument is 1 - Re1*Re2*Re3*...

        #Thermal flux in one diffraction-limited pixel in W/um.  Other samplings can be made by changing the pre-factor, presumably.
      
        therm_flux = 2.*scipy.constants.c**2.*scipy.constants.h*emiss/((gold_refrac_data[:,wavelidx])**3.*(1.e-6)**2.*(np.exp(scipy.constants.h*scipy.constants.c/(gold_refrac_data[:,wavelidx]*1.e-6*scipy.constants.k*self.temperature))-1))
      
        wavelcol = pyfits.Column(name="Wavelength", array=gold_refrac_data[:,wavelidx], format="E", unit="um")
        if self.num_surfaces == 0:
            print("num_surfaces = ", self.num_surfaces)
            fluxcol = pyfits.Column(name="Flux", array=np.zeros(np.shape(therm_flux))+1e-99, format="E", unit="W / um")
        else:
            fluxcol = pyfits.Column(name="Flux", array=therm_flux, format="E", unit="W / um")
        thermhdu = pyfits.BinTableHDU.from_columns([wavelcol, fluxcol])

        return(thermhdu)
