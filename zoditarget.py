import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import os
from spectrum import Spectrum
from scipy import interpolate

#This class carries the exozodiacal background for the observation.  The relevant portions of the interface are the same as for Target().

class ZodiTarget():
    def __init__(self, instrument, distance=10.0, spec_path=None, exozodi_level=1.0, spec_reso=1e5):
        self.distance = distance
        self.instrument = instrument
        self.exozodi_level = exozodi_level  #Level of zodi relative to Solar System
        self.spec_path = spec_path
        self.spec_reso = spec_reso
        self.spec_data = self.sumZodiFlux().data
        self.wavelength = self.spec_data["Wavelength"]
        self.flux = self.spec_data["Flux"]
        self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)

        #This function converts the simulation data into the format that the rest of the program expects.
        #Because the simulation is on a large physical scale with relatively fine resolution, each pixel on
        #the instrument corresponds to many pixels in the simulation data.  Therefore, it has to be summed
        #before being used.

    def sumZodiFlux(self):
    
        #Location of the exoEarth in the exozodi simulation.  Using one side in particular
        #because the simulation includes the Solar System planets, and we don't want to
        #double count the Earth.  We will unfortunately however include light from Mars,
        #but c'est la vie.  Format is y, x.
        planetcen = [1666, 1633]

        wavel = []
        zodi_per_pix = []

        #haystacks_tmp = pyfits.open(self.spec_path+"/"+os.listdir(self.spec_path)[0])
        
        #yy, xx = np.mgrid[:haystacks_tmp[1].data.shape[0], :haystacks_tmp[1].data.shape[1]]
        #circle = (xx - planetcen[1])**2. + (yy - planetcen[0])**2.
        
        #haystacks_tmp.close()
        
        for zodicube in sorted(os.listdir(self.spec_path)):
            
            haystacks = pyfits.open(self.spec_path+"/"+zodicube)

            N_EXT = haystacks[0].header['N_EXT']
            wavel_temp = haystacks[N_EXT+1].data
            wavel.append(wavel_temp)
            pixel_scale = haystacks[0].header['PIXSCALE'] * u.au  #Pixel scale of the simulation, in AU/pix
            scale_distance = haystacks[0].header["DIST"] #This is needed later on outside the loop
        
            #Make a circular aperture corresponding to a lambda/D sized planet PSF
            pix_rad = np.array((wavel_temp*u.um*self.distance*u.pc/(self.instrument.telescope_size*u.m*pixel_scale)).decompose()/2)
            yy, xx = np.mgrid[:haystacks[1].data.shape[0], :haystacks[1].data.shape[1]]
            circle = (xx - planetcen[1])**2. + (yy - planetcen[0])**2.
            #print pix_rad
            #Sum up the zodi inside the planet PSF at each wavelength
            for i in range(N_EXT):
                zodi_per_pix.append(np.sum(haystacks[i+1].data[circle < pix_rad[i]**2.]))

        #Scale the exozodi brightness to the proper distance and total exozodi level; 
        #the Haystacks model is the Solar System at 10 pc
        zodi_per_pix = np.array(zodi_per_pix) * (scale_distance/self.distance)**2 * self.exozodi_level
        #Ensure that the data are ordered by wavelength
        wavel = np.array(wavel).ravel()                                               
        wavel_order = np.argsort(wavel)
        wavel = wavel[wavel_order]
        zodi_per_pix = zodi_per_pix[wavel_order]
        
        wavelcol = pyfits.Column(name="Wavelength", array=wavel, format="E", unit="um")
        fluxcol = pyfits.Column(name="Flux", array=zodi_per_pix, format="E", unit="Jy")
        zodihdu = pyfits.BinTableHDU.from_columns([wavelcol, fluxcol])

        return zodihdu