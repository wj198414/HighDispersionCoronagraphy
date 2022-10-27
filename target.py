import astropy.io.fits as pyfits
from spectrum import Spectrum

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