from spectrum import Spectrum

#This class carries the thermal background of the instrument.  The relevant portions of the interface are the same as for Target().
#Instances of ThermTarget() may be called by Spectrum() with no known issues.

class ThermTarget():
   def __init__(self, instrument, spec_reso=1e5):
        self.spec_reso = spec_reso
        self.spec_data = instrument.thermal_background.data
        self.wavelength = self.spec_data["Wavelength"]
        self.flux = self.spec_data["Flux"]
        self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
        self.spec_header = instrument.thermal_background.header