import numpy as np
import astropy.io.ascii as ascii
import pickle


class Atmosphere():
    def __init__(self, spec_tran_path=None, spec_radi_path=None, radial_vel=1e1):
        self.spec_tran_path = spec_tran_path
        self.spec_radi_path = spec_radi_path
        self.radial_vel = radial_vel
        if self.spec_tran_path != None:
            # with open(spec_tran_path, "rb") as handle:
            #     [self.spec_tran_wav, self.spec_tran_flx] = pickle.load(handle)
            tab = np.loadtxt(spec_tran_path)
            self.spec_tran_wav, self.spec_tran_flx = tab[:, 0], tab[:, 1]
            # fill in the hole between 5.6 and 7.0 micron
            self.spec_tran_wav = np.hstack([np.arange(5.6, 7.0, 1e-5), self.spec_tran_wav]) # to avoid missing information in optical between 5.6 and 7.0 micron
            self.spec_tran_flx = np.hstack([np.zeros(np.shape(np.arange(5.6, 7.0, 1e-5))) + 1e-99, self.spec_tran_flx])
            self.spec_tran_wav = np.hstack([np.arange(0.3, 0.9, 1e-5), self.spec_tran_wav]) # to avoid missing information in optical below 0.9 micron
            self.spec_tran_flx = np.hstack([np.zeros(np.shape(np.arange(0.3, 0.9, 1e-5))) + 1.0, self.spec_tran_flx])
            idx = np.argsort(self.spec_tran_wav)
            self.spec_tran_wav = self.spec_tran_wav[idx]
            self.spec_tran_flx = self.spec_tran_flx[idx] 
        else:
            self.spec_tran_wav = np.arange(0.1, 5.0, 1e-5)
            self.spec_tran_flx = np.zeros(np.shape(self.spec_tran_wav)) + 1.0
        if self.spec_radi_path != None:
            self.spec_radi_data = ascii.read(spec_radi_path)
            self.spec_radi_wav = self.spec_radi_data["col1"][:] # in nm
            self.spec_radi_wav = self.spec_radi_wav / 1e3 # now in micron
            self.spec_radi_flx = self.spec_radi_data["col2"][:] # in ph/s/arcsec**2/nm/m**2
            self.spec_radi_flx = self.spec_radi_flx * 1e3 # now in ph/s/arcsec**2/micron/m**2
            self.spec_radi_wav = np.hstack([np.arange(0.1, 0.9, 1e-5), self.spec_radi_wav]) # to avoid missing information in optical below 0.9 micron
            self.spec_radi_flx = np.hstack([np.zeros(np.shape(np.arange(0.1, 0.9, 1e-5))) + 1e-99, self.spec_radi_flx])
            # fill in the hole between 5.6 and 7.0 micron
            self.spec_radi_wav = np.hstack([np.arange(5.6, 7.0, 1e-5), self.spec_radi_wav]) # to avoid missing information in optical below 0.9 micron
            self.spec_radi_flx = np.hstack([np.zeros(np.shape(np.arange(5.6, 7.0, 1e-5))) + 2093984.0, self.spec_radi_flx])
            idx = np.argsort(self.spec_radi_wav)
            self.spec_radi_wav = self.spec_radi_wav[idx]
            self.spec_radi_flx = self.spec_radi_flx[idx]
            
        else:
            self.spec_radi_wav = np.arange(0.1, 5.0, 1e-5)
            self.spec_radi_flx = np.zeros(np.shape(self.spec_radi_wav)) + 1e-99

    def getTotalSkyFlux(self, wav_min, wav_max, tel_size=10.0, multiple_lambda_D=1.0, t_exp=1e3, eta_ins=0.1):
        # get total flux of sky emission
        # flx in ph/s/arcsec^2/nm/m^2
        idx = ((self.spec_radi_wav < wav_max) & (self.spec_radi_wav > wav_min))
        wav = self.spec_radi_wav[idx]
        flx = self.spec_radi_flx[idx]
        wav_int = np.abs(wav[1:-1] - wav[0:-2])
        fiber_size = np.nanmedian(wav) * 1e-6 / tel_size / np.pi * 180.0 * 3600.0
        fiber_size = fiber_size * multiple_lambda_D # multiple times lambda / D
        flx_skybg_total = np.sum(flx[0:-2] * t_exp * fiber_size **2 * wav_int * np.pi * (tel_size / 2.0)**2) * eta_ins
        
        return(flx_skybg_total)
