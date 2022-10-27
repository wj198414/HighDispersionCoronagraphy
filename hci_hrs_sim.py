import numpy as np
import astropy.io.fits as pyfits

from target import Target
from thermtarget import ThermTarget
from zoditarget import ZodiTarget
from instrument import Instrument
from spectrum import Spectrum
from atmosphere import Atmosphere
from hcihrsobservation import HCI_HRS_Observation
from hcihrsreduction import HCI_HRS_Reduction

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
    #initDict = readInit(init_file="SunEarth_4m.init")
    initDict = readInit(init_file="ROS128b_J.init")
    wav_min, wav_max, t_exp = np.float32(initDict["wav_min"]), np.float32(initDict["wav_max"]), np.float32(initDict["t_exp"])
    target_pl = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["pl_spec_path"], inclination_deg=np.float32(initDict["pl_inclination_deg"]), rotation_vel=np.float32(initDict["pl_rotation_vel"]), radial_vel=np.float32(initDict["pl_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
    target_st = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["st_spec_path"], inclination_deg=np.float32(initDict["st_inclination_deg"]), rotation_vel=np.float32(initDict["st_rotation_vel"]), radial_vel=np.float32(initDict["st_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
    wav_med = (wav_min + wav_max) / 2.0
    if initDict["spec_tran_path"] != "None":
        atmosphere = Atmosphere(spec_tran_path=initDict["spec_tran_path"], spec_radi_path=initDict["spec_radi_path"])
    else:
        atmosphere = None
    instrument = Instrument(wav_med, telescope_size=np.float32(initDict["telescope_size"]), pl_st_contrast=np.float32(initDict["pl_st_contrast"]), spec_reso=np.float32(initDict["spec_reso"]), read_noise=np.float32(initDict["read_noise"]), dark_current=np.float32(initDict["dark_current"]), fiber_size=np.float32(initDict["fiber_size"]), pixel_sampling=np.float32(initDict["pixel_sampling"]), throughput=np.float32(initDict["throughput"]), wfc_residual=np.float32(initDict["wfc_residual"]), num_surfaces=np.float32(initDict["num_surfaces"]), temperature=np.float32(initDict["temperature"]))
    thermal_background = ThermTarget(instrument, spec_reso=np.float32(initDict["spec_reso"]))
    zodi = ZodiTarget(instrument, distance=np.float32(initDict["distance"]), spec_path=initDict["zodi_spec_path"], exozodi_level=np.float32(initDict["exozodi_level"]), spec_reso=np.float32(initDict["spec_reso"]))
    hci_hrs = HCI_HRS_Observation(wav_min, wav_max, t_exp, target_pl, target_st, instrument, thermal_background, zodi, atmosphere=atmosphere)
    print("Star flux: {0} \nLeaked star flux: {1}\nPlanet flux: {2}\nPlanet flux per pixel: {3}\nThermal background flux:  {4}\nThermal background flux per pixel:  {5}\nSky flux: {6}\nSky flux per pixel: {7}\nSky transmission: {8}\nTotal pixel number: {9}\n".format(hci_hrs.star_total_flux, hci_hrs.star_total_flux * instrument.pl_st_contrast, hci_hrs.planet_total_flux, hci_hrs.planet_total_flux / (len(hci_hrs.obs_spec_resample.flux) + 0.0), hci_hrs.therm_total_flux, hci_hrs.therm_total_flux / (len(hci_hrs.obs_therm_resample.flux) + 0.0), hci_hrs.sky_total_flux, hci_hrs.sky_total_flux / (len(hci_hrs.obs_therm_resample.flux) + 0.0), hci_hrs.sky_transmission, len(hci_hrs.obs_therm_resample.flux)))
    spec = pyfits.open(initDict["template_path"])
    template = Spectrum(spec[1].data["Wavelength"], spec[1].data["Flux"], spec_reso=np.float32(initDict["spec_reso"]))
    if initDict["spec_tran_path"] != "None":
        hci_hrs_red = HCI_HRS_Reduction(hci_hrs, template, save_flag=False, obj_tag=initDict["obj_tag"], template_tag=initDict["template_tag"],speckle_flag=False)
    else:
        hci_hrs_red = HCI_HRS_Reduction(hci_hrs, template, save_flag=False, obj_tag=initDict["obj_tag"], template_tag=initDict["template_tag"],speckle_flag=True)
__main__()








