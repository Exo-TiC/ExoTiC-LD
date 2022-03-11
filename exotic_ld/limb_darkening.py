"""
This is a self-standing module that calculates limb darkening parameters for either 1D or 3D stellar models. It
returns the parameters for 4-parameter, 3-parameter, quadratic and linear limb darkening models.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import readsav
from scipy.interpolate import interp1d, splev, splrep
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter


def limb_dark_fit(mode, wsdata, M_H, Teff, logg, dirsen, ld_model='1D'):
    """
    Calculates stellar limb-darkening coefficients for a given wavelength bin.

    Modes Currently Supported:
    Spectroscopic:
    HST STIS G750L, G750M, G430L gratings
    HST WFC3 UVIS/G280+1, UVIS/G280-1, IR/G102, IR/G141 grisms
    JWST NIRSpec Prism, G395H, G395M, G235H, G235M, G140H-f100, G140M-f100, G140H-f070, G140M-f070
    JWST NIRISS SOSSo1, SOSSo2
    JWST NIRCam F322W2, F444
    JWST MIRI LRS
    Photometric:
    TESS
    Spitzer IRAC Ch1 (3.6 microns), Ch2 (4.5 microns)

    Procedure from Sing et al. (2010, A&A, 510, A21).
    Uses 3D limb darkening from Magic et al. (2015, A&A, 573, 90).
    Uses photon FLUX Sum over (lambda*dlamba).
    :param mode: string; mode to use Spectroscopic: ('STIS_G430L','STIS_G750L', 'WFC3_G280p1', 'WFC3_G280n1', 'WFC3_G102', 'WFC3_G141', 'NIRSpec_Prism', 'NIRSpec_G395H', 'NIRSpec_G395M', 'NIRSpec_G235H', 'NIRSpec_G235M', 'NIRSpec_G140Hf100', 'NIRSpec_G140Mf100', 'NIRSpec_G140Hf070', 'NIRSpec_G140Mf070', 'NIRISS_SOSSo1', 'NIRISS_SOSSo2', 'NIRCam_F322W2', 'NIRCam_F444', 'MIRI_LRS'), Photometric: ('IRAC_Ch1', 'IRAC_Ch2', 'TESS')
    :param wsdata: array; data wavelength solution for range required
    :param M_H: float; stellar metallicity
    :param Teff: float; stellar effective temperature (K)
    :param logg: float; stellar gravity
    :param dirsen: string; path to main limb darkening directory downloaded from Zenodo V2.1
    :param ld_model: string; '1D' or '3D', makes choice between limb darkening models; default is 1D
    :return: uLD: float; linear limb darkening coefficient
    aLD, bLD: float; quadratic limb darkening coefficients
    cp1, cp2, cp3, cp4: float; three-parameter limb darkening coefficients
    c1, c2, c3, c4: float; non-linear limb-darkening coefficients
    """

    print('You are using the', str(ld_model), 'limb darkening models.')

    if ld_model == '1D':

        direc = os.path.join(dirsen, 'Kurucz')

        print('Current Directories Entered:')
        print('  ' + dirsen)
        print('  ' + direc)

        # Select metallicity
        M_H_Grid = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
        M_H_Grid_load = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 21, 22, 23, 24])
        optM = (abs(M_H - M_H_Grid)).argmin()
        MH_ind = M_H_Grid_load[optM]

        # Determine which model is to be used, by using the input metallicity M_H to figure out the file name we need
        file_list = 'kuruczlist.sav'
        sav1 = readsav(os.path.join(direc, file_list))
        model = bytes.decode(sav1['li'][MH_ind])  # Convert object of type "byte" to "string"

        # Select Teff and subsequently logg
        Teff_Grid = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500])
        optT = (abs(Teff - Teff_Grid)).argmin()

        logg_Grid = np.array([4.0, 4.5, 5.0])
        optG = (abs(logg - logg_Grid)).argmin()

        if logg_Grid[optG] == 4.0:
            Teff_Grid_load = np.array([8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 138])

        elif logg_Grid[optG] == 4.5:
            Teff_Grid_load = np.array([9, 20, 31, 42, 53, 64, 75, 86, 97, 108, 119, 129, 139])

        elif logg_Grid[optG] == 5.0:
            Teff_Grid_load = np.array([10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 130, 140])

        # Where in the model file is the section for the Teff we want? Index T_ind tells us that.
        T_ind = Teff_Grid_load[optT]
        header_rows = 3    #  How many rows in each section we ignore for the data reading
        data_rows = 1221   # How  many rows of data we read
        line_skip_data = (T_ind + 1) * header_rows + T_ind * data_rows   # Calculate how many lines in the model file we need to skip in order to get to the part we need (for the Teff we want).
        line_skip_header = T_ind * (data_rows + header_rows)

        # Read the header, in case we want to have the actual Teff, logg and M_H info.
        # headerinfo is a pandas object.
        headerinfo = pd.read_csv(os.path.join(dirsen, direc, model), delim_whitespace=True, header=None,
                                 skiprows=line_skip_header, nrows=1)

        Teff_model = headerinfo[1].values[0]
        logg_model = headerinfo[3].values[0]
        MH_model = headerinfo[6].values[0]
        MH_model = float(MH_model[1:-1])

        print('\nClosest values to your inputs:')
        print('Teff: ', Teff_model)
        print('M_H: ', MH_model)
        print('log(g): ', logg_model)

        # Read the data; data is a pandas object.
        data = pd.read_csv(os.path.join(dirsen, direc, model), delim_whitespace=True, header=None,
                              skiprows=line_skip_data, nrows=data_rows)

        # Unpack the data
        ws = data[0].values * 10   # Import wavelength data
        f0 = data[1].values / (ws * ws)
        f1 = data[2].values * f0 / 100000.
        f2 = data[3].values * f0 / 100000.
        f3 = data[4].values * f0 / 100000.
        f4 = data[5].values * f0 / 100000.
        f5 = data[6].values * f0 / 100000.
        f6 = data[7].values * f0 / 100000.
        f7 = data[8].values * f0 / 100000.
        f8 = data[9].values * f0 / 100000.
        f9 = data[10].values * f0 / 100000.
        f10 = data[11].values * f0 / 100000.
        f11 = data[12].values * f0 / 100000.
        f12 = data[13].values * f0 / 100000.
        f13 = data[14].values * f0 / 100000.
        f14 = data[15].values * f0 / 100000.
        f15 = data[16].values * f0 / 100000.
        f16 = data[17].values * f0 / 100000.

        # Make single big array of them
        fcalc = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16])
        phot1 = np.zeros(fcalc.shape[0])

        # Define mu
        mu = np.array([1.000, .900, .800, .700, .600, .500, .400, .300, .250, .200, .150, .125, .100, .075, .050, .025, .010])

        # Passed on to main body of function are: ws, fcalc, phot1, mu

    elif ld_model == '3D':

        direc = os.path.join(dirsen, '3DGrid')

        print('Current Directories Entered:')
        print('  ' + dirsen)
        print('  ' + direc)

        # Select metallicity
        M_H_Grid = np.array([-3.0, -2.0, -1.0, 0.0])  # Available metallicity values in 3D models
        M_H_Grid_load = ['30', '20', '10', '00']  # The according identifiers to individual available M_H values
        optM = (abs(M_H - M_H_Grid)).argmin()  # Find index at which the closes M_H values from available values is to the input M_H.

        # Select Teff
        Teff_Grid = np.array([4000, 4500, 5000, 5500, 5777, 6000, 6500, 7000])  # Available Teff values in 3D models
        optT = (abs(Teff - Teff_Grid)).argmin()  # Find index at which the Teff values is, that is closest to input Teff.

        # Select logg, depending on Teff. If several logg possibilities are given for one Teff, pick the one that is
        # closest to user input (logg).

        if Teff_Grid[optT] == 4000:
            logg_Grid = np.array([1.5, 2.0, 2.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 4500:
            logg_Grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5000:
            logg_Grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5500:
            logg_Grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5777:
            logg_Grid = np.array([4.4])
            optG = 0

        elif Teff_Grid[optT] == 6000:
            logg_Grid = np.array([3.5, 4.0, 4.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 6500:
            logg_Grid = np.array([4.0, 4.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 7000:
            logg_Grid = np.array([4.5])
            optG = 0

        # Select Teff and Log g. Mtxt, Ttxt and Gtxt are then put together as string to load correct files.
        Mtxt = M_H_Grid_load[optM]
        Ttxt = "{:2.0f}".format(Teff_Grid[optT] / 100)
        if Teff_Grid[optT] == 5777:
            Ttxt = "{:4.0f}".format(Teff_Grid[optT])
        Gtxt = "{:2.0f}".format(logg_Grid[optG] * 10)

        #
        file = 'mmu_t' + Ttxt + 'g' + Gtxt + 'm' + Mtxt + 'v05.flx'
        print('Filename:', file)

        # Read data from IDL .sav file
        sav = readsav(os.path.join(direc, file))  # readsav reads an IDL .sav file
        ws = sav['mmd'].lam[0]  # read in wavelength
        flux = sav['mmd'].flx  # read in flux
        Teff_model = Teff_Grid[optT]
        logg_model = logg_Grid[optG]
        MH_model = str(M_H_Grid[optM])

        print('\nClosest values to your inputs:')
        print('Teff: ', Teff_model)
        print('M_H: ', MH_model)
        print('log(g): ', logg_model)

        f0 = flux[0]
        f1 = flux[1]
        f2 = flux[2]
        f3 = flux[3]
        f4 = flux[4]
        f5 = flux[5]
        f6 = flux[6]
        f7 = flux[7]
        f8 = flux[8]
        f9 = flux[9]
        f10 = flux[10]

        # Make single big array of them
        fcalc = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
        phot1 = np.zeros(fcalc.shape[0])

        # Mu from grid
        # 0.00000    0.0100000    0.0500000     0.100000     0.200000     0.300000   0.500000     0.700000     0.800000     0.900000      1.00000
        mu = sav['mmd'].mu

        # Passed on to main body of function are: ws, fcalc, phot1, mu

    ### Load response function and interpolate onto kurucz model grid

    ## Spectroscopic Modes
    # FOR Hubble STIS
    if mode == 'STIS_G430L':
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/HST_STIS_G430L_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1

    if mode == 'STIS_G750L':
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/HST_STIS_G750L_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1

    # FOR Hubble WFC3
    if mode == 'WFC3_G141': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/HST_WFC3_G141_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1
        
    if mode == 'WFC3_G102': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/HST_WFC3_G102_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1

    if mode == 'WFC3_G280p1': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/HST_WFC3_G280p1_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1

    if mode == 'WFC3_G280n1':  
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/HST_WFC3_G280n1_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1        

    # FOR Webb NIRSpec
    if mode == 'NIRSpec_Prism':  
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_prism_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1

    if mode == 'NIRSpec_G395H': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G395H_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1

    if mode == 'NIRSpec_G395M': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G395M_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1        

    if mode == 'NIRSpec_G235H': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G235H_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1
        
    if mode == 'NIRSpec_G235M': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G235M_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1        
        
    if mode == 'NIRSpec_G140Hf100': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G140H-f100_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1
        
    if mode == 'NIRSpec_G140Mf100': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G140M-f100_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1
        
    if mode == 'NIRSpec_G140Hf070': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G140H-f070_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1
        
    if mode == 'NIRSpec_G140Mf070': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRSpec_G140M-f070_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1
        
    # For Webb NIRISS SOSS
    if mode == 'NIRISS_SOSSo1': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRISS_SOSSo1_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    
        
    if mode == 'NIRISS_SOSSo2': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRISS_SOSSo2_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    

    # For Webb NIRCam
    if mode == 'NIRCam_F322W2': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRCam_F322W2_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    
        
    if mode == 'NIRCam_F444': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_NIRCam_F444_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    
        
    # For Webb MIRI
    if mode == 'MIRI_LRS': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/JWST_MIRI_LRS_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    

    ## Photometric Modes    
    # Spitzer IRAC
    if mode == 'IRAC_Ch1': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/Spitzer_IRAC_Ch1_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    

    if mode == 'IRAC_Ch2': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/Spitzer_IRAC_Ch2_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1            

    # TESS    
    if mode == 'TESS': 
        sav = pd.read_csv(os.path.join(dirsen,'sensitivity_files/TESS_throughput.csv'))
        wssens = sav['wave'].values
        sensitivity = sav['tp'].values
        wdel = 1    
        
    widek = np.arange(len(wsdata))
    wsmode = wssens
    wsmode = np.concatenate((np.array([wsmode[0] - wdel - wdel, wsmode[0] - wdel]),
                            wsmode,
                            np.array([wsmode[len(wsmode) - 1] + wdel,
                                      wsmode[len(wsmode) - 1] + wdel + wdel])))

    respoutmode = sensitivity / np.max(sensitivity)
    respoutmode = np.concatenate((np.zeros(2), respoutmode, np.zeros(2)))

    inter_resp = interp1d(wsmode, respoutmode, bounds_error=False, fill_value=0)
    respout = inter_resp(ws)  # interpolate sensitivity curve onto model wavelength grid

    wsdata = np.concatenate((np.array([wsdata[0] - wdel - wdel, wsdata[0] - wdel]), wsdata,
                             np.array([wsdata[len(wsdata) - 1] + wdel, wsdata[len(wsdata) - 1] + wdel + wdel])))
    respwavebin = wsdata / wsdata * 0.0
    widek = widek + 2  # need to add two indicies to compensate for padding with 2 zeros
    respwavebin[widek] = 1.0
    data_resp = interp1d(wsdata, respwavebin, bounds_error=False, fill_value=0)
    reswavebinout = data_resp(ws)  # interpolate data onto model wavelength grid

    # Integrate over the spectra to make synthetic photometric points.
    for i in range(fcalc.shape[0]):  # Loop over spectra at diff angles
        fcal = fcalc[i, :]
        Tot = int_tabulated(ws, ws * respout * reswavebinout)
        phot1[i] = (int_tabulated(ws, ws * respout * reswavebinout * fcal, sort=True)) / Tot

    if ld_model == '1D':
        yall = phot1 / phot1[0]
    elif ld_model == '3D':
        yall = phot1 / phot1[10]

    Co = np.zeros((6, 4))   # NOT-REUSED

    A = [0.0, 0.0, 0.0, 0.0]  # c1, c2, c3, c4      # NOT-REUSED
    x = mu[1:]     # wavelength
    y = yall[1:]   # flux
    weights = x / x   # NOT-REUSED

    # Start fitting the different models
    fitter = LevMarLSQFitter()

    # Fit a four parameter non-linear limb darkening model and get fitted variables, c1, c2, c3, c4.
    corot_4_param = nonlinear_limb_darkening()
    corot_4_param = fitter(corot_4_param, x, y)
    c1, c2, c3, c4 = corot_4_param.parameters

    # Fit a three parameter non-linear limb darkening model and get fitted variables, cp2, cp3, cp4 (cp1 = 0).
    corot_3_param = nonlinear_limb_darkening()
    corot_3_param.c0.fixed = True  # 3 param is just 4 param with c0 = 0.0
    corot_3_param = fitter(corot_3_param, x, y)
    cp1, cp2, cp3, cp4 = corot_3_param.parameters

    # Fit a quadratic limb darkening model and get fitted parameters aLD and bLD.
    quadratic = quadratic_limb_darkening()
    quadratic = fitter(quadratic, x, y)
    aLD, bLD = quadratic.parameters

    # Fit a linear limb darkening model and get fitted variable uLD.
    linear = nonlinear_limb_darkening()
    linear.c0.fixed = True
    linear.c2.fixed = True
    linear.c3.fixed = True
    linear = fitter(linear, x, y)
    uLD = linear.c1.value

    print('\nLimb darkening parameters:')
    print("4param \t{:0.8f}\t{:0.8f}\t{:0.8f}\t{:0.8f}".format(c1, c2, c3, c4))
    print("3param \t{:0.8f}\t{:0.8f}\t{:0.8f}".format(cp2, cp3, cp4))
    print("Quad \t{:0.8f}\t{:0.8f}".format(aLD, bLD))
    print("Linear \t{:0.8f}".format(uLD))

    return uLD, c1, c2, c3, c4, cp1, cp2, cp3, cp4, aLD, bLD

def int_tabulated(X, F, sort=False):
    Xsegments = len(X) - 1

    # Sort vectors into ascending order.
    if not sort:
        ii = np.argsort(X)
        X = X[ii]
        F = F[ii]

    while (Xsegments % 4) != 0:
        Xsegments = Xsegments + 1

    Xmin = np.min(X)
    Xmax = np.max(X)

    # Uniform step size.
    h = (Xmax + 0.0 - Xmin) / Xsegments
    # Compute the interpolates at Xgrid.
    # x values of interpolates >> Xgrid = h * FINDGEN(Xsegments + 1L) + Xmin
    z = splev(h * np.arange(Xsegments + 1) + Xmin, splrep(X, F))

    # Compute the integral using the 5-point Newton-Cotes formula.
    ii = (np.arange((len(z) - 1) / 4, dtype=int) + 1) * 4

    return np.sum(2.0 * h * (7.0 * (z[ii - 4] + z[ii]) + 32.0 * (z[ii - 3] + z[ii - 1]) + 12.0 * z[ii - 2]) / 45.0)


@custom_model
def nonlinear_limb_darkening(x, c0=0.0, c1=0.0, c2=0.0, c3=0.0):
    """
    Define non-linear limb darkening model with four parameters c0, c1, c2, c3.
    """
    model = (1. - (c0 * (1. - x ** (1. / 2)) + c1 * (1. - x ** (2. / 2)) + c2 * (1. - x ** (3. / 2)) + c3 *
                   (1. - x ** (4. / 2))))
    return model


@custom_model
def quadratic_limb_darkening(x, aLD=0.0, bLD=0.0):
    """
    Define linear limb darkening model with parameters aLD and bLD.
    """
    model = 1. - aLD * (1. - x) - bLD * (1. - x) ** (4. / 2.)
    return model
