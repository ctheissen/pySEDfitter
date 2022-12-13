from __future__ import print_function
import sys, os, os.path, time, gc, time
import multiprocessing as mp
mp.set_start_method('fork')
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from astropy.table import Table, join
from astropy.constants import sigma_sb
import glob
import astroroutines as AS
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pysynphot as pysyn
from matplotlib.offsetbox import AnchoredText
from InterpolateModel import InterpModel, InterpModelLog
from SyntheticPhot import SynPhot
from astropy.io import fits
import emcee
import matplotlib.gridspec as gridspec
import random
import smart
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

v = Vizier(timeout=100000, columns=["**", "+_r"], vizier_server='vizier.cfa.harvard.edu')

# You can set these variables
TEST         = False
multiprocess = True
MCMC         = False
plotWalkers  = True
plotCorner   = True
noise        = True
walkers      = 100 
steps        = 1000
size         = 1000 # number of random models to use
checkFiles   = False # set this to True if you stopped the script and want to restart without doing previous fits
modelset, instrument = 'PHOENIX-ACES-AGSS-COND-2011', 'lowres100'
#modelset, instrument = 'PHOENIX-BTSETTL-CIFIST2011-2015', 'lowres100'
outputfluxunits = u.erg / (u.s * u.cm**2 * u.angstrom)
outputwaveunits = u.angstrom

#chainsDir    = 'Latemovers_chains' # directory that will be created to store the chains
#plotsDir     = 'Latemovers_SEDs' # directory that will be created to store the plots
chainsDir    = '/mnt/TESS/Single_Sources/%s/chains'%modelset # directory that will be created to store the chains
plotsDir     = '/mnt/TESS/Single_Sources/%s/Plots'%modelset # directory that will be created to store the plots
writeDir     = '/mnt/TESS/Single_Sources/%s/luminosity_radius'%modelset # directory that will be created to store the outputs
writeDir2    = '/mnt/TESS/Single_Sources/%s/luminosity_radius_compressed'%modelset # directory that will be created to store the compressed params

# MCMC variables
nwalkers, nsteps, iteration = walkers, steps, -1
burn = nsteps-100

# Plotting variables
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=7)
plt.rc('axes', labelsize=10)

# Start timing counter to see how fast it runs
start = time.time()

# Define constants
light_speed = 2.99792458e5  # speed of light (km/s)


###############################################################

ra  = 57.7396885   
dec = 18.3017689 
obj = 'LP413-53'
saveFile     = "MCMC_%s.dat"%obj
fileName     = "MCMC_%s_Radius_Luminosity.csv"%obj

###############################################################

# Get the filter response curves
jcurve, hcurve, kcurve                                        = AS.transCurves('2mass', o='ang')
ucurve, gcurve, rcurve, icurve, zcurve                        = AS.transCurves('sdss_doi', norm='normalize')
opcurve, gpcurve, rpcurve, ipcurve, zpcurve, ypcurve, wPcurve = AS.transCurves('panstarrs_ps1', norm='normalize')
zcurve_uk, ycurve_uk, jcurve_uk, hcurve_uk, kcurve_uk         = AS.transCurves('ukidss', norm='normalize', o='ang')
Gcurve, BPcurve, RPcurve                                      = AS.transCurves('gaia_edr3', o='ang')
w1curve, w2curve, w3curve, w4curve                            = AS.transCurves('wise', o='ang')
'''
plt.plot(gcurve[0], gcurve[1], label='sdss')
plt.plot(gpcurve[0], gpcurve[1], label='panstarrs')
plt.plot(zcurve_uk[0]*1e4, zcurve_uk[1], label='UKIDSS Z')
plt.plot(hcurve_uk[0]*1e4, hcurve_uk[1], label='UKIDSS H')
plt.legend()
plt.show()
sys.exit()
'''
if plotCorner: import corner

# Save this for downsampling models to plot later
#newwaves = np.logspace(3, np.log10(800000), 1000)

# Make the directories if they don't exist
if not os.path.exists(chainsDir):
    os.makedirs(chainsDir)
if not os.path.exists(plotsDir):
    os.makedirs(plotsDir)
if not os.path.exists(writeDir):
    os.makedirs(writeDir)
if not os.path.exists(writeDir2):
    os.makedirs(writeDir2)
'''
T1 = Table.read('PHOENIX-ACES-AGSS-COND-2011_lowres1_Photometry_Table.fits')
#T1 = T0[np.where(T0['en']==0)]
T1['teff']  = T1['Teff']
T1['logg']  = T1['Logg']
T1['M_H']   = T1['Metal']
'''

# Check if the object has already been processed
if checkFiles: filelist = set(glob.glob('%s/*.npy'%chainsDir))

#############################################################

def bilinear_interpolation(x, y, points):

        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        interpFlux = 10**((q11 * (x2 - x) * (y2 - y) +
                           q21 * (x - x1) * (y2 - y) +
                           q12 * (x2 - x) * (y - y1) +
                           q22 * (x - x1) * (y - y1)
                           ) / ((x2 - x1) * (y2 - y1) + 0.0))

        return interpFlux

#############################################################

count = 1


print(ra, dec)
c = SkyCoord(ra, dec, unit=('deg', 'deg'), frame='icrs')
results = v.query_region(c, radius = 10*u.arcsec, catalog=['SDSS', 'GAIA','UKIDSS', 'PANSTARRS', 'WISE', '2MASS'])#, cache=False)
print(results)
#print(results['II/314/las8'])
#sys.exit()

# Get SDSS DR16
try: 
    tableS = results['V/154/sdss16'].filled(-9999)
    tableS.pprint_all()
    imag   = tableS['_tab1_12'].data[0]
    imagu  = tableS['e_ipmag'].data[0]
    zmag   = tableS['_tab1_13'].data[0]
    zmagu  = tableS['e_zpmag'].data[0]
    sdss16 = True
except: 
    sdss16 = False

# Get UKIDSS
try: 
    tableU0 = results['II/319/las9'].filled(-9999)
    tableU0.pprint_all()
    tableU = tableU0[np.where(tableU0['m']==1)]
    tableU.pprint()
    ymag_uk   = tableU['Ymag'].data[0]
    ymagu_uk  = tableU['e_Ymag'].data[0]
    j1mag_uk  = tableU['Jmag1'].data[0]
    j1magu_uk = tableU['e_Jmag1'].data[0]
    j2mag_uk  = tableU['Jmag2'].data[0]
    j2magu_uk = tableU['e_Jmag2'].data[0]
    hmag_uk   = tableU['Hmag'].data[0]
    hmagu_uk  = tableU['e_Hmag'].data[0]
    kmag_uk   = tableU['Kmag'].data[0]
    kmagu_uk  = tableU['e_Kmag'].data[0]
    if j1mag_uk != -9999 and j2mag_uk != -9999:
        jmag_uk  = np.average([j1mag_uk, j1mag_uk])
        jmagu_uk = np.average([j1magu_uk, j1magu_uk])
    elif j1mag_uk == -9999: 
        jmag_uk = j2mag_uk
        jmagu_uk = j2magu_uk
    elif j2mag_uk == -9999: 
        jmag_uk = j1mag_uk
        jmagu_uk = j1magu_uk
    ukidss = True
except: 
    ukidss = False

#ukidss = False
# Get Gaia
try:
    tableG  = results['I/350/gaiaedr3'].filled(-9999)
    tableG.pprint_all()
    Gmag    = tableG['FG'].data[0]
    Gmagu   = tableG['e_FG'].data[0]
    RPmag   = tableG['FRP'].data[0]
    RPmagu  = tableG['e_FRP'].data[0]
    Gaia    = True
except: 
    Gaia    = False


# Get Pan-STARRS
try:
    tableP  = results['II/349/ps1'].filled(-9999)
    tableP.pprint_all()
    ipmag   = tableP['imag'].data[0]
    ipmagu  = tableP['e_imag'].data[0]
    zpmag   = tableP['zmag'].data[0]
    zpmagu  = tableP['e_zmag'].data[0]
    ypmag   = tableP['ymag'].data[0]
    ypmagu  = tableP['e_ymag'].data[0]
    panstarrs    = True
except: 
    panstarrs    = False

# Get 2MASS
try:
    table2  = results['II/246/out'].filled(-9999)
    table2.pprint_all()
    jmag   = table2['Jmag'].data[0]
    jmagu  = table2['e_Jmag'].data[0]
    hmag   = table2['Hmag'].data[0]
    hmagu  = table2['e_Hmag'].data[0]
    kmag   = table2['Kmag'].data[0]
    kmagu  = table2['e_Kmag'].data[0]
    twomass    = True
except: 
    twomass    = False

# Get CATWISE
try:
    tableW  = results['II/365/catwise'].filled(-9999)
    tableW.pprint_all()
    w1mag   = tableW['W1mproPM'].data[0]
    w1magu  = tableW['e_W2mproPM'].data[0]
    w2mag   = tableW['W1mproPM'].data[0]
    w2magu  = tableW['e_W2mproPM'].data[0]
    wise    = True
except: 
    wise    = False


#imag    = T['imag'][i]
#imagu   = T['e_imag'][i]
#zmag    = T['zmag'][i]
#zmagu   = T['e_zmag'][i]

#ipmag   = T['PANSTARRS_imag'][i]
#ipmagu  = T['PANSTARRS_e_imag'][i]
#zpmag   = T['PANSTARRS_zmag'][i]
#zpmagu  = T['PANSTARRS_e_zmag'][i]
#ypmag   = T['PANSTARRS_ymag'][i]
#ypmagu  = T['PANSTARRS_e_ymag'][i]

#jmag   = T['Jmag'][i]
#jmagu  = T['e_Jmag'][i]
#hmag   = T['Hmag'][i]
#hmagu  = T['e_Hmag'][i]
#kmag   = T['Kmag'][i] 
#kmagu  = T['e_Kmag'][i]
#w1mag  = T['W1mag'][i]
#w1magu = T['e_W1mag'][i]
#w2mag  = T['W2mag'][i]
#w2magu = T['e_W2mag'][i]
#w3mag  = T['W3mag'][i]
#w3magu = T['e_W3mag'][i]
#w4mag  = T['W4mag'][i]
#w4magu = T['e_W4mag'][i]
print(ukidss, Gaia)
#print(w3mag, w3magu, w4mag, w4magu)
'''
sed1 = SED(name='%s'%obj)
sed1.add_photometry('Gaia.G', Gmag, mag_unc=Gmagu, system='Vega')
#sed1.add_photometry('Gaia.RP', RPmag, mag_unc=RPmagu, system='Vega')
sed1.add_photometry('2MASS.J', jmag, mag_unc=jmagu, system='Vega')
sed1.add_photometry('2MASS.H', hmag, mag_unc=hmagu, system='Vega')
sed1.add_photometry('2MASS.Ks', kmag, mag_unc=kmagu, system='Vega')
sed1.add_photometry('WISE.W1', w1mag, mag_unc=w1magu, system='Vega')
sed1.add_photometry('WISE.W2', w2mag, mag_unc=w2magu, system='Vega')
sed1.fit_modelgrid(BTSettl(), mcmc=True)
sys.exit()
'''

# Just drop these values
#w1mag, w1magu = -9999, -9999
#w2mag, w2magu = -9999, -9999
w3mag, w3magu = -9999, -9999 # Models don't cover this range
w4mag, w4magu = -9999, -9999 # Models don't cover this range
#Gmag, Gmagu = -9999, -9999
#RPmag, RPmagu = -9999, -9999
#imag, imagu = -9999, -9999
#ipmag, ipmagu = -9999, -9999
#zpmag, zpmagu = -9999, -9999
#ypmag, ypmagu = -9999, -9999

aG, aGun   = -9999, -9999
aRP, aRPun = -9999, -9999
ai, aiun   = -9999, -9999
az, azun   = -9999, -9999
aip, aipun = -9999, -9999
azp, azpun = -9999, -9999
ayp, aypun = -9999, -9999
aj, ajun   = -9999, -9999
ah, ahun   = -9999, -9999
ak, akun   = -9999, -9999
aw1, aw1un = -9999, -9999
aw2, aw2un = -9999, -9999
aw3, aw3un = -9999, -9999
aw4, aw4un = -9999, -9999

ayu, ayuun = -9999, -9999
aju, ajuun = -9999, -9999
ahu, ahuun = -9999, -9999
aku, akuun = -9999, -9999

binary = []

class star(object):
    def __init__(self, *args, **kwargs):
        # Define the constants for this object
        self.mcmc                   = False

star.keys = []

#print(Gmagu, RPmagu, imagu, zmagu, ipmagu, zpmagu, ypmagu, jmagu, hmagu, kmagu, w1magu, w2magu)


if Gmagu in [-9999, 0] : binary.append(1)
else: 
    Gmags = (Gmag, Gmagu)
    binary.append(0)
    #print('1')
    #aG, aGun = AS.magtoflux(Gmag, 'ang', f0=Gmag, gaia_edr3='G', mag_uncert = Gmagu, S2N=S2N)
    aG, aGun = AS.magtoflux(Gmag, 'ang', f0=Gmag, gaia_edr3='G', mag_uncert = Gmagu, flux_error = Gmagu)
    star.keys.append('Gmag')

if RPmagu in [-9999, 0] : binary.append(1)
else: 
    RPmags = (RPmag, RPmagu)
    binary.append(0)
    #print('1')
    #aRP, aRPun = AS.magtoflux(RPmag, 'ang', f0=RPmag, gaia_edr3='G', mag_uncert = Gmagu, S2N=S2N)
    aRP, aRPun = AS.magtoflux(RPmag, 'ang', f0=RPmag, gaia_edr3='RP', mag_uncert = RPmagu, flux_error = RPmagu)
    star.keys.append('RPmag')

if imagu in [-9999, 0]: binary.append(1)
else: 
    imags = (imag, imagu)
    binary.append(0)
    #print('2')
    ai, aiun = AS.magtoflux(imag, 'ang', sdss='i', mag_uncert = imagu)
    #print(ai, aiun)
    #sys.exit()
    star.keys.append('imag')

if zmagu in [-9999, 0]: binary.append(1)
else: 
    zmags = (zmag, zmagu)
    binary.append(0)
    #print('3')
    az, azun = AS.magtoflux(zmag, 'ang', sdss='z', mag_uncert = zmagu)
    star.keys.append('zmag')

if ipmagu in [-9999, 0]: binary.append(1)
else: 
    ipmags = (ipmag, ipmagu)
    binary.append(0)
    #print('4')
    aip, aipun = AS.magtoflux(ipmag, 'ang', ps='i', mag_uncert = ipmagu)
    star.keys.append('ipmag')

if zpmagu in [-9999, 0]: binary.append(1)
else: 
    zpmags = (zpmag, zpmagu)
    binary.append(0)
    #print('5')
    azp, azpun = AS.magtoflux(zpmag, 'ang', ps='z', mag_uncert = zpmagu)
    star.keys.append('zpmag')

if ypmagu in [-9999, 0]: binary.append(1)
else: 
    ypmags = (ypmag, ypmagu)
    binary.append(0)
    #print('6')
    ayp, aypun = AS.magtoflux(ypmag, 'ang', ps='y', mag_uncert = ypmagu)
    star.keys.append('ypmag')

if jmagu in [-9999, 0]: binary.append(1)
else: 
    jmags = (jmag, jmagu)
    binary.append(0)
    aj, ajun = AS.magtoflux(jmag, 'ang', mass='j', mag_uncert = jmagu)
    star.keys.append('jmag')
if hmagu in [-9999, 0]: binary.append(1)
else: 
    hmags = (hmag, hmagu)
    binary.append(0)
    #print('7')
    ah, ahun = AS.magtoflux(hmag, 'ang', mass='h', mag_uncert = hmagu)
    star.keys.append('hmag')
if kmagu in [-9999, 0]: binary.append(1)
else: 
    kmags = (kmag, kmagu)
    binary.append(0)
    ak, akun = AS.magtoflux(kmag, 'ang', mass='k', mag_uncert = kmagu)
    star.keys.append('ksmag')
    
if w1magu in [-9999, 0]: binary.append(1)
else: 
    w1mags = (w1mag, w1magu)
    binary.append(0)
    #print('8')
    aw1, aw1un = AS.magtoflux(w1mag, 'ang', wise=1, powerlaw=-2, mag_uncert = w1magu)
    star.keys.append('w1mag')

if w2magu in [-9999, 0]: binary.append(1)
else: 
    w2mags = (w2mag, w2magu)
    #print(w2mags)
    binary.append(0)
    aw2, aw2un = AS.magtoflux(w2mag, 'ang', wise=2, powerlaw=-2, mag_uncert = w2magu)
    star.keys.append('w2mag')
'''
if w3magu in [-9999, 0]: binary.append(1)
else: 
    w3mags = (w3mag, w3magu)
    #print(w3mags)
    binary.append(0)
    aw3, aw3un = AS.magtoflux(w3mag, 'ang', wise=3, powerlaw=-2, mag_uncert = w3magu)
    star.keys.append('w3mag')

if w4magu in [-9999, 0]: binary.append(1)
else: 
    w4mags = (w4mag, w4magu)
    #print(w4mags)
    binary.append(0)
    aw4, aw4un = AS.magtoflux(w4mag, 'ang', wise=4, powerlaw=-2, mag_uncert = w4magu)
    star.keys.append('w4mag')
'''
print(star.keys)
#print(aw3, aw4)
#sys.exit()



if ukidss: 

    print(ymagu_uk, jmagu_uk, hmagu_uk, kmagu_uk)   
    if ymagu_uk in [-9999, 0] : binary.append(1)
    else: 
        ymags_uk = (ymag_uk, ymagu_uk)
        binary.append(0)
        ayu, ayuun = AS.magtoflux(ymag_uk, 'ang', ukidss='y', mag_uncert = ymagu_uk)
        star.keys.append('ymag_uk')
    
    if jmagu_uk in [-9999, 0] : binary.append(1)
    else: 
        jmags_uk = (jmag_uk, jmagu_uk)
        binary.append(0)
        aju, ajuun = AS.magtoflux(jmag_uk, 'ang', ukidss='j', mag_uncert = jmagu_uk)
        star.keys.append('jmag_uk')
            
    if hmagu_uk in [-9999, 0] : binary.append(1)
    else: 
        hmags_uk = (hmag_uk, hmagu_uk)
        binary.append(0)
        ahu, ahuun = AS.magtoflux(hmag_uk, 'ang', ukidss='h', mag_uncert = hmagu_uk)
        star.keys.append('hmag_uk')

    if kmagu_uk in [-9999, 0] : binary.append(1)
    else: 
        kmags_uk = (kmag_uk, kmagu_uk)
        binary.append(0)
        aku, akuun = AS.magtoflux(kmag_uk, 'ang', ukidss='k', mag_uncert = kmagu_uk)
        star.keys.append('kmag_uk')

    print(ymags_uk, jmags_uk, hmags_uk, kmags_uk)

#print(jmag, jmag_uk)
#print(aj, aju)
#sys.exit()
#if w3magu == -9999: binary.append(1)
#else: 
#    w3mags = (w3mag, w3magu)
#    binary.append(0)
#    a11, a11un = AS.magtoflux(w3mag, 'ang', wise=3, powerlaw=-2, mag_uncert = w3magu)
#    star.keys.append('w3mag')


#FluxesNormed = np.array([a2, a3, a6, a7, a8, a9, a10, a11])
#FluxesUncert = np.array([a2un, a3un, a6un, a7un, a8un, a9un, a10un, a11un])

#FluxesNormed = np.ma.array(FluxesNormed, mask=binary)
#FluxesUncert = np.ma.array(FluxesUncert, mask=binary)
#print(aG, ai, az, aip, azp, ayp, aj, ah, ak, aw1, aw2, aw3, aw4)
FluxesNormed  = np.ma.array([aG, aRP, ai, az, aip, azp, ayp, aj, ah, ak, aw1, aw2], mask=binary[0:12], dtype=np.float)
FluxesUncert  = np.ma.array([aGun, aRPun, aiun, azun, aipun, azpun, aypun, ajun, ahun, akun, aw1un, aw2un], mask=binary[0:12], dtype=np.float)
if ukidss:
    FluxesUK      = np.ma.array([ayu, aju, ahu, aku], mask=binary[12:], dtype=np.float)
    FluxesUK_Un   = np.ma.array([ayuun, ajuun, ahuun, akuun], mask=binary[12:], dtype=np.float)
    FluxesNormed  = np.ma.concatenate([FluxesNormed, FluxesUK])
    FluxesUncert  = np.ma.concatenate([FluxesUncert, FluxesUK_Un])
star.fluxes   = FluxesNormed
star.fluxerrs = FluxesUncert
'''
print(star.keys)
print(star.fluxes)
print(star.fluxerrs)
print(len(star.keys))
print(len(star.fluxes))
print(len(star.fluxerrs))
sys.exit()
'''


if MCMC:
    def ln_likelihood(parameters, fluxes1, fluxerrs1, keys, noise=True, ukidss=False):

        if noise: 
            if ukidss:
                teff, logg, metal, scale, noise1, noise2, noise3, noise4, noise5, noise6 = parameters
            else:
                teff, logg, metal, scale, noise1, noise2, noise3, noise4, noise5 = parameters
        else: 
            Teff, logg, metal, scale = parameters

    
        model  = smart.Model(teff=teff, logg=logg, metal=metal, modelset=modelset, instrument=instrument)
        waves  = model.wave
        fluxes = model.flux
        
        modelFluxesTry = SynPhot(Waves=waves, Fluxes=fluxes*10**scale, bands=keys, Ang=True)
        '''
        print('1', len(keys))
        print('2', len(modelFluxesTry))
        print('3', binary, len(binary))
        print('4', len(fluxes1.compressed()))
        print(fluxes1)
        print(fluxes1.compressed())
        '''

        if ukidss:
            sigma = np.ma.array( np.concatenate( (fluxerrs1[:2].flatten()*noise1, 
                                    fluxerrs1[2:4].flatten()*noise2, 
                                    fluxerrs1[4:7].flatten()*noise3, 
                                    fluxerrs1[7:10].flatten()*noise4, 
                                    fluxerrs1[10:12].flatten()*noise5,
                                    fluxerrs1[12:].flatten()*noise6,
                                    ) ), mask=binary, dtype=np.float )
        else:
            sigma = np.ma.array( np.concatenate( (fluxerrs1[:2].flatten()*noise1, 
                                    fluxerrs1[2:4].flatten()*noise2, 
                                    fluxerrs1[4:7].flatten()*noise3, 
                                    fluxerrs1[7:10].flatten()*noise4, 
                                    fluxerrs1[10:].flatten()*noise5,
                                    ) ), mask=binary, dtype=np.float )
        #print(sigma)
        #sys.exit()
        #print(fluxes1.compressed().shape)
        #print(sigma.compressed().shape)
        #print(np.array(modelFluxesTry).flatten().shape)
        LogChiSq = -0.5*np.sum( ( (fluxes1.compressed() - np.array(modelFluxesTry).flatten() ) / (sigma.compressed()) )**2 + \
                        np.log( (sigma.compressed())**2 ) ) 
        #print LogChiSq
        return LogChiSq


    def ln_prior(parameters, noise=True, ukidss=False):
        # Define the priors. Since the scaling factor covers a lot of ground I used a
         # Jefferies prior.

        #TestScale = 200.
        #Tmin, Tmax         = 1500, 5500 # BT-Settl
        Tmin, Tmax         = 2300, 6000 # Phoenix ACES
        loggmin, loggmax   = 2.5, 5.5
        metalmin, metalmax = -4, 1
        Scalemin, Scalemax = -40., 0.
        if noise: 
            noisefactormin, noisefactormax = 1, 100
        #deltaT, deltaG, Scale = (Tmax-Tmin)/TestScale, loggmax-loggmin, 10**Scalemax/10**Scalemin
        #deltaT, deltaG, Scale = (Tmax-Tmin), loggmax-loggmin, 10**Scalemax/10**Scalemin
          
        if noise:
            if ukidss:
                Teff, logg, metal, scale, noise1, noise2, noise3, noise4, noise5, noise6 = parameters
            else: 
                Teff, logg, metal, scale, noise1, noise2, noise3, noise4, noise5 = parameters
        else:
            Teff, logg, metal, scale = parameters

        #if  Teff != 0 and logg != 0:
        #if  Tmin/TestScale <= Teff <= Tmax/TestScale and loggmin <= logg <= loggmax and Scalemin <= scale <= Scalemax:
        if noise:
            if ukidss:
                if  Tmin <= Teff <= Tmax and loggmin <= logg <= loggmax and metalmin <= metal <= metalmax and Scalemin <= scale <= Scalemax and \
                noisefactormin < noise1 < 10000 and noisefactormin < noise2 < 100 and noisefactormin < noise3 < 100 and \
                noisefactormin < noise4 < 1.5 and noisefactormin < noise5 < 1.5 and noisefactormin < noise6 < 100:
                    return 0
                else:
                    return -np.inf
            else:
                if  Tmin <= Teff <= Tmax and loggmin <= logg <= loggmax and metalmin <= metal <= metalmax and Scalemin <= scale <= Scalemax and \
                noisefactormin < noise1 < 10000 and noisefactormin < noise2 < 100 and noisefactormin < noise3 < 100 and \
                noisefactormin < noise4 < 1.5 and noisefactormin < noise5 < 1.5:
                    return 0
                else:
                    return -np.inf
        else:
            if  Tmin <= Teff <= Tmax and loggmin <= logg <= loggmax and Scalemin <= scale <= Scalemax:
                #return -np.log(deltaT) - np.log(deltaG) - np.log(10**scale * np.log(Scale))
                return 0#-np.log(deltaT) - np.log(deltaG) - np.log(10**scale * np.log(Scale))
            else:
                return -np.inf
    
    

    def ln_probability(parameters, fluxes1, fluxerrs1, keys, noise=True, ukidss=False):

        # Get the parameters
        #print(noise)
        #print(ukidss)
        #print(parameters)
        if noise:
            if ukidss:
                Temp, logg, metal, scale, noise1, noise2, noise3, noise4, noise5, noise6 = parameters
            else:
                Temp, logg, metal, scale, noise1, noise2, noise3, noise4, noise5 = parameters
        else:
            Temp, logg, scale = parameters

        ###print 'PARAMS', Temp*200, logg, scale
     
        # Call the prior function, check that it doesn't return -inf then create the log-
        # probability function
        priors = ln_prior(parameters, ukidss=ukidss)
        #print 'PRIORS', priors
          
        if not np.isfinite(priors):
            return -np.inf
        else:
            #print 'TOTAL', priors + ln_likelihood(mod, weights, parameters, star, model, binary)
            return priors + ln_likelihood(parameters, fluxes1, fluxerrs1, keys, ukidss=ukidss)

    

    # Almost there! Now we must initialize our walkers. Remember that emcee uses a bunch of 
    # walkers, and we define their starting distribution. 
    if noise:
        if ukidss:
            n_dim, n_walkers, n_steps = 10, walkers, steps
        else:
            n_dim, n_walkers, n_steps = 9, walkers, steps
    else:
        n_dim, n_walkers, n_steps = 3, walkers, steps

    Temp_rand  = []
    logg_rand  = []
    metal_rand = []
    #min_val, max_val = 1e-30, 1e-10
    scale_min_val, scale_max_val = -40., 0.
    scale_rand = []
    if noise: 
        noise_rand  = []
        noise_rand2 = []
        noise_rand3 = []
        noise_rand4 = []
        noise_rand5 = []
        noise_rand6 = []

    T0, LogG, Scale0, Noise0 = 3000, 5., -20, 1.
      
    for i in range(n_walkers):
        """
        #Temp_rand.append( random.uniform( np.min(model.modeltable['Temp']/200.), np.max(model.modeltable['Temp']/200.) ) )
        Temp_rand.append( random.uniform( np.min(model.modeltable['Temp']), np.max(model.modeltable['Temp']) ) )
        logg_rand.append( random.uniform( np.min(model.modeltable['Logg']), np.max(model.modeltable['Logg']) ) )
        #scale_rand.append(10 ** random.uniform(np.log10(min_val), np.log10(max_val)))
        scale_rand.append(random.uniform(scale_min_val, scale_max_val))
        """
        #Temp_rand.append( random.uniform( 2300, 6000 ) )
        Temp_rand.append( random.uniform( 1500, 6000 ) )
        logg_rand.append( random.uniform( 2.5, 5.5 ) )
        metal_rand.append( random.uniform( -4, 1 ) )
        #scale_rand.append(10 ** random.uniform(np.log10(min_val), np.log10(max_val)))
        scale_rand.append( random.uniform(-40, 0) )
        if noise: 
            noise_rand.append(  random.uniform(1, 1000) )
            noise_rand2.append( random.uniform(1, 100) )
            noise_rand3.append( random.uniform(1, 100) )
            noise_rand4.append( random.uniform(1, 1.5) )
            noise_rand5.append( random.uniform(1, 1.5) )
            noise_rand6.append( random.uniform(1, 100) )
        #scale_rand.append( random.uniform(1e-30, 1e-10) )

    # positions should be a list of N-dimensional arrays where N is the number of parameters 
    #  you have. The length of the list should match n_walkers.
    positions = []
    for param in range(n_walkers):
        if ukidss:
            positions.append(np.array([Temp_rand[param], logg_rand[param], metal_rand[param], scale_rand[param], noise_rand[param], noise_rand2[param], noise_rand3[param], noise_rand4[param], noise_rand5[param], noise_rand6[param]]))
        else:
            positions.append(np.array([Temp_rand[param], logg_rand[param], metal_rand[param], scale_rand[param], noise_rand[param], noise_rand2[param], noise_rand3[param], noise_rand4[param], noise_rand5[param]]))
    positions = np.array(positions)
    print(positions.shape)
    #print(nsteps)
        
    # Set up and run the emcee sampler
    start = time.time()
    if multiprocess:    ## multiprocessing 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, n_dim, ln_probability, kwargs={'ukidss':ukidss}, args=(star.fluxes, star.fluxerrs, star.keys), pool=pool, moves=emcee.moves.KDEMove())
            sampler.run_mcmc(positions, nsteps, progress=True)
            print('TIME1:', time.time()-start)
    else:
        print(star.fluxes)
        print(star.fluxerrs)
        print(star.keys)
        print(len(star.fluxes))
        print(len(star.fluxerrs))
        print(len(star.keys))
        sampler = emcee.EnsembleSampler(nwalkers, n_dim, ln_probability, kwargs={'ukidss':ukidss}, args=(star.fluxes, star.fluxerrs, star.keys), moves=emcee.moves.KDEMove() )
        sampler.run_mcmc(positions, nsteps, progress=True)
        print('TIME1:', time.time()-start)

    # Save the sampler
    np.save('%s/%s.dat'%(chainsDir, obj), sampler.chain)
    print('TIME2:', time.time()-start)

    # Grab the chains
    nwalkers, step, ndim = sampler.chain.shape

    if plotWalkers:
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(ndim, 1)
        gs.update(hspace=0.1)

        for i in range(ndim):
          ax = fig.add_subplot(gs[i, :])
          for j in range(nwalkers):
              ax.plot(np.arange(1,int(step+1)),
                  sampler.chain[j,:,i],'k',alpha=0.2)
              #ax.set_ylabel(ylabels[i])
        fig.align_ylabels()
        plt.xlabel('nstep')
        plt.savefig('%s/walkers_%s.png'%(plotsDir, obj), dpi=300, bbox_inches='tight')




    # Now apply the burn by cutting those steps out of the chain. 
    chain_burnt = sampler.chain[:, burn:, :] 
    # Also flatten the chain to just a list of samples
    samples = chain_burnt.reshape((-1, ndim))
    
    # So that's all well and good, but what are my best-fit parameter values and uncertainties?
    if ukidss:
        T_mcmc, g_mcmc, m_mcmc, scale_mcmc, noise1_mcmc, noise2_mcmc, noise3_mcmc, noise4_mcmc, noise5_mcmc, noise6_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84],axis=0)))

    else:
        T_mcmc, g_mcmc, m_mcmc, scale_mcmc, noise1_mcmc, noise2_mcmc, noise3_mcmc, noise4_mcmc, noise5_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84],axis=0)))

    if plotCorner:
        triangle_samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

        if ukidss:
            fig = corner.corner(triangle_samples, 
                                labels=['Teff', 'Logg', 'Metal', 'Scale', 'Noise1', 'Noise2', 'Noise3', 'Noise4', 'Noise5', 'Noise6'],
                                truths=[T_mcmc[0], g_mcmc[0], m_mcmc[0], scale_mcmc[0], noise1_mcmc[0], noise2_mcmc[0], noise3_mcmc[0], noise4_mcmc[0], noise5_mcmc[0], noise6_mcmc[0]],
                                quantiles=[0.16, 0.84],
            #                  range = [.98, .98, .98]
                              )
        else:
            fig = corner.corner(triangle_samples, 
                                labels=['Teff', 'Logg', 'Metal', 'Scale', 'Noise1', 'Noise2', 'Noise3', 'Noise4', 'Noise5'],
                                truths=[T_mcmc[0], g_mcmc[0], m_mcmc[0], scale_mcmc[0], noise1_mcmc[0], noise2_mcmc[0], noise3_mcmc[0], noise4_mcmc[0], noise5_mcmc[0]],
                                quantiles=[0.16, 0.84],
            #                  range = [.98, .98, .98]
                              )

        fig.savefig('%s/corner_%s.png'%(plotsDir, obj), dpi=600, bbox_inches='tight')
    
    ffile = open(chainsDir +'/'+ saveFile, "w")
    #print(chainsDir +'/'+ saveFile)
    if ukidss:
        ffile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(obj, T_mcmc[0], T_mcmc[1], T_mcmc[2], g_mcmc[0], g_mcmc[1], g_mcmc[2], 
                                                          m_mcmc[0], m_mcmc[1], m_mcmc[2],
                                                          scale_mcmc[0], scale_mcmc[1], scale_mcmc[2], 
                                                          noise1_mcmc[0], noise1_mcmc[1], noise1_mcmc[2], 
                                                          noise2_mcmc[0], noise2_mcmc[1], noise2_mcmc[2], 
                                                          noise3_mcmc[0], noise3_mcmc[1], noise3_mcmc[2], 
                                                          noise4_mcmc[0], noise4_mcmc[1], noise4_mcmc[2], 
                                                          noise5_mcmc[0], noise5_mcmc[1], noise5_mcmc[2], 
                                                          noise6_mcmc[0], noise6_mcmc[1], noise6_mcmc[2]
                                                          ))

    else:
        ffile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(obj, T_mcmc[0], T_mcmc[1], T_mcmc[2], g_mcmc[0], g_mcmc[1], g_mcmc[2], 
                                                          m_mcmc[0], m_mcmc[1], m_mcmc[2],
                                                          scale_mcmc[0], scale_mcmc[1], scale_mcmc[2], 
                                                          noise1_mcmc[0], noise1_mcmc[1], noise1_mcmc[2], 
                                                          noise2_mcmc[0], noise2_mcmc[1], noise2_mcmc[2], 
                                                          noise3_mcmc[0], noise3_mcmc[1], noise3_mcmc[2], 
                                                          noise4_mcmc[0], noise4_mcmc[1], noise4_mcmc[2], 
                                                          noise5_mcmc[0], noise5_mcmc[1], noise5_mcmc[2]
                                                          ))
        
    ffile.close()
    


##################################################### The rest is for plotting

if MCMC == False:
    samplerchain = np.load('%s/%s.dat.npy'%(chainsDir, obj))       # Now apply the burn by cutting those steps out of the chain. 
    print(samplerchain.shape)
    ndim = samplerchain.shape[2]
    print(ndim)
    chain_burnt = samplerchain[:, burn:, :] 
    # Also flatten the chain to just a list of samples
    samples = chain_burnt.reshape((-1, ndim))
    
    # So that's all well and good, but what are my best-fit parameter values and uncertainties?
    if ukidss:
        T_mcmc, g_mcmc, m_mcmc, scale_mcmc, noise1_mcmc, noise2_mcmc, noise3_mcmc, noise4_mcmc, noise5_mcmc, noise6_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84],axis=0)))
    else:
        T_mcmc, g_mcmc, m_mcmc, scale_mcmc, noise1_mcmc, noise2_mcmc, noise3_mcmc, noise4_mcmc, noise5_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84],axis=0)))
print(T_mcmc)
print(g_mcmc)
print(m_mcmc)
'''
Temp          = 2463#T_mcmc[0]
Logg          = 4.4#g_mcmc[0]
Metal         = 0.4#m_mcmc[0]
ScalingFactor = -21#scale_mcmc[0]
'''
Temp          = T_mcmc[0]
Logg          = g_mcmc[0]
Metal         = m_mcmc[0]
ScalingFactor = scale_mcmc[0]


# Get the flux values and uncertanties
if binary[0] == 0: 
    aG, aGun   = AS.magtoflux(Gmag, 'SED', f0=Gmag, gaia_edr3='G', mag_uncert = Gmagu, flux_error=Gmagu)
else: 
    aG, aGun = 0, -9999

if binary[1] == 0: 
    aRP, aRPun   = AS.magtoflux(RPmag, 'SED', f0=RPmag, gaia_edr3='RP', mag_uncert = RPmagu, flux_error=RPmagu)
else: 
    aRP, aRPun = 0, -9999

if binary[2] == 0: 
    ai, aiun   = AS.magtoflux(imag,'SED', sdss='i', mag_uncert=imagu)
else: 
    ai, aiun = 0, -9999

if binary[3] == 0: 
    az, azun   = AS.magtoflux(zmag, 'SED', sdss='z', mag_uncert=zmagu)
else: 
    az, azun = 0, -9999

if binary[4] == 0: 
    aip, aipun   = AS.magtoflux(ipmag, 'SED', ps='i', mag_uncert=ipmagu)
else: 
    aip, aipun = 0, -9999

if binary[5] == 0: 
    azp, azpun   = AS.magtoflux(zpmag, 'SED', ps='z', mag_uncert=zpmagu)
else: 
    azp, azpun = 0, -9999

if binary[6] == 0: 
    ayp, aypun   = AS.magtoflux(ypmag, 'SED', ps='y', mag_uncert=ypmagu)
else: 
    ayp, aypun = 0, -9999

if binary[7] == 0: 
    aj, ajun   = AS.magtoflux(jmag, 'SED', mass='j', mag_uncert=jmagu)
else: 
    aj, ajun = 0, -9999

if binary[8] == 0: 
    ah, ahun   = AS.magtoflux(hmag, 'SED', mass='h', mag_uncert=hmagu)
else: 
    ah, ahun = 0, -9999

if binary[9] == 0: 
    ak, akun   = AS.magtoflux(kmag, 'SED', mass='k', mag_uncert=kmagu)
else: 
    ak, akun = 0, -9999

if binary[10] == 0: 
    aw1, aw1un   = AS.magtoflux(w1mag, 'SED', wise=1, powerlaw=-2, mag_uncert=w1magu)
else: 
    aw1, aw1un = 0, -9999

if binary[11] ==0: 
    aw2, aw2un = AS.magtoflux(w2mag, 'SED', wise=2, powerlaw=-2, mag_uncert=w2magu)
else: 
    aw2, aw2un = 0, -9999
'''
if binary[12] ==0: 
    aw3, aw3un = AS.magtoflux(w3mag, 'SED', wise=3, powerlaw=-2, mag_uncert=w3magu)
else: 
    aw3, aw3un = 0, -9999

if binary[13] ==0: 
    aw4, aw4un = AS.magtoflux(w4mag, 'SED', wise=4, powerlaw=-2, mag_uncert=w4magu)
else: 
    aw4, aw4un = 0, -9999
'''
if ukidss: 

    if binary[12] == 0: 
        ayu, ayuun   = AS.magtoflux(ymag_uk, 'SED', ukidss='y', mag_uncert=ymagu_uk)
    else: 
        ayu, ayuun = 0, -9999

    if binary[13] == 0: 
        aju, ajuun   = AS.magtoflux(jmag_uk, 'SED', ukidss='j', mag_uncert=jmagu_uk)
    else: 
        aju, ajuun = 0, -9999

    if binary[14] == 0: 
        ahu, ahuun   = AS.magtoflux(hmag_uk, 'SED', ukidss='h', mag_uncert=hmagu_uk)
    else: 
        ahu, ahuun = 0, -9999

    if binary[15] == 0: 
        aku, aukun   = AS.magtoflux(kmag_uk, 'SED', ukidss='k', mag_uncert=kmagu_uk)
    else: 
        aku, akuun = 0, -9999

print(aj, aju)


fluxesG   = np.ma.array([aG,aRP], mask=binary[:2], dtype=np.float)
fluxesGun = np.ma.array([aGun,aRPun], mask=binary[:2], dtype=np.float)
fluxesS   = np.ma.array([ai,az], mask=binary[2:4], dtype=np.float)
fluxesSun = np.ma.array([aiun,azun], mask=binary[2:4], dtype=np.float)
fluxesP   = np.ma.array([aip,azp,ayp], mask=binary[4:7], dtype=np.float)
fluxesPun = np.ma.array([aipun,azpun,aypun], mask=binary[4:7], dtype=np.float)
fluxes2   = np.ma.array([aj,ah,ak], mask=binary[7:10], dtype=np.float)
fluxes2un = np.ma.array([ajun,ahun,akun], mask=binary[7:10], dtype=np.float)
fluxesW   = np.ma.array([aw1,aw2], mask=binary[10:12], dtype=np.float)
fluxesWun = np.ma.array([aw1un,aw2un], mask=binary[10:12], dtype=np.float)
if ukidss:
    fluxesU   = np.ma.array([ayu,aju,ahu,aku], mask=binary[12:], dtype=np.float)
    fluxesUun = np.ma.array([ayuun,ajuun,ahuun,akuun], mask=binary[12:], dtype=np.float)

waves1 = np.ma.array([6390.7e-10, 7825.1e-10], mask=binary[:2], dtype=np.float)
waves2 = np.ma.array([7481e-10, 8931e-10], mask=binary[2:4], dtype=np.float)
waves3 = np.ma.array([7520e-10, 8660e-10, 9620e-10], mask=binary[4:7], dtype=np.float)
waves4 = np.ma.array([1.235e-6, 1.662e-6, 2.159e-6], mask=binary[7:10], dtype=np.float)
waves5 = np.ma.array([3.3526e-6, 4.6028e-6], mask=binary[10:12], dtype=np.float)
if ukidss:
    wavesU = np.ma.array([1.0305e-6, 1.2483e-6, 1.6313e-6, 2.2010e-6], mask=binary[12:], dtype=np.float)


# Interpolate the best-fit model model
#modelWaves, newModelFlux   = InterpModelLog(T_mcmc[0], g_mcmc[0])#, high=False)
#newModelFlux2              = newModelFlux * modelWaves  # ergs/s/cm^2
model  = smart.Model(teff=Temp, logg=Logg, metal=Metal, modelset=modelset, instrument=instrument)
newModelFlux  = model.flux
modelWaves    = model.wave
#print(np.max(modelWaves / 1e4), np.min(modelWaves / 1e4))
newModelFlux2 = model.flux * modelWaves # convert to ergs/s/cm^2
ScalingFactor = 10**ScalingFactor

#plt.loglog(modelWaves, newModelFlux2)
#plt.show()
#sys.exit()

# Interpolate the flux at bandpass wavelengths, this gives ergs/s/cm^2/ang
interG = np.interp(modelWaves, Gcurve[0], Gcurve[1])
Gsum   = np.trapz(interG * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(Gcurve[1], x=Gcurve[0])

interRP = np.interp(modelWaves, RPcurve[0], RPcurve[1])
RPsum   = np.trapz(interRP * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(RPcurve[1], x=RPcurve[0])

interi = np.interp(modelWaves, icurve[0], icurve[1])
isum   = np.trapz(interi * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(icurve[1], x=icurve[0])

interz = np.interp(modelWaves, zcurve[0], zcurve[1])
zsum   = np.trapz(interz * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(zcurve[1], x=zcurve[0])

interip = np.interp(modelWaves, ipcurve[0], ipcurve[1])
ipsum   = np.trapz(interip * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(ipcurve[1], x=ipcurve[0])

interzp = np.interp(modelWaves, zpcurve[0], zpcurve[1])
zpsum   = np.trapz(interzp * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(zpcurve[1], x=zpcurve[0])

interyp = np.interp(modelWaves, ypcurve[0], ypcurve[1])
ypsum   = np.trapz(interyp * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(ypcurve[1], x=ypcurve[0])

interJ = np.interp(modelWaves, jcurve[0], jcurve[1])
Jsum   = np.trapz(interJ * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(jcurve[1], x=jcurve[0])

interH = np.interp(modelWaves, hcurve[0], hcurve[1])
Hsum   = np.trapz(interH * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(hcurve[1], x=hcurve[0])

interK = np.interp(modelWaves, kcurve[0], kcurve[1])
Ksum   = np.trapz(interK * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(kcurve[1], x=kcurve[0])

interW1 = np.interp(modelWaves, w1curve[0], w1curve[1])
W1sum = np.trapz(interW1 * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(w1curve[1], x=w1curve[0])

interW2 = np.interp(modelWaves, w2curve[0], w2curve[1])
W2sum = np.trapz(interW2 * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(w2curve[1], x=w2curve[0])

#interW3 = np.interp(modelWaves, w3curve[0], w3curve[1])
#W3sum = np.trapz(interW3 * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(w3curve[1], x=w3curve[0])

#interW4 = np.interp(modelWaves, w4curve[0], w4curve[1])
#W4sum = np.trapz(interW4 * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(w4curve[1], x=w4curve[0])

if ukidss:
    intery = np.interp(modelWaves, ycurve_uk[0], ycurve_uk[1])
    yusum   = np.trapz(intery * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(ycurve_uk[1], x=ycurve_uk[0])

    interJ = np.interp(modelWaves, jcurve_uk[0], jcurve_uk[1])
    Jusum   = np.trapz(interJ * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(jcurve_uk[1], x=jcurve_uk[0])

    interH = np.interp(modelWaves, hcurve_uk[0], hcurve_uk[1])
    Husum   = np.trapz(interH * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(hcurve_uk[1], x=hcurve_uk[0])

    interK = np.interp(modelWaves, kcurve_uk[0], kcurve_uk[1])
    Kusum   = np.trapz(interK * newModelFlux * ScalingFactor, x=modelWaves) / np.trapz(kcurve_uk[1], x=kcurve_uk[0])


wavesTG = np.ma.array([6390.7, 7825.1], mask=binary[:2], dtype=np.float)
TestFluxesGAIA = np.ma.array([Gsum, RPsum], mask=binary[:2], dtype=np.float) * wavesTG

wavesTS = np.ma.array([7481., 8931], mask=binary[2:4], dtype=np.float)
TestFluxesSDSS = np.ma.array([isum, zsum], mask=binary[2:4], dtype=np.float)  * wavesTS

wavesTP = np.ma.array([7520., 8660, 9620], mask=binary[4:7], dtype=np.float)
TestFluxesPan = np.ma.array([ipsum, zpsum, ypsum], mask=binary[4:7], dtype=np.float)  * wavesTP

wavesT2 = np.ma.array([1.235e4, 1.662e4, 2.159e4], mask=binary[7:10], dtype=np.float)
TestFluxes2MASS = np.ma.array([Jsum, Hsum, Ksum], mask=binary[7:10], dtype=np.float)  * wavesT2

#wavesTW = np.ma.array([3.3526e4, 4.6028e4, 11.5608e4, 22.0883e4], mask=binary[9:11], dtype=np.float)
#TestFluxesWISE = np.ma.array([W1sum, W2sum, W3sum, W4sum], mask=binary[9:11], dtype=np.float)  * wavesTW

wavesTW = np.ma.array([3.3526e4, 4.6028e4], mask=binary[10:12], dtype=np.float)
TestFluxesWISE = np.ma.array([W1sum, W2sum], mask=binary[10:12], dtype=np.float)  * wavesTW

if ukidss:
    wavesUK = np.ma.array([1.0305e4, 1.2483e4, 1.6313e4, 2.2010e4], mask=binary[12:], dtype=np.float)
    TestFluxesUK = np.ma.array([yusum, Jusum, Husum, Kusum], mask=binary[12:], dtype=np.float)  * wavesUK


######################################## Create the figure    

capsize = 7
markersize = 6

fig = plt.figure(1, figsize=(7.1, 4))
ax = fig.add_subplot(111)


ax.errorbar(waves1.compressed() / 1e-6, fluxesG.compressed(), yerr = fluxesGun.compressed(), fmt='s', markerfacecolor='0.1', ms=markersize, markeredgecolor='0.1', ecolor='0.1', capsize = capsize, zorder=10, label=r"\textit{Gaia}")
ax.errorbar(waves2.compressed() / 1e-6, fluxesS.compressed(), yerr = fluxesSun.compressed(), fmt='o', markerfacecolor='0.3', ms=markersize, markeredgecolor='0.3', ecolor='0.3', capsize = capsize, zorder=10, label=r"SDSS")
ax.errorbar(waves3.compressed() / 1e-6, fluxesP.compressed(), yerr = fluxesPun.compressed(), fmt='D', markerfacecolor='0.5', ms=markersize, markeredgecolor='0.5', ecolor='0.5', capsize = capsize, zorder=10, label=r"Pan-STARRS")
ax.errorbar(waves4.compressed() / 1e-6, fluxes2.compressed(), yerr = fluxes2un.compressed(), fmt='^', markerfacecolor='0.7', ms=markersize, markeredgecolor='0.7', ecolor='0.7', capsize = capsize, zorder=10, label=r"2MASS")
ax.errorbar(waves5.compressed() / 1e-6, fluxesW.compressed(), yerr = fluxesWun.compressed(), fmt='>', markerfacecolor='0.9', ms=markersize, markeredgecolor='0.9', ecolor='0.9', capsize = capsize, zorder=10, label=r"\textit{WISE}")
if ukidss:
    ax.errorbar(wavesU.compressed() / 1e-6, fluxesU.compressed(), yerr = fluxesUun.compressed(), fmt='<', markerfacecolor='C9', ms=markersize, markeredgecolor='C9', ecolor='C9', capsize = capsize, zorder=10, label=r"UKIDSS")

ax.scatter(wavesTG.compressed() * 1e-4, TestFluxesGAIA.compressed(), marker='s', facecolors='none', s=50, zorder=6, edgecolors='r')
ax.scatter(wavesTS.compressed() * 1e-4, TestFluxesSDSS.compressed(), marker='o', facecolors='none', s=50, zorder=6, edgecolors='r')
ax.scatter(wavesTP.compressed() * 1e-4, TestFluxesPan.compressed(), marker='D', facecolors='none', s=50, zorder=6, edgecolors='r')
ax.scatter(wavesT2.compressed() * 1e-4, TestFluxes2MASS.compressed(), marker='^', facecolors='none', s=50, zorder=6, edgecolors='r')
ax.scatter(wavesTW.compressed() * 1e-4, TestFluxesWISE.compressed(), marker='>', facecolors='none', s=50, zorder=6, edgecolors='r')
if ukidss:
    ax.scatter(wavesUK.compressed() * 1e-4, TestFluxesUK.compressed(), marker='<', facecolors='none', s=50, zorder=6, edgecolors='r')


ax.set_xscale('log')
ax.set_yscale('log')

# Get the y-axis limits for later use
ymin, ymax = ax.get_ylim()


##### Get the spectra
t1 = Table.read('/home/ctheissen/Downloads/spex_prism_0350+1818_151003.csv')
t1 = Table.read('/home/ctheissen/Downloads/spex_prism_0350+1818_151006.csv')
print(t1.colnames)
ax.plot(t1['wave (micron)'], t1['flux (erg/micron/s/cm^2)'] * t1['wave (micron)'] * 9e3, alpha=0.5, label='SpeX', c='C0')
#ax.plot(t1['wave (micron)'], t1['flux (erg/micron/s/cm^2)'] * 5e3, alpha=0.5, label='SpeX')
#sys.exit()
##### Get the spectra

# Now we plot the model
# Downsample the flux
#newspec  = AS.rebin_spec(modelWaves, newModelFlux2, newwaves)
#ax.plot(newwaves / 1e4, newspec * ScalingFactor, 'r', zorder = -100, alpha=0.3, label=r'BT-Settl (%s K)'%int(np.around(Temp)))
ax.plot(modelWaves / 1e4, newModelFlux2 * ScalingFactor, 'r', zorder = -100, alpha=0.3, label=r'%s (%s K)'%(modelset, int(np.around(Temp))))

ax.legend(frameon=False, scatterpoints=1, numpoints=1)

# Axes Labels
ax.set_ylim(ymin, ymax)
ax.set_xlim(.3, 40)

#fig.savefig("%s/SED_%s.pdf"%(plotsDir, obj), dpi=600, bbox_inches='tight')
fig.savefig("%s/SED_%s.png"%(plotsDir, obj), dpi=600, bbox_inches='tight')

if TEST: 
    plt.show()
    sys.exit()

plt.close('all')

gc.collect() # Collect the garbage

    

sys.exit()
########################################################################################## Radius and Luminosity




print(tableG['Plx'].data[0], tableG['e_Plx'].data[0])

# Load the sampler
samplerchain = np.load('%s/%s.dat.npy'%(chainsDir, obj))

# Grab the chains
nwalkers, step, ndim = samplerchain.shape


# Now apply the burn by cutting those steps out of the chain. 
chain_burnt = samplerchain[:, burn:, :] 
# Also flatten the chain to just a list of samples
samples = chain_burnt.reshape((-1, ndim))

# Pick a random Temp, Logg. Metallicity and create a model
FLUXES = []
SCALES = []
TEFFS  = []
# Pick a random Temp, Logg. Metallicity and create a model
print('Getting Random Model')
counter1 = 1
for i in np.random.randint(0, high=samples.shape[0], size=size):
  print('%s/%s'%(counter1, size), end='\r')
  #model  = smart.Model(teff=samples[i,0], logg=samples[i,1], metal=samples[i,2], modelset=modelset, instrument=instrument)
  model  = smart.Model(teff=samples[i,0], logg=samples[i,1], metal=0, modelset=modelset, instrument=instrument)
  fluxes = model.flux * 10**samples[i,2]# * outputfluxunits
  waves  = model.wave * outputwaveunits
  FLUXES.append(fluxes)
  SCALES.append(10**samples[i,2])
  TEFFS.append(samples[i,0])
  counter1+=1
  
distances = 1000/np.random.normal(tableG['Plx'].data[0], tableG['e_Plx'].data[0], size) * u.pc

radius2  = np.sqrt(np.array(SCALES)) * distances.to(u.R_sun)

FLUXES = np.array(FLUXES) * outputfluxunits
print('Luminosity')
luminosity = np.array([np.trapz(fluxdist[0], x=waves).value * fluxdist[1].to(u.cm).value**2 * 4 * np.pi for fluxdist in zip(FLUXES, distances)]) * u.erg/u.s # * distance.to(u.cm)**2 * 4 * np.pi 
  #print(np.trapz(fluxes, x=waves))
  #print(distances[i].to(u.cm))
  #luminosity = np.trapz(fluxes, x=waves) * distances[i].to(u.cm)**2 * 4 * np.pi 
#print(luminosity)
  #lumins.append(luminosity)
#luminosity = np.array(lumins)
#print(luminosity)
#print(sigma_sb.to(u.erg / (u.s*u.K**4*u.m**2)))
print('Radius')
radius     = np.sqrt( luminosity / ( 4*np.pi*sigma_sb.to(u.erg / (u.s*u.K**4*u.m**2))*(TEFFS*u.K)**4 ) ).to(u.R_sun)
print('Object done in %s seconds'%(time.time()-start))

print(np.median(luminosity.value), np.mean(luminosity.value), np.std(luminosity.value))
print(np.median(radius.value), np.mean(radius.value), np.std(radius.value))
print(np.median(radius2.value), np.mean(radius2.value), np.std(radius2.value))
print('Luminosity: %0.2f +%0.2f - %0.2f'%(np.median(luminosity.to(u.L_sun).value), np.percentile(luminosity.to(u.L_sun).value, 84)-np.median(luminosity.to(u.L_sun).value), np.median(luminosity.to(u.L_sun).value)-np.percentile(luminosity.to(u.L_sun).value, 16)))
print('Radius1: %0.2f +%0.2f - %0.2f'%(np.median(radius.value), np.percentile(radius.value, 84) - np.median(radius.value), np.median(radius.value) - np.percentile(radius.value, 16)))
print('Radius2: %0.2f +%0.2f - %0.2f'%(np.median(radius2.value), np.percentile(radius2.value, 84) - np.median(radius2.value), np.median(radius2.value) - np.percentile(radius2.value, 16)))

if TEST: sys.exit()

print(np.median(luminosity.value), np.mean(luminosity.value), np.std(luminosity.value))
print(np.median(radius.value), np.mean(radius.value), np.std(radius.value))
#plt.hist(radius.value, bins = int(np.sqrt(len(radius.value))), histtype='step')
#plt.xlabel(r'$R$ ($R_\odot$)')
#plt.show()
#sys.exit()

## So that's all well and good, but what are my best-fit parameter values and uncertainties?
#T_mcmc, g_mcmc, m_mcmc, scale_mcmc, noise1_mcmc, noise2_mcmc, noise3_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84],axis=0)))





np.savez_compressed('%s/compressed_%s'%(writeDir2, obj), luminosities=luminosity.data, radii=radius.data, radii2=radius2.data)
#sys.exit()

ffile = open('%s/Luminosity_Radius_%s.txt'%(writeDir, obj), "w")
ffile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%('Luminosity', 'Luminosity+', 'Luminosity-', 'LuminosityS', 'LuminosityS+', 'LuminosityS-', 'Radius', 'Radius+', 'Radius-', 'Radius2', 'Radius2+', 'Radius2-'))
ffile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(np.median(luminosity.value), np.percentile(luminosity.value, 84)-np.median(luminosity.value), np.median(luminosity.value)-np.percentile(luminosity.value, 16),
                                                     np.median(luminosity.to(u.L_sun).value), np.percentile(luminosity.to(u.L_sun).value, 84)-np.median(luminosity.to(u.L_sun).value), np.median(luminosity.to(u.L_sun).value)-np.percentile(luminosity.to(u.L_sun).value, 16),
                                                     np.median(radius.value), np.percentile(radius.value, 84)-np.median(radius.value), np.median(radius.value)-np.percentile(radius.value, 16),
                                                     np.median(radius2.value), np.percentile(radius2.value, 84)-np.median(radius2.value), np.median(radius2.value)-np.percentile(radius2.value, 16) ) ) 
ffile.close()
del FLUXES
del luminosity
del radius
del distances
gc.collect()
    











    
# Let me know it's done
print("Done in %2.2f seconds"% (time.time() - start))
print("Done in %2.2f minutes"% ((time.time() - start)/60.0))
print("Done in %2.2f hours"% ((time.time() - start)/3600.0))
print("DONE!")
