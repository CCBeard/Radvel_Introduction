'''
######################################################################
Customized RadVel input file for:
    191939 (Atmospheric target TOI 1339.01, n.b. is a 4+ planet system)
Created 2020-07-14
by Jack Lubin (based on earlier work by Joey, Ryan, Ian, and Lauren)
Goal: A setup file that allows for 4 planets + trend, all fixed at circular
######################################################################
'''
import radvel
import numpy as np
import pandas as pd

starname = '191939'
nplanets = 4
fitting_basis = 'per tc secosw sesinw k'
bjd0 = 0.
planet_letters = {1:'b', 2:'c', 3:'d', 4:'e'}

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets, basis='per tc e w k', planet_letters=planet_letters) # initialize Parameters object

#TESS known planets - Period, Tc taken from Badenas-Agusti et al. 2020
#eccentricity and omega fixed at zero for inner three planets
#initial K values are educated guesses based on S16_SPOC TKS target vetting spreadsheet

#Planet b
anybasis_params['per1'] = radvel.Parameter(value=8.880290)
anybasis_params['tc1'] = radvel.Parameter(value=2458715.356133)
anybasis_params['e1'] = radvel.Parameter(value=0.0)
anybasis_params['w1'] = radvel.Parameter(value=0.000000)
anybasis_params['k1'] = radvel.Parameter(value=3.850000)
#Planet c
anybasis_params['per2'] = radvel.Parameter(value=28.580507)
anybasis_params['tc2'] = radvel.Parameter(value=2458726.053366)
anybasis_params['e2'] = radvel.Parameter(value=0.0)
anybasis_params['w2'] = radvel.Parameter(value=0.000000)
anybasis_params['k2'] = radvel.Parameter(value=2.380000)
#Planet d
anybasis_params['per3'] = radvel.Parameter(value=38.352445)
anybasis_params['tc3'] = radvel.Parameter(value=2458743.551787)
anybasis_params['e3'] = radvel.Parameter(value=0.0)
anybasis_params['w3'] = radvel.Parameter(value=0.000000)
anybasis_params['k3'] = radvel.Parameter(value=2.38)

# Planet e - No known transit
anybasis_params['per4'] = radvel.Parameter(value=101.5)
anybasis_params['tc4'] = radvel.Parameter(value=2459050.0)
anybasis_params['e4'] = radvel.Parameter(value=0.0)
anybasis_params['w4'] = radvel.Parameter(value=0.000000)
anybasis_params['k4'] = radvel.Parameter(value=20.0)

# very important to include a time_base when using trend and curvature!!
time_base = 2458847.780463
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
anybasis_params['curv'] = radvel.Parameter(value=0.0)

#import data
data = pd.read_csv('HD191939_TKS_IX_data.csv',
dtype={'time': np.float64, 'mnvel': np.float64, 'err': np.float64, 'tel': str})
bin_t, bin_vel, bin_err, bin_tel = radvel.utils.bintels(data['time'].values, data['mnvel'].values, data['errvel'].values, data['tel'].values, binsize=0.1)
data = pd.DataFrame([], columns=['time', 'mnvel', 'errvel', 'tel'])
data['time'] = bin_t
data['mnvel'] = bin_vel
data['errvel'] = bin_err
data['tel'] = bin_tel

instnames = ['APF', 'HIRES']
ntels = len(instnames)
anybasis_params['gamma_APF'] = radvel.Parameter(value=0.0, vary=False, linear=True)
anybasis_params['jit_APF'] = radvel.Parameter(value=1.0)
anybasis_params['gamma_HIRES'] = radvel.Parameter(value=0.0, vary=False, linear=True)
anybasis_params['jit_HIRES'] = radvel.Parameter(value=1.0)

params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)
mod = radvel.RVModel(params, time_base=time_base) # Don't forget to include time_base!!

# Parameters fixed from Badenas-Agusti et al. 2020
# Easier to fix the transiting planets' periods and conjunction times rather than fit for them
# It just adds more parameters to fit, and they are paramters we know from photometry
# But feel free to let them vary and try out different priors
mod.params['per1'].vary = False
mod.params['tc1'].vary  = False
mod.params['per2'].vary = False
mod.params['tc2'].vary  = False
mod.params['per3'].vary = False
mod.params['tc3'].vary  = False
mod.params['per4'].vary = True
mod.params['tc4'].vary  = True

# Feel free to try eccentric fits
mod.params['secosw1'].vary = False # Force a circular fit
mod.params['sesinw1'].vary = False
mod.params['secosw2'].vary = False # Force a circular fit
mod.params['sesinw2'].vary = False
mod.params['secosw3'].vary = False # Force a circular fit
mod.params['sesinw3'].vary = False
mod.params['secosw4'].vary = False # Force a circular fit
mod.params['sesinw4'].vary = False

mod.params['dvdt'].vary = True    # allow linear trend
mod.params['curv'].vary = True    # allow curvature

mod.params['jit_HIRES'].vary = True
mod.params['jit_APF'].vary = True

priors = [
          #radvel.prior.PositiveKPrior(nplanets), #-Jack turned off prior on advice from Andrew, 12/8/20
          radvel.prior.HardBounds('per4', 1.0, 1000.0),
          radvel.prior.HardBounds('tc4', params['tc4'].value - 50.0, params['tc4'].value + 50.0),
          radvel.prior.HardBounds('jit_HIRES', 0.0, 10.0),
          radvel.prior.HardBounds('jit_APF', 0.0, 10.0),
          radvel.prior.HardBounds('dvdt', -1.0, 1.0),
          radvel.prior.HardBounds('curv', -0.1, 0.1)
         ]

stellar = dict(mstar=0.807, mstar_err=0.029)
planet = dict(rp1=3.39, rp_err1=0.07, rp2=3.08, rp_err2=0.07, rp3=3.04, rp_err3=0.07) # From Rae's transit modelling
