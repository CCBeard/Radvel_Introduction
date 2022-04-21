import numpy as np
import pandas as pd
import os
import radvel
from radvel.plot import orbit_plots, mcmc_plots
from scipy import optimize
import matplotlib.pyplot as plt

#Read in the data
data = pd.read_csv('../../Data/AUMIC_rv.csv', sep=',')

#Define the usual stuff
t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
tel_arr = np.array(data.tel)
telgrps = data.groupby('tel').groups
instnames = telgrps.keys()
#can't insert keys into the likelihood
#have to make a different version of instnames that's an nparray
telnames = []
for inst in instnames:
    telnames.append(inst)
telnames = np.array(telnames)



#define the hyperparameter names
hnames = {}
for inst in instnames:
    hnames[inst]=['gp_amp_'+inst,
        'gp_explength',
            'gp_perlength',
             'gp_per']




P1 = 8.4629991 # orbital period for planet b [days]
Tc1 = 2458330.39046 #Time of inferior conjunction for planet b (BJD)
P2 = 18.858991 # orbital period for planet c [days]
Tc2 = 2458342.2243 #Time of inferior conjunction for planet c (BJD)

#adopting free parameters identically to those used in Cale et al. 2021
nplanets=2
params = radvel.Parameters(nplanets,basis='per tc e w k')

params['per1'] = radvel.Parameter(value=P1,vary=False)
params['tc1'] = radvel.Parameter(value=Tc1,vary=False)
params['e1'] = radvel.Parameter(value=0.187,vary=True)
params['w1'] = radvel.Parameter(value=1.5452,vary=True)
params['k1'] = radvel.Parameter(value=10.21)

params['per2'] = radvel.Parameter(value=P2,vary=False)
params['tc2'] = radvel.Parameter(value=Tc2,vary=False)
params['e2'] = radvel.Parameter(value=0.,vary=False)
params['w2'] = radvel.Parameter(value=np.pi,vary=False)
params['k2'] = radvel.Parameter(value=3.62)

params['dvdt'] = radvel.Parameter(value=0.0,vary=False)
params['curv'] = radvel.Parameter(value=0.0,vary=False)

#add starting values for the hyperparameters
params['gp_explength'] = radvel.Parameter(value=100, vary=False) #This is eta_t in Cale et al. 2021
params['gp_perlength'] = radvel.Parameter(value=0.28, vary=False) #This is eta_4 in radvel, eta_l in Cale et al. 2021
params['gp_per'] = radvel.Parameter(value=4.836, vary=True) #Stellar rotation period.  Units: daysts: (m/s)**2.
for inst in instnames:
    params['gp_amp_'+inst] = radvel.Parameter(value=np.std(vel[tel_arr==inst]), vary=True)



gpmodel = radvel.model.RVModel(params) #put this into an RV model context



likes = []
def initialize(tel_suffix):
    # Instantiate a separate likelihood object for each instrument.
	indices = telgrps[tel_suffix]
	like = radvel.likelihood.GPLikelihood(gpmodel, t[indices], vel[indices],
	errvel[indices], hnames[tel_suffix], suffix='_'+tel_suffix,
	kernel_name="QuasiPer")

	params['gamma_'+tel_suffix] = radvel.Parameter(value=1.0)
	if tel_suffix == 'hires':
		params['jit_'+tel_suffix] = radvel.Parameter(value=3.0, vary=False)
	else:
		params['jit_'+tel_suffix] = radvel.Parameter(value=0.0, vary=False)
	likes.append(like)

#note that we're making a likelihood for each instrument, UNLIKE in the Chromatic case
for tel in instnames:
    initialize(tel)


gplike = radvel.likelihood.CompositeLikelihood(likes)


post = radvel.posterior.Posterior(gplike)

#only a few planet parameters are varying. Priors taken from Cale et al. 2021
post.priors += [radvel.prior.PositiveKPrior(nplanets)]
post.priors += [radvel.prior.Gaussian('e1', 0.189, 0.04)]
post.priors += [radvel.prior.Gaussian('w1', 1.5449655, 0.004)]

#Instrumental priors, taken from Cale et al. 2021
post.priors += [radvel.prior.HardBounds('gamma_hires', -300.0, 300.0)]# * [radvel.prior.Gaussian('gamma_hires', 0, 100.0)]
post.priors += [radvel.prior.HardBounds('gamma_carmenesvis', -300.0, 300.0)]# * [radvel.prior.Gaussian('gamma_carmenesvis', 0, 100.0)]
post.priors += [radvel.prior.HardBounds('gamma_carmenesnir', -300.0, 300.0)]# * [radvel.prior.Gaussian('gamma_carmenesnir', 0, 100.0)]
post.priors += [radvel.prior.HardBounds('gamma_tres', -300.0, 300.0)]# * [radvel.prior.Gaussian('gamma_tres', 0, 100.0)]
post.priors += [radvel.prior.HardBounds('gamma_spirou', -300.0, 300.0)]# * [radvel.prior.Gaussian('gamma_spirou', 0, 100.0)]
post.priors += [radvel.prior.HardBounds('gamma_ishell', -300.0, 300.0)]# * [radvel.prior.Gaussian('gamma_ishell', 0, 100.0)]



#Two of the hyperparameters are fixed. We add the other GP hyperparameter priors here
for inst in instnames:
    post.priors += [radvel.prior.Gaussian('gp_amp_'+inst, np.std(vel[tel_arr==inst]), 30.)]
post.priors += [radvel.prior.Gaussian('gp_per', 4.836, 0.001)]



print(post)

#Optimize before starting MCMC chains
post = radvel.fitting.maxlike_fitting(post, verbose=True, method='Nelder-Mead')


nwalk=150 #150 independent walkers
#Warning, this GP is slow. This might take a while
chains = radvel.mcmc(post,nrun=15000,ensembles=3,nwalkers=nwalk, serial=True)

#print the median posterior estimates
quants = chains.quantile([0.159, 0.5, 0.841]) # median & 1sigma limits of posterior distribu
for par in post.params.keys():
	if post.params[par].vary:
		med = quants[par][0.5]
		high = quants[par][0.841] - med
		low = med - quants[par][0.159]
		err = np.mean([high,low])
		err = radvel.utils.round_sig(err)
		med, err, errhigh = radvel.utils.sigfig(med, err)
		print('{} : {} +/- {}'.format(par, med, err))


GPPlot = orbit_plots.GPMultipanelPlot(
post,
subtract_gp_mean_model=False,
plot_likelihoods_separately=False,
subtract_orbit_model=False,
telnames = instnames,
telfmts={'spirou':dict(fmt='o',label='SPIROU',color='gray',markersize=7,markeredgecolor='black', mew=1., zorder=1.),
'tres':dict(fmt='o',label='TRES',color='gold', markersize=7, markeredgecolor='black', mew = 1., zorder=1.),
'ishell':dict(fmt='o', label='iSHELL', color='indianred', markersize=7, markeredgecolor='black', mew=1., zorder=1.),
'carmenesvis':dict(fmt='o', label='CARMENES-VIS', color='green', markersize=7, markeredgecolor='black', mew=1., zorder=1.),
'carmenesnir':dict(fmt='o', label='CARMENES-NIR', color='violet', markersize=7, markeredgecolor='black', mew=1., zorder=1.),
'hires':dict(fmt='o', label='HIRES', color='dodgerblue', markersize=7, markeredgecolor='black', mew=1., zorder=1.),},
yscale_sigma=4.,
)
GPPlot.plot_multipanel()
plt.show()






#make a corner plot
Corner = mcmc_plots.CornerPlot(post,chains)
Corner.plot()
